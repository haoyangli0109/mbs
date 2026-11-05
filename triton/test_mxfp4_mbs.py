"""Correctness of the Triton MXFP4/MBS kernels.

  HIP_VISIBLE_DEVICES=4 python test_mxfp4_mbs.py

Checks, in order:
  1. use_mbs=False quantization == aiter's MXFP4 quantizer, bit for bit
     (row-major and shuffled scales);
  2. MBS-S / MBS-D(16) / MBS-D(256) quantization == the torch reference here,
     bit for bit (MBS-D may differ on exact SSE ties, reported);
  3. the dequantized tensors agree with the paper-simulation reference
     /tmp/mbs_paper_sim/fp4sim.py (mode="even") in QSNR;
  4. GEMM == dequant(A) @ dequant(B)^T (fp32) for every MBS side option,
     and the MBS-off GEMM == aiter's asm gemm_a4w4.
"""

import sys

import torch

import mxfp4_mbs as mm

torch.manual_seed(0)
dev = "cuda"


def heavy_tailed(rows, cols, dtype=torch.bfloat16):
    """Student-t(3) rows with per-row and per-column magnitude spread, like LLM
    activations (a few large channels)."""
    x = torch.distributions.StudentT(3.0).sample((rows, cols)).to(dev)
    x = x * torch.exp(torch.randn(rows, 1, device=dev) * 0.5)
    x = x * torch.exp(torch.randn(1, cols, device=dev) * 0.7)
    return x.to(dtype)


def qsnr(ref, q):
    ref = ref.double()
    return (10 * torch.log10(ref.pow(2).sum() / (ref - q.double()).pow(2).sum())).item()


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        check.failed = True


check.failed = False


def test_plain_vs_aiter():
    from aiter.utility.fp4_utils import dynamic_mxfp4_quant

    for M, K in ((1, 256), (7, 4096), (100, 12288), (1000, 4096)):
        x = heavy_tailed(M, K)
        q_ref, s_ref = dynamic_mxfp4_quant(x, shuffle=False)
        q, s, m = mm.mbs_quant(x, use_mbs=False)
        check(f"plain quant == aiter M={M} K={K}",
              torch.equal(q, q_ref.view(torch.uint8)) and torch.equal(s, s_ref.view(torch.uint8)))
        _, s_ref_sh = dynamic_mxfp4_quant(x, shuffle=True)
        _, s_sh, _ = mm.mbs_quant(x, use_mbs=False, shuffle_scale=True)
        s_ref_sh = s_ref_sh.view(torch.uint8)
        valid = (M + 31) // 32 * 32
        rows = s_ref_sh.shape[0]
        same = torch.equal(s_sh.view(-1)[: valid * s_sh.shape[1]],
                           s_ref_sh.reshape(rows, -1).view(-1)[: valid * s_sh.shape[1]])
        check(f"shuffled scale == aiter M={M} K={K}", same)


def test_mbs_vs_ref():
    for ncand in (1, 16, 256):
        for M, K in ((5, 512), (64, 4096), (300, 1024)):
            x = heavy_tailed(M, K)
            q, s, m = mm.mbs_quant(x, mb=128, ncand=ncand)
            q_r, s_r, m_r = mm.mbs_quant_ref(x, mb=128, ncand=ncand)
            m_same = (m == m_r).float().mean().item()
            if ncand == 1:
                ok = torch.equal(q, q_r) and torch.equal(s, s_r) and torch.equal(m, m_r)
            else:
                # summation order differs from torch, so exact SSE ties may break differently;
                # every mismatching block must have the same SSE within rounding
                xm = x.float().reshape(M, K // 128, 128)
                err = (mm.mbs_dequant(q, s, m) - x.float()).reshape(M, -1, 128).pow(2).sum(-1)
                err_r = (mm.mbs_dequant(q_r, s_r, m_r) - x.float()).reshape(M, -1, 128).pow(2).sum(-1)
                rel = ((err - err_r).abs() / err_r.clamp(min=1e-30)).max().item()
                ok = m_same > 0.999 and rel < 1e-5
                del xm
            check(f"MBS quant ncand={ncand} == ref M={M} K={K}", ok, f"(m agree {m_same:.5f})")
        for mb in (256,):
            x = heavy_tailed(33, 2048)
            q, s, m = mm.mbs_quant(x, mb=mb, ncand=1)
            q_r, s_r, m_r = mm.mbs_quant_ref(x, mb=mb, ncand=1)
            if ncand == 1:
                check(f"MBS-S mb={mb} == ref", torch.equal(q, q_r) and torch.equal(m, m_r))


def test_vs_paper_sim():
    sys.path.insert(0, "/tmp/mbs_paper_sim")
    try:
        import fp4sim
    except ImportError:
        print("[SKIP] /tmp/mbs_paper_sim/fp4sim.py not found")
        return
    x = heavy_tailed(512, 4096).float()
    for name, ours, sim in (
        ("MXFP4 even", mm.mbs_quant(x, use_mbs=False), fp4sim.mx_qdq(x, 32, "even")),
        ("MBS-S @32", mm.mbs_quant(x, ncand=1), fp4sim.mbs_qdq(x, 32, 128, "even")),
        ("MBS-D @32", mm.mbs_quant(x, ncand=16), fp4sim.mbs_qdq(x, 32, 128, "even", dynamic=True)),
    ):
        q, s, m = ours
        d = mm.mbs_dequant(q, s, m if m is not None else torch.zeros(512, 32, dtype=torch.uint8, device=dev))
        a, b = qsnr(x, d), qsnr(x, sim)
        same = (d == sim).float().mean().item()
        check(f"{name} vs fp4sim", abs(a - b) < 0.01, f"QSNR {a:.3f} vs {b:.3f} dB, elements equal {same:.5f}")
    d256 = mm.mbs_dequant(*mm.mbs_quant(x, ncand=256))
    print(f"       MBS-D(256 candidates) QSNR {qsnr(x, d256):.3f} dB")


def test_gemm():
    from aiter.ops.shuffle import shuffle_weight

    w = torch.randn(256, 512, device=dev).to(torch.bfloat16)
    q, _, _ = mm.mbs_quant(w, use_mbs=False)
    check("preshuffle_b == aiter shuffle_weight(16, 16)",
          torch.equal(mm.preshuffle_b(q), shuffle_weight(q.view(torch.float4_e2m1fn_x2), layout=(16, 16)).view(torch.uint8)))

    shapes = ((1, 256, 512), (17, 384, 1024), (32, 256, 1024), (128, 128, 1024), (333, 1008, 2048), (1024, 4096, 4096))
    for M, N, K in shapes:
        x = heavy_tailed(M, K)
        w = torch.randn(N, K, device=dev).to(torch.bfloat16)
        for mb in (128, 256, 512, 1024):
            if K % mb:
                continue
            qa, sa, ma = mm.mbs_quant(x, mb=mb, ncand=1)
            qb, sb, mb_w = mm.mbs_quant(w, mb=mb, ncand=16)
            A_full = mm.quant_activation(x, mb=mb, keep_m=True)
            B_full = mm.prepare_weight(w, mb=mb, ncand=16)
            if mb == 128:
                # the quantizer's ratios (recomputed from the previous block) == host ratios from m
                r_host, sl_host = mm.mbs_ratios(A_full["m"])
                same = torch.equal(A_full["ratio"][:, :M], r_host) and torch.equal(A_full["sigma_last"], sl_host)
                check(f"activation ratio/sigma_last == from m M={M} K={K}", same)
            for a_mbs, b_mbs in ((True, True), (True, False), (False, True), (False, False)):
                if mb != 128 and not (a_mbs and b_mbs):
                    continue
                ref = mm.mbs_gemm_ref((qa, sa, ma if a_mbs else torch.zeros_like(ma)),
                                      (qb, sb, mb_w if b_mbs else torch.zeros_like(mb_w)), mb=mb)
                A = dict(A_full, ratio=A_full["ratio"] if a_mbs else None)
                B = dict(B_full, ratio=B_full["ratio"] if b_mbs else None)
                cfg = dict(BLOCK_M=32 if M >= 32 else 16, BLOCK_N=64, BLOCK_K=256 if K % 512 else 512)
                if max(mb, cfg["BLOCK_K"]) > K:
                    cfg["BLOCK_K"] = 256
                out = mm.mbs_gemm(A, B, mb=mb, out_dtype=torch.float32, config=cfg)
                rel = ((out - ref).norm() / ref.norm()).item()
                check(f"GEMM M={M} N={N} K={K} mb={mb} mbs A={a_mbs:d} B={b_mbs:d}", rel < 2e-5, f"fp32 rel err {rel:.2e}")
            # bf16 output and the large default tile
            out = mm.mbs_gemm(A_full, B_full, mb=mb, config=dict(BLOCK_M=128 if M >= 32 else 16, BLOCK_N=128,
                                                                 BLOCK_K=256 if K % 512 else 512))
            ref = mm.mbs_gemm_ref((qa, sa, ma), (qb, sb, mb_w), mb=mb)
            rel = ((out.float() - ref).norm() / ref.norm()).item()
            check(f"GEMM bf16 M={M} N={N} K={K} mb={mb}", rel < 4e-3, f"rel err {rel:.2e}")


def test_plain_gemm_vs_asm():
    import aiter

    M, N, K = 512, 2048, 4096
    x = heavy_tailed(M, K)
    w = torch.randn(N, K, device=dev).to(torch.bfloat16)
    A = mm.quant_activation(x, use_mbs=False)
    B = mm.prepare_weight(w, use_mbs=False)
    y = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
    ref = aiter.gemm_a4w4(A["q"].view(torch.float4_e2m1fn_x2), B["q"].view(torch.float4_e2m1fn_x2),
                          A["s"].view(torch.float8_e8m0fnu), B["s"].view(torch.float8_e8m0fnu), y).float()
    out = mm.mbs_gemm(A, B).float()
    rel = ((out - ref).norm() / ref.norm()).item()
    check("plain GEMM == aiter asm gemm_a4w4", rel < 1e-2, f"rel err {rel:.2e}, max abs {(out - ref).abs().max().item():.3g}")


if __name__ == "__main__":
    test_plain_vs_aiter()
    test_mbs_vs_ref()
    test_vs_paper_sim()
    test_gemm()
    test_plain_gemm_vs_asm()
    print("ALL PASS" if not check.failed else "SOME CHECKS FAILED")
    sys.exit(1 if check.failed else 0)
