"""MXFP4 with Macro Block Scaling (MBS, arXiv:2603.08713) for gfx950, in Triton.

Format of a tensor with rows R and reduction dim K:
  * q : fp4 e2m1 codes, two per byte (low nibble = even k), uint8 [R, K/2];
  * s : one E8M0 scale per 32 k, chosen with the aiter/Quark "even" rule
        (the paper's OAS up to a 1.75-mantissa tie), uint8 [R, K/32];
  * m : one E0M8 factor per macro block of MB k (128 by default), stored as its
        8 mantissa bits, f = 1 + m/256 in [1, 2), uint8 [R, K/MB].
A macro block is multiplied by f before plain MXFP4 quantization, so
    x ~= fp4(q) * 2^(s - 127) / f.
With m = 0 everywhere this is exactly aiter's MXFP4.

MBS-S picks m from the top 8 mantissa bits of 6/amax(macro block) (paper eq. 3).
MBS-D searches `ncand` candidates for the smallest squared error: 16 spread
geometrically over one octave starting at the MBS-S factor (as in the paper's
description), or all 256 values.

GEMM:  C[i, j] = sum_kb  sigA[i, kb] * sigB[j, kb] * P_kb[i, j],   sigma = 1/f,
where P_kb is the scaled-MFMA (E8M0) partial sum over macro block kb.
"""

import bisect
import functools
import json
import os

import torch
import triton
import triton.language as tl

FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


# ----------------------------------------------------------------------------
# Reference (torch). Mirrors the kernels' arithmetic so results compare exactly.
# ----------------------------------------------------------------------------


def _fp4_code_ref(a):
    """|value| (fp32, already divided by the E8M0 scale) -> e2m1 code 0..7.
    Round to nearest, ties to even code, saturate at 6."""
    return (
        (a > 0.25).int()
        + (a >= 0.75).int()
        + (a > 1.25).int()
        + (a >= 1.75).int()
        + (a > 2.5).int()
        + (a >= 3.5).int()
        + (a > 5.0).int()
    )


def _e8m0_even_ref(amax):
    """amax (fp32 >= 0) -> unbiased E8M0 exponent, aiter's `even` rule."""
    r = (amax.view(torch.int32) + 0x200000) & 0x7F800000
    return (((r >> 23) & 0xFF) - 129).clamp(-127, 127)


def _octave_candidates(ncand, device):
    return torch.tensor(
        [2.0 ** (-j / ncand) for j in range(ncand)], dtype=torch.float32, device=device
    )


def _mx_qdq_ref(xs):
    """MXFP4 quant-dequant of xs [..., n*32] (fp32). Returns (codes, exps, deq)."""
    shp = xs.shape
    xb = xs.reshape(*shp[:-1], shp[-1] // 32, 32)
    e = _e8m0_even_ref(xb.abs().amax(-1))
    inv = ((127 - e) << 23).view(torch.float32)
    qx = xb * inv[..., None]
    code = _fp4_code_ref(qx.abs())
    val = torch.tensor(FP4_VALUES, device=xs.device)[code.long()]
    scale = ((e + 127).clamp(min=0) << 23).view(torch.float32)
    deq = torch.where(qx < 0, -val, val) * scale[..., None]
    sign = (qx.view(torch.int32) >> 31) & 1
    return (code | (sign << 3)).reshape(shp), e, deq.reshape(shp)


def mbs_factor_ref(x, mb=128, ncand=1):
    """E0M8 mantissa m [R, K/mb] (int32) that MBS picks for x [R, K]."""
    R, K = x.shape
    xm = x.float().reshape(R, K // mb, mb)
    amax = xm.abs().amax(-1)
    m0 = ((6.0 / amax).view(torch.int32) & 0x7F8000) >> 15
    if ncand == 1:
        return m0
    if ncand == 256:
        cands = [torch.full_like(m0, j) for j in range(256)]
    else:
        f0 = 1.0 + m0.float() / 256.0
        cands = []
        for c in _octave_candidates(ncand, x.device).tolist():
            f = f0 * torch.tensor(c, dtype=torch.float32, device=x.device)
            f = torch.where(f < 1.0, f * 2.0, f)
            cands.append(((f - 1.0) * 256.0).int())
    best_m, best_err = None, None
    for mj in cands:
        f = 1.0 + mj.float() / 256.0
        _, _, deq = _mx_qdq_ref(xm * f[..., None])
        err = (deq / f[..., None] - xm).pow(2).sum(-1)
        if best_m is None:
            best_m, best_err = mj, err
        else:
            better = err < best_err
            best_m = torch.where(better, mj, best_m)
            best_err = torch.where(better, err, best_err)
    return best_m


def mbs_quant_ref(x, mb=128, ncand=1, use_mbs=True):
    """Reference quantizer: (q uint8 [R,K/2], s uint8 [R,K/32], m uint8 [R,K/mb])."""
    R, K = x.shape
    if use_mbs:
        m = mbs_factor_ref(x, mb, ncand)
    else:
        m = torch.zeros(R, K // mb, dtype=torch.int32, device=x.device)
    f = (1.0 + m.float() / 256.0).repeat_interleave(mb, dim=1)
    code, e, _ = _mx_qdq_ref(x.float() * f)
    q = (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8)
    return q, (e + 127).to(torch.uint8), m.to(torch.uint8)


def mbs_dequant(q, s, m, mb=128):
    """(q, s, m) -> fp32 [R, K]. s, m row-major (not shuffled)."""
    R = q.shape[0]
    q = q.view(torch.uint8)
    codes = torch.stack([q & 0xF, q >> 4], dim=-1).reshape(R, -1).long()
    lut = torch.tensor(FP4_VALUES + tuple(-v for v in FP4_VALUES), device=q.device)
    val = lut[codes]
    scale = torch.exp2(s.view(torch.uint8).float() - 127.0).repeat_interleave(32, dim=1)
    f = (1.0 + m.view(torch.uint8).float() / 256.0).repeat_interleave(mb, dim=1)
    return val * scale / f


def mbs_gemm_ref(a, b, mb=128):
    """C = dequant(a) @ dequant(b)^T in fp32; a, b are (q, s, m) triples."""
    return mbs_dequant(*a, mb=mb) @ mbs_dequant(*b, mb=mb).T


# ----------------------------------------------------------------------------
# Quantization kernel
# ----------------------------------------------------------------------------


@triton.jit
def _fp4_code(a):
    """|value| (fp32, divided by the E8M0 scale) -> e2m1 code (int32 0..7):
    round to nearest, ties to even code, saturate at 6."""
    c = (a > 0.25).to(tl.int32) + (a >= 0.75).to(tl.int32)
    c += (a > 1.25).to(tl.int32) + (a >= 1.75).to(tl.int32)
    c += (a > 2.5).to(tl.int32) + (a >= 3.5).to(tl.int32)
    c += (a > 5.0).to(tl.int32)
    return c


@triton.jit
def _fp4_value(c):
    """e2m1 magnitude code 0..7 -> fp32 value."""
    normal = ((((c >> 1) + 126) << 23) | ((c & 1) << 22)).to(tl.float32, bitcast=True)
    return tl.where(c < 2, c.to(tl.float32) * 0.5, normal)


@triton.jit
def _e8m0_even(amax):
    """amax (fp32 >= 0) -> unbiased E8M0 exponent (int32), aiter's `even` rule."""
    r = (amax.to(tl.int32, bitcast=True) + 0x200000) & 0x7F800000
    return tl.minimum(tl.maximum(((r >> 23) & 0xFF) - 129, -127), 127)


@triton.jit
def _block_sse(x, f, BLOCK_M: tl.constexpr, MB: tl.constexpr):
    """Squared error of MXFP4(x * f) / f against x, per row. x [BLOCK_M, MB]."""
    xs = (x * f[:, None]).reshape(BLOCK_M, MB // 32, 32)
    e = _e8m0_even(tl.max(tl.abs(xs), axis=2))
    inv = ((127 - e) << 23).to(tl.float32, bitcast=True)
    scale = (tl.maximum(e + 127, 0) << 23).to(tl.float32, bitcast=True)
    qx = xs * inv[:, :, None]
    val = _fp4_value(_fp4_code(tl.abs(qx))) * scale[:, :, None]
    deq = tl.where(qx < 0, -val, val)
    deq = tl.math.div_rn(deq, f[:, None, None])
    d = deq - x.reshape(BLOCK_M, MB // 32, 32)
    return tl.sum(tl.sum(d * d, axis=2), axis=1)


@triton.jit
def _static_m(amax, BLOCK_M: tl.constexpr):
    """MBS-S: top 8 mantissa bits of 6/amax (6/0 = inf has a zero mantissa)."""
    six = tl.full((BLOCK_M,), 6.0, tl.float32)
    return (tl.math.div_rn(six, amax).to(tl.int32, bitcast=True) & 0x7F8000) >> 15


@triton.jit
def _mbs_quant_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    m_ptr,
    cand_ptr,
    r_ptr,
    sl_ptr,
    M,
    NKB,
    stride_xm,
    stride_qm,
    stride_sm,
    stride_mm,
    stride_mk,
    stride_rk,
    scale_rows,
    scaleN_pad,
    MB: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NCAND: tl.constexpr,
    USE_MBS: tl.constexpr,
    SHUFFLE: tl.constexpr,
    WRITE_M: tl.constexpr,
    RATIO_OUT: tl.constexpr,
):
    """One program: BLOCK_M rows x one macro block of MB k.

    RATIO_OUT (static factors only) also writes what the GEMM consumes:
    ratio[kb, row] = f[kb]/f[kb-1] (fp32, f[-1] = 1) and, from the last
    macro block, sigma_last[row] = 1/f. The previous block's factor is
    recomputed from its amax, so no program waits on another."""
    pid_m = tl.program_id(0)
    kb = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_ok = offs_m < M
    offs_k = kb * MB + tl.arange(0, MB)
    x = tl.load(
        x_ptr + offs_m[:, None].to(tl.int64) * stride_xm + offs_k[None, :],
        mask=row_ok[:, None],
        other=0.0,
    ).to(tl.float32)

    if USE_MBS:
        m0 = _static_m(tl.max(tl.abs(x), axis=1), BLOCK_M)
        if NCAND == 1:
            mval = m0
        else:
            f0 = 1.0 + m0.to(tl.float32) * (1.0 / 256.0)
            best_m = m0
            best_err = tl.full((BLOCK_M,), float("inf"), tl.float32)
            for j in range(NCAND):
                if NCAND == 256:
                    mj = tl.full((BLOCK_M,), 0, tl.int32) + j
                else:
                    fj = f0 * tl.load(cand_ptr + j)
                    fj = tl.where(fj < 1.0, fj * 2.0, fj)
                    mj = ((fj - 1.0) * 256.0).to(tl.int32)
                err = _block_sse(
                    x, 1.0 + mj.to(tl.float32) * (1.0 / 256.0), BLOCK_M, MB
                )
                better = err < best_err
                best_err = tl.where(better, err, best_err)
                best_m = tl.where(better, mj, best_m)
            mval = best_m
        f = 1.0 + mval.to(tl.float32) * (1.0 / 256.0)
        x = x * f[:, None]
        if WRITE_M:
            tl.store(
                m_ptr + offs_m.to(tl.int64) * stride_mm + kb * stride_mk,
                mval.to(tl.uint8),
                mask=row_ok,
            )
        if RATIO_OUT:
            tl.static_assert(NCAND == 1, "ratios are recomputed from static factors")
            x_prev = tl.load(
                x_ptr
                + offs_m[:, None].to(tl.int64) * stride_xm
                + (offs_k - MB)[None, :],
                mask=row_ok[:, None] & (kb > 0),
                other=0.0,
            ).to(tl.float32)
            f_prev = 1.0 + _static_m(tl.max(tl.abs(x_prev), axis=1), BLOCK_M).to(
                tl.float32
            ) * (1.0 / 256.0)
            tl.store(
                r_ptr + kb * stride_rk + offs_m, tl.math.div_rn(f, f_prev), mask=row_ok
            )
            tl.store(
                sl_ptr + offs_m,
                tl.math.div_rn(tl.full((BLOCK_M,), 1.0, tl.float32), f),
                mask=row_ok & (kb == NKB - 1),
            )

    xs = x.reshape(BLOCK_M, MB // 32, 32)
    e = _e8m0_even(tl.max(tl.abs(xs), axis=2))
    inv = ((127 - e) << 23).to(tl.float32, bitcast=True)
    qx = xs * inv[:, :, None]
    code = _fp4_code(tl.abs(qx)) | (((qx.to(tl.int32, bitcast=True) >> 31) & 1) << 3)
    lo, hi = tl.split(code.reshape(BLOCK_M, MB // 2, 2))
    packed = (lo | (hi << 4)).to(tl.uint8)
    offs_qk = kb * (MB // 2) + tl.arange(0, MB // 2)
    tl.store(
        q_ptr + offs_m[:, None].to(tl.int64) * stride_qm + offs_qk[None, :],
        packed,
        mask=row_ok[:, None],
    )

    e8m0 = tl.where(row_ok[:, None], e + 127, 0).to(tl.uint8)
    offs_b = kb * (MB // 32) + tl.arange(0, MB // 32)
    if SHUFFLE:
        # aiter's shuffled E8M0 layout (shuffle_scale / quant shuffle=True)
        r = offs_m[:, None]
        b = offs_b[None, :]
        s_offs = (
            (r % 32) // 16
            + ((b % 8) // 4) * 2
            + (r % 16) * 4
            + (b % 4) * 64
            + (b // 8) * 256
            + (r // 32).to(tl.int64) * 32 * scaleN_pad
        )
        tl.store(s_ptr + s_offs, e8m0, mask=(r < scale_rows))
    else:
        tl.store(
            s_ptr + offs_m[:, None].to(tl.int64) * stride_sm + offs_b[None, :],
            e8m0,
            mask=row_ok[:, None],
        )


@functools.cache
def _cands(ncand, device):
    return _octave_candidates(max(ncand, 1), device)


def _quant(x, mb, ncand, use_mbs, shuffle_scale, m_layout, ratio_out, block_m):
    assert x.dim() == 2 and x.stride(1) == 1
    M, K = x.shape
    assert K % mb == 0 and mb % 32 == 0, (K, mb)
    dev = x.device
    q = torch.empty(M, K // 2, dtype=torch.uint8, device=dev)
    if shuffle_scale:
        scaleN_pad = triton.cdiv(K // 32, 8) * 8
        # rows past round_up(M, 32) are never read (the GEMM wraps row groups)
        s = torch.empty(
            triton.cdiv(M, 256) * 256, scaleN_pad, dtype=torch.uint8, device=dev
        )
        valid_rows = triton.cdiv(M, 32) * 32
    else:
        s = torch.empty(M, K // 32, dtype=torch.uint8, device=dev)
        scaleN_pad, valid_rows = K // 32, M
    m, stride_mm, stride_mk = q, 0, 0
    if use_mbs and m_layout == "rows":
        m = torch.empty(M, K // mb, dtype=torch.uint8, device=dev)
        stride_mm, stride_mk = K // mb, 1
    elif use_mbs and m_layout == "kb":
        m = torch.empty(K // mb, M, dtype=torch.uint8, device=dev)
        stride_mm, stride_mk = 1, M
    ratio = sigma_last = q
    ld_r = 0
    if use_mbs and ratio_out:
        # 16-aligned rows let the GEMM vectorize its loads
        ld_r = triton.cdiv(M, 16) * 16
        ratio = torch.empty(K // mb, ld_r, dtype=torch.float32, device=dev)
        sigma_last = torch.empty(M, dtype=torch.float32, device=dev)
    if block_m is None:
        # small M: no masked rows, and enough programs to fill the GPU
        block_m = (
            8 if ncand > 1 else min(32, triton.next_power_of_2(max(valid_rows, 1)))
        )
        while block_m > 8 and triton.cdiv(valid_rows, block_m) * (K // mb) < 512:
            block_m //= 2
    num_warps = (
        4 if ncand > 1 or block_m * mb >= 4096 else (2 if block_m * mb >= 1024 else 1)
    )
    grid = (triton.cdiv(max(M, valid_rows), block_m), K // mb)
    _mbs_quant_kernel[grid](
        x,
        q,
        s,
        m,
        _cands(ncand, dev),
        ratio,
        sigma_last,
        M,
        K // mb,
        x.stride(0),
        q.stride(0),
        s.stride(0),
        stride_mm,
        stride_mk,
        ld_r,
        valid_rows,
        scaleN_pad,
        MB=mb,
        BLOCK_M=block_m,
        NCAND=ncand,
        USE_MBS=use_mbs,
        SHUFFLE=shuffle_scale,
        WRITE_M=use_mbs and m_layout is not None,
        RATIO_OUT=use_mbs and ratio_out,
        num_warps=num_warps,
    )
    return (
        q,
        s,
        (m if use_mbs and m_layout else None),
        (ratio if ld_r else None),
        (sigma_last if ld_r else None),
    )


def mbs_quant(
    x,
    mb=128,
    ncand=1,
    use_mbs=True,
    shuffle_scale=False,
    m_transposed=False,
    block_m=None,
):
    """Quantize x [M, K] (bf16/fp16/fp32) to MXFP4 (+ MBS).

    Returns (q uint8 [M, K/2], s uint8, m uint8 or None):
      s is [M, K/32] row-major, or aiter's shuffled layout padded to
        [round_up(M, 256), round_up(K/32, 8)] when shuffle_scale;
      m is [M, K/mb], or [K/mb, M] (contiguous along M) when m_transposed.
    """
    q, s, m, _, _ = _quant(
        x,
        mb,
        ncand,
        use_mbs,
        shuffle_scale,
        "kb" if m_transposed else "rows",
        False,
        block_m,
    )
    return q, s, m


# ----------------------------------------------------------------------------
# GEMM kernel on the production layouts: A row-major with shuffled E8M0 scales
# (row-major below 32 rows), B preshuffled 16x16 (aiter shuffle_weight) with
# shuffled scales, MBS mantissas m stored transposed [K/MB, rows].
# ----------------------------------------------------------------------------


@triton.jit
def _pid_grouped(pid, num_pid_m, num_pid_n, GROUP_M: tl.constexpr):
    group = GROUP_M * num_pid_n
    first_m = (pid // group) * GROUP_M
    size_m = tl.minimum(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group) % size_m
    pid_n = (pid % group) // size_m
    return pid_m, pid_n


@triton.jit
def _unshuffle_scales(s, ROWS: tl.constexpr, KS: tl.constexpr):
    """[ROWS/32, KS*32] bytes in aiter's shuffled E8M0 layout -> [ROWS, KS]."""
    return (
        s.reshape(ROWS // 32, KS // 8, 4, 16, 2, 2, 1)
        .permute(0, 5, 3, 1, 4, 2, 6)
        .reshape(ROWS, KS)
    )


@triton.jit
def _unshuffle_b(b, BLOCK_N: tl.constexpr, KH: tl.constexpr):
    """[BLOCK_N/16, KH*16] preshuffled fp4x2 bytes -> [KH, BLOCK_N] (K-major operand)."""
    return (
        b.reshape(1, BLOCK_N // 16, KH // 32, 2, 16, 16)
        .permute(0, 1, 4, 2, 3, 5)
        .reshape(BLOCK_N, KH)
        .trans(1, 0)
    )


@triton.jit
def _half_scale_offsets(H: tl.constexpr):
    """Byte offsets, inside one 32-row x 256-K block of shuffled scales, of the
    128 scales of K half H, ordered as (k-block%4, row%16, row/16)."""
    i = tl.arange(0, 128)
    return (i // 32) * 64 + ((i // 2) % 16) * 4 + H * 2 + (i % 2)


@triton.jit
def _half_scales_to_rows(s, ROWS: tl.constexpr):
    """[ROWS/32, 128] half-stage scales (see _half_scale_offsets) -> [ROWS, 4]."""
    return s.reshape(ROWS // 32, 4, 16, 2).permute(0, 3, 2, 1).reshape(ROWS, 4)


@triton.jit(do_not_specialize=["M"])
def _mbs_gemm_kernel(
    a_ptr,
    as_ptr,
    ra_ptr,
    sla_ptr,
    b_ptr,
    bs_ptr,
    rb_ptr,
    slb_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_asm,
    stride_rak,
    stride_bn16,
    stride_bsn,
    stride_rbk,
    stride_cm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    MB: tl.constexpr,
    A_MBS: tl.constexpr,
    B_MBS: tl.constexpr,
    A_SCALE_SHUFFLED: tl.constexpr,
):
    """C[M, N] = dequant(A) @ dequant(B)^T with per-macro-block factors.

    K advances BLOCK_K per iteration, as BLOCK_K/CH scaled-MFMA dots of
    CH = min(MB, BLOCK_K) k each. With D_kb = sum_{j<=kb} P_j/(fA_j fB_j) the
    accumulator holds D_kb * fA_kb * fB_kb: entering macro block kb it is
    multiplied by ratio[kb] = f_kb/f_kb-1 of each side and the MFMAs add P_kb
    in place; the epilogue multiplies by sigma_last = 1/f of the last block.
    One accumulator, 2 FP32 multiplies per output per macro block (1 with
    one-sided MBS). Macro blocks longer than BLOCK_K rescale on their first
    iteration only. ratio [K/MB, rows] / sigma_last [rows] are fp32; rows past
    M (N) read 1.
    """
    CH: tl.constexpr = min(MB, BLOCK_K)
    NCH: tl.constexpr = BLOCK_K // CH
    ITERS_PER_MB: tl.constexpr = MB // BLOCK_K if MB > BLOCK_K else 1
    tl.static_assert(CH >= 128 and CH % 128 == 0, "dot chunks are whole MFMA K steps")
    if CH == 128:
        tl.static_assert(NCH % 2 == 0, "MB=128 needs BLOCK_K % 256 == 0")

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m, pid_n = _pid_grouped(pid, num_pid_m, num_pid_n, GROUP_M)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_ok = rows < M
    col_ok = cols < N
    offs_am = rows % M
    offs_bn16 = (pid_n * (BLOCK_N // 16) + tl.arange(0, BLOCK_N // 16)) % (N // 16)
    offs_bsn = (pid_n * (BLOCK_N // 32) + tl.arange(0, BLOCK_N // 32)) % tl.cdiv(N, 32)

    a_ptrs = (
        a_ptr
        + offs_am[:, None].to(tl.int64) * stride_am
        + tl.arange(0, CH // 2)[None, :]
    )
    b_ptrs = (
        b_ptr
        + offs_bn16[:, None].to(tl.int64) * stride_bn16
        + tl.arange(0, CH * 8)[None, :]
    )
    # scales: a 32-row group holds 256 bytes per 256-K stage; with CH = 128 a dot
    # needs one interleaved half of a stage (see _half_scale_offsets)
    if CH == 128:
        bs_ptrs = bs_ptr + offs_bsn[:, None].to(tl.int64) * stride_bsn
    else:
        bs_ptrs = (
            bs_ptr
            + offs_bsn[:, None].to(tl.int64) * stride_bsn
            + tl.arange(0, CH)[None, :]
        )
    if A_SCALE_SHUFFLED:
        offs_asm = (pid_m * (BLOCK_M // 32) + tl.arange(0, BLOCK_M // 32)) % tl.cdiv(
            M, 32
        )
        if CH == 128:
            as_ptrs = as_ptr + offs_asm[:, None].to(tl.int64) * stride_asm
        else:
            as_ptrs = (
                as_ptr
                + offs_asm[:, None].to(tl.int64) * stride_asm
                + tl.arange(0, CH)[None, :]
            )
    else:
        as_ptrs = (
            as_ptr
            + offs_am[:, None].to(tl.int64) * stride_asm
            + tl.arange(0, CH // 32)[None, :]
        )
    ra_ptrs = ra_ptr + rows
    rb_ptrs = rb_ptr + cols

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(K // BLOCK_K):
        for j in tl.static_range(NCH):
            a = tl.load(a_ptrs + j * (CH // 2))
            b = _unshuffle_b(tl.load(b_ptrs + j * (CH * 8)), BLOCK_N, CH // 2)
            if CH == 128:
                soff = (j // 2) * 256 + _half_scale_offsets(j % 2)[None, :]
                bs = _half_scales_to_rows(tl.load(bs_ptrs + soff), BLOCK_N)
                if A_SCALE_SHUFFLED:
                    a_s = _half_scales_to_rows(tl.load(as_ptrs + soff), BLOCK_M)
                else:
                    a_s = tl.load(as_ptrs + j * 4)
            else:
                bs = _unshuffle_scales(tl.load(bs_ptrs + j * CH), BLOCK_N, CH // 32)
                if A_SCALE_SHUFFLED:
                    a_s = _unshuffle_scales(
                        tl.load(as_ptrs + j * CH), BLOCK_M, CH // 32
                    )
                else:
                    a_s = tl.load(as_ptrs + j * (CH // 32))
            if (A_MBS or B_MBS) and ITERS_PER_MB == 1:
                kb = i * NCH + j
                if A_MBS and B_MBS:
                    r = (
                        tl.load(ra_ptrs + kb * stride_rak, mask=row_ok, other=1.0)[
                            :, None
                        ]
                        * tl.load(rb_ptrs + kb * stride_rbk, mask=col_ok, other=1.0)[
                            None, :
                        ]
                    )
                elif A_MBS:
                    r = tl.load(ra_ptrs + kb * stride_rak, mask=row_ok, other=1.0)[
                        :, None
                    ]
                else:
                    r = tl.load(rb_ptrs + kb * stride_rbk, mask=col_ok, other=1.0)[
                        None, :
                    ]
                acc = tl.dot_scaled(a, a_s, "e2m1", b, bs, "e2m1", acc * r)
            else:
                # compile-time condition around a run-time one; they cannot merge
                if (A_MBS or B_MBS) and ITERS_PER_MB > 1:  # noqa: SIM102
                    if i % ITERS_PER_MB == 0:
                        kb = i // ITERS_PER_MB
                        if A_MBS:
                            acc = (
                                acc
                                * tl.load(
                                    ra_ptrs + kb * stride_rak, mask=row_ok, other=1.0
                                )[:, None]
                            )
                        if B_MBS:
                            acc = (
                                acc
                                * tl.load(
                                    rb_ptrs + kb * stride_rbk, mask=col_ok, other=1.0
                                )[None, :]
                            )
                acc = tl.dot_scaled(a, a_s, "e2m1", b, bs, "e2m1", acc)
        a_ptrs += BLOCK_K // 2
        b_ptrs += BLOCK_K * 8
        bs_ptrs += BLOCK_K
        if A_SCALE_SHUFFLED:
            as_ptrs += BLOCK_K
        else:
            as_ptrs += BLOCK_K // 32

    if A_MBS:
        acc = acc * tl.load(sla_ptr + rows, mask=row_ok, other=1.0)[:, None]
    if B_MBS:
        acc = acc * tl.load(slb_ptr + cols, mask=col_ok, other=1.0)[None, :]

    c_ptrs = c_ptr + rows[:, None].to(tl.int64) * stride_cm + cols[None, :]
    tl.store(
        c_ptrs, acc.to(c_ptr.type.element_ty), mask=row_ok[:, None] & col_ok[None, :]
    )


GEMM_CONFIG = {
    "BLOCK_M": 128,
    "BLOCK_N": 128,
    "BLOCK_K": 256,
    "GROUP_M": 4,
    "num_warps": 4,
    "num_stages": 2,
    "waves_per_eu": 2,
    "matrix_instr_nonkdim": 16,
}


def mbs_gemm(a, b, mb=128, out_dtype=torch.bfloat16, config=None, out=None):
    """C[M, N] = dequant(a) @ dequant(b)^T.

    a: {"q": [M, K/2] uint8, "s": E8M0 scales, "s_shuffled": bool (default True),
        "ratio": fp32 [K/mb, ld >= M], "sigma_last": fp32 [M]} (ratio None: plain)
       Shuffled scales are the 2-D [round_up(M, 256), round_up(K/32, 8)] buffer
       of `mbs_quant(..., shuffle_scale=True)`; row-major ones are [M, K/32]
       (the only option when BLOCK_M < 32).
    b: {"q": [N, K/2] uint8 preshuffled with shuffle_weight(layout=(16, 16)),
        "s": shuffled scales, "ratio": fp32 [K/mb, N], "sigma_last": fp32 [N]}
    `quant_activation` and `prepare_weight` build these.
    """
    M, Kp = a["q"].shape
    N = b["q"].shape[0]
    K = 2 * Kp
    cfg = dict(GEMM_CONFIG, **(config or {}))
    step = max(mb, cfg["BLOCK_K"])
    assert K % step == 0, f"K={K} must be a multiple of max(MB, BLOCK_K)={step}"
    assert N % 16 == 0, N
    a_shuf = a.get("s_shuffled", True)
    if a_shuf:
        assert cfg["BLOCK_M"] >= 32, "shuffled A scales need BLOCK_M >= 32"
    if out is None:
        out = torch.empty(M, N, dtype=out_dtype, device=a["q"].device)
    a_r, b_r = a.get("ratio"), b.get("ratio")
    a_s = a["s"].view(torch.uint8)
    if a_shuf:
        a_s = a_s.view(-1, a_s.shape[-1] * 32)
    b_s = b["s"].view(torch.uint8)
    b_s = b_s.view(-1, b_s.shape[-1] * 32)
    b_q = b["q"].view(torch.uint8).view(N // 16, -1)
    grid = (triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"]),)
    _mbs_gemm_kernel[grid](
        a["q"].view(torch.uint8),
        a_s,
        a_r if a_r is not None else out,
        a["sigma_last"] if a_r is not None else out,
        b_q,
        b_s,
        b_r if b_r is not None else out,
        b["sigma_last"] if b_r is not None else out,
        out,
        M,
        N,
        K,
        a["q"].stride(0),
        a_s.stride(0),
        a_r.stride(0) if a_r is not None else 0,
        b_q.stride(0),
        b_s.stride(0),
        b_r.stride(0) if b_r is not None else 0,
        out.stride(0),
        MB=mb,
        A_MBS=a_r is not None,
        B_MBS=b_r is not None,
        A_SCALE_SHUFFLED=a_shuf,
        **cfg,
    )
    return out


def preshuffle_b(q):
    """aiter's shuffle_weight(layout=(16, 16)) for fp4x2 bytes [N, K/2]: each
    16-row group stores its 16-byte K chunks column by column."""
    N, Kp = q.shape
    return (
        q.view(N // 16, 16, Kp // 32, 2, 16)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
        .view(N, Kp)
    )


def mbs_ratios(m):
    """E0M8 mantissas [rows, K/MB] -> the GEMM's (ratio [K/MB, rows], sigma_last [rows]),
    fp32, ratio[kb] = f[kb]/f[kb-1] with f[-1] = 1."""
    f = 1.0 + m.float() / 256.0
    ratio = f.clone()
    ratio[:, 1:] = f[:, 1:] / f[:, :-1]
    return ratio.t().contiguous(), (1.0 / f[:, -1]).contiguous()


def prepare_weight(w, mb=128, ncand=16, use_mbs=True):
    """Quantize a weight [N, K] (once, at load time) into the GEMM's B operand."""
    q, s, m, _, _ = _quant(w, mb, ncand, use_mbs, True, "rows", False, None)
    out = {"q": preshuffle_b(q), "s": s, "m": m, "ratio": None, "sigma_last": None}
    if use_mbs:
        out["ratio"], out["sigma_last"] = mbs_ratios(m)
    return out


def quant_activation(x, mb=128, use_mbs=True, shuffle=None, keep_m=False):
    """Quantize an activation [M, K] (MBS-S) into the GEMM's A operand. Scales
    are shuffled from 32 rows on, row-major below (as aiter's preshuffle GEMM)."""
    if shuffle is None:
        shuffle = x.shape[0] >= 32
    q, s, m, ratio, sigma_last = _quant(
        x, mb, 1, use_mbs, shuffle, "rows" if keep_m else None, True, None
    )
    return {
        "q": q,
        "s": s,
        "s_shuffled": shuffle,
        "m": m,
        "ratio": ratio,
        "sigma_last": sigma_last,
    }


# ----------------------------------------------------------------------------
# Tile config selection
# ----------------------------------------------------------------------------

CONFIG_FILE = os.environ.get(
    "MBS_GEMM_CONFIGS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "mbs_gemm_configs.json"),
)


def variant_name(mb, a_mbs, b_mbs):
    """Key of a GEMM variant in the tuned table: plain, mbs<mb>, mbs<mb>_a, mbs<mb>_w."""
    if not (a_mbs or b_mbs):
        return "plain"
    return f"mbs{mb}" + ("" if a_mbs and b_mbs else ("_a" if a_mbs else "_w"))


@functools.cache
def _config_table():
    if not os.path.exists(CONFIG_FILE):
        return {}
    with open(CONFIG_FILE) as f:
        raw = json.load(f)
    table = {}
    for key, by_m in raw.items():
        ms = sorted(int(m) for m in by_m)
        table[key] = (
            ms,
            [{k: v for k, v in by_m[str(m)].items() if k != "us"} for m in ms],
        )
    return table


def default_config(M):
    if M <= 16:
        return {
            "BLOCK_M": 16,
            "BLOCK_N": 64,
            "BLOCK_K": 512,
            "num_warps": 2,
            "waves_per_eu": 2,
            "num_stages": 2,
            "matrix_instr_nonkdim": 16,
            "GROUP_M": 1,
        }
    if M <= 128:
        return {
            "BLOCK_M": 32,
            "BLOCK_N": 128,
            "BLOCK_K": 256,
            "num_warps": 4,
            "waves_per_eu": 2,
            "num_stages": 2,
            "matrix_instr_nonkdim": 16,
            "GROUP_M": 1,
        }
    if M <= 1024:
        return {
            "BLOCK_M": 64,
            "BLOCK_N": 128,
            "BLOCK_K": 256,
            "num_warps": 4,
            "waves_per_eu": 2,
            "num_stages": 2,
            "matrix_instr_nonkdim": 16,
            "GROUP_M": 4,
        }
    return {
        "BLOCK_M": 128,
        "BLOCK_N": 256,
        "BLOCK_K": 256,
        "num_warps": 4,
        "waves_per_eu": 1,
        "num_stages": 2,
        "matrix_instr_nonkdim": 16,
        "GROUP_M": 4,
    }


@functools.lru_cache(maxsize=4096)
def select_config(M, N, K, mb, a_mbs, b_mbs):
    """Tuned config of the smallest tuned M >= M (the largest one past the table),
    for this variant and (N, K); the nearest tuned macro-block size of the same
    sidedness, then default_config, when the exact variant was not tuned."""
    table = _config_table()
    names = [variant_name(mb, a_mbs, b_mbs)]
    if a_mbs or b_mbs:
        suffix = names[0][len(f"mbs{mb}") :]
        names += [f"mbs{b}{suffix}" for b in (256, 128, 512, 1024) if b != mb]
    cfg = None
    for name in names:
        entry = table.get(f"{name}|N={N}|K={K}")
        if entry is not None:
            ms, cfgs = entry
            cfg = dict(cfgs[min(bisect.bisect_left(ms, M), len(ms) - 1)])
            break
    if cfg is None:
        cfg = default_config(M)
    if M >= 32:
        cfg["BLOCK_M"] = max(cfg["BLOCK_M"], 32)
    # keep K a multiple of the step and MB=128 chunks paired inside 256-K stages
    while cfg["BLOCK_K"] > 256 and K % max(mb, cfg["BLOCK_K"]):
        cfg["BLOCK_K"] //= 2
    return cfg


def mbs_linear(x, w, mb=128, act_mbs=True, out_dtype=torch.bfloat16):
    """y = x @ W^T for x [M, K] bf16 and a weight from `prepare_weight`:
    MBS-S activation quantization (when act_mbs) + the MBS GEMM."""
    M, K = x.shape
    N = w["q"].shape[0]
    A = quant_activation(x, mb=mb, use_mbs=act_mbs)
    cfg = select_config(M, N, K, mb, act_mbs, w.get("ratio") is not None)
    return mbs_gemm(A, w, mb=mb, out_dtype=out_dtype, config=cfg)
