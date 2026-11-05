"""Timing and input helpers shared by the benchmarks in this directory."""

import statistics

import torch


def graph_time_us(fn, n_calls=20, n_replays=7, warmup=3):
    """Per-call time of `fn` from CUDA-graph replays (hot L2, no launch gaps).

    Captures `n_calls` back-to-back calls into one graph and replays it
    `n_replays` times; returns (median, min) of the per-call time in us.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(n_calls):
            fn()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(n_replays):
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / n_calls)
    return statistics.median(times), min(times)


def tflops(M, N, K, us):
    return 2.0 * M * N * K / us / 1e6


# (name, N, K) of the dense Linear layers of Qwen3.5-9B (hidden 4096,
# intermediate 12288, 16 q heads x 256 with output gate, 4 kv heads, GDN with
# 16/32 key/value heads of 128).
QWEN35_9B_LINEARS = [
    ("gdn.in_proj_qkvz", 12288, 4096),
    ("gdn.out_proj", 4096, 4096),
    ("attn.qkv_proj", 10240, 4096),
    ("attn.o_proj", 4096, 4096),
    ("mlp.gate_up_proj", 24576, 4096),
    ("mlp.down_proj", 4096, 12288),
]

# The shapes `test_gemm_a4w4.py` in this repo sweeps (hidden 5120 model).
REPO_SHAPES = [
    ("mlp.gate_proj", 17408, 5120),
    ("mlp.down_proj", 5120, 17408),
    ("linear_attn.in_proj_qkv", 10240, 5120),
    ("linear_attn.in_proj_z", 6144, 5120),
    ("o_proj", 5120, 6144),
    ("self_attn.q_proj", 12288, 5120),
    ("self_attn.kv_proj", 1024, 5120),
]
