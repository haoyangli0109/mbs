"""Example: invoke the mbs tensor_add C++ kernel from Python.

Build first with:  make
Then run:          python3 example.py   (or: make run)
"""

import os
import sys

import torch

# Make sure the freshly built .so (in ./build) is importable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "build"))

import example  # noqa: E402  (the compiled extension)
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
import time


@triton.jit
def copy_kernel(x_ptr, x_stride, y_ptr):
    pid_m = tl.program_id(0)
    offset = pid_m * x_stride + tl.arange(0, 64 * 64)
    x = tl.load(x_ptr + offset)
    tl.store(y_ptr + offset, x)


def triton_copy(x: torch.Tensor, y: torch.Tensor) -> None:
    assert x.is_contiguous() and y.is_contiguous() and x.numel() == y.numel()
    n = x.numel()
    grid = (256,)
    copy_kernel[grid](x, x.stride(0), y)



def main() -> None:
    print(f"current_pid = {os.getpid()}")
    example.mbs_factor_experiment()

if __name__ == "__main__":
    main()
