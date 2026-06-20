from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor


@lru_cache(maxsize=1)
def _load_extension():
    if not torch.cuda.is_available():
        return None
    if os.environ.get("UDLF_DISABLE_CUSTOM_SCAN") == "1":
        return None
    try:
        from torch.utils.cpp_extension import load

        root = Path(__file__).resolve().parent / "csrc"
        return load(
            name="udlf_selective_scan_cuda",
            sources=[str(root / "selective_scan.cpp"), str(root / "selective_scan_cuda.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
            extra_cflags=["/O2"] if os.name == "nt" else ["-O3"],
            verbose=os.environ.get("UDLF_CUDA_BUILD_VERBOSE") == "1",
        )
    except (ImportError, OSError, RuntimeError):
        if os.environ.get("UDLF_CUDA_BUILD_STRICT") == "1":
            raise
        return None


def custom_scan_available() -> bool:
    return _load_extension() is not None


def custom_scan_supported(d_state: int) -> bool:
    return d_state == 16 and torch.cuda.is_available() and os.environ.get("UDLF_DISABLE_CUSTOM_SCAN") != "1"


class _SelectiveScan(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        u: Tensor,
        delta: Tensor,
        a: Tensor,
        b: Tensor,
        c: Tensor,
        d: Tensor,
        z: Tensor,
        delta_bias: Tensor,
    ) -> Tensor:
        extension = _load_extension()
        if extension is None:
            raise RuntimeError("UDLF custom selective-scan CUDA extension is unavailable")
        original_dtypes = tuple(value.dtype for value in (u, delta, a, b, c, d, z, delta_bias))
        inputs = tuple(value.float().contiguous() for value in (u, delta, a, b, c, d, z, delta_bias))
        y, checkpoints = extension.forward(*inputs)
        ctx.save_for_backward(*inputs, checkpoints)
        ctx.original_dtypes = original_dtypes
        return y.to(original_dtypes[0])

    @staticmethod
    def backward(ctx, grad_y: Tensor):
        extension = _load_extension()
        if extension is None:
            raise RuntimeError("UDLF custom selective-scan CUDA extension disappeared during backward")
        u, delta, a, b, c, d, z, delta_bias, checkpoints = ctx.saved_tensors
        grads = extension.backward(
            grad_y.float().contiguous(), u, delta, a, b, c, d, z, delta_bias, checkpoints
        )
        return tuple(grad.to(dtype) for grad, dtype in zip(grads, ctx.original_dtypes, strict=True))


def selective_scan(
    u: Tensor,
    delta: Tensor,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    d: Tensor,
    *,
    z: Tensor,
    delta_bias: Tensor,
) -> Tensor:
    return _SelectiveScan.apply(u, delta, a, b, c, d, z, delta_bias)
