"""Elementary functions with consistent accuracy on every backend.

``torch.expm1`` on Apple MPS is evaluated as ``exp(x) - 1`` (relative error ~1e-2 at
x = 1e-5 in float32), which defeats the cancellation-free formulations used for
low precision.  :func:`expm1` below uses a Taylor series for small arguments.
"""

from __future__ import annotations

import math

import torch

_EXPM1_SWITCH = 0.5


def _expm1_coeffs(dtype):
    # truncation error |x|^(n+1)/(n+1)! relative to |x| at |x| = 0.5 below eps
    n = 12 if torch.finfo(dtype).eps > 1e-10 else 20
    return [1.0 / math.factorial(k) for k in range(1, n + 1)]


def expm1(x: torch.Tensor) -> torch.Tensor:
    """exp(x) - 1 accurate for small |x| on all devices."""
    if x.device.type == "cpu" or x.device.type == "cuda":
        return torch.expm1(x)
    small = torch.abs(x) < _EXPM1_SWITCH
    xs = torch.where(small, x, torch.zeros_like(x))
    coeffs = _expm1_coeffs(x.dtype)
    acc = torch.full_like(xs, coeffs[-1])
    for c in reversed(coeffs[:-1]):
        acc = acc * xs + c
    series = acc * xs
    big = torch.exp(torch.where(small, torch.zeros_like(x), x)) - 1.0
    return torch.where(small, series, big)
