"""Compute-device selection for the language models (CPU or NVIDIA GPU via CUDA).

``PV_DEVICE`` = ``auto`` (default: CUDA if a working GPU build of PyTorch sees a GPU, else CPU), ``cuda`` or ``cpu``.
The statistical pipeline never uses this module.
"""

from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def pick_device() -> str:
    want = os.environ.get("PV_DEVICE", "auto").lower()
    if want == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        return "cpu"
    if want == "cuda":
        raise RuntimeError("PV_DEVICE=cuda but PyTorch cannot see a CUDA GPU (CPU-only PyTorch build or driver issue)")
    return "cpu"


def device_info() -> dict:
    d = pick_device()
    info = {"device": d}
    if d == "cuda":
        import torch

        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    return info
