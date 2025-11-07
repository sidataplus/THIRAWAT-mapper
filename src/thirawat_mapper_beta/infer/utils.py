"""Shared inference utilities for the beta CLIs."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def resolve_device(name: Optional[str]) -> str:
    """Return best-available device: cuda → mps → cpu, with runtime sanity checks."""
    try:
        import torch  # runtime import

        wanted = (name or "auto").lower()
        has_cuda = torch.cuda.is_available()
        has_mps = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()

        def _ok(dev: str) -> bool:
            try:
                torch.tensor([0], device=dev)
                return True
            except Exception:
                return False

        if wanted in ("", "auto", "none"):
            if has_cuda and _ok("cuda"):
                return "cuda"
            if has_mps and _ok("mps"):
                return "mps"
            return "cpu"
        if wanted == "cuda":
            if has_cuda and _ok("cuda"):
                return "cuda"
            return "mps" if has_mps and _ok("mps") else "cpu"
        if wanted == "mps":
            if has_mps and _ok("mps"):
                return "mps"
            return "cuda" if has_cuda and _ok("cuda") else "cpu"
        return "cpu"
    except Exception:
        return "cpu"


def configure_torch_for_infer(device: str) -> None:
    """Enable fast matmul/TF32 knobs when safe to improve GPU utilization."""
    try:
        import torch  # runtime import

        torch.set_float32_matmul_precision("high")
        if device == "cuda":
            try:
                torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception:
        pass


def minmax_normalize(series: pd.Series) -> pd.Series:
    """Return min-max normalised copy of the series (safe for constant columns)."""
    vmin = float(series.min())
    vmax = float(series.max())
    rng = vmax - vmin
    if rng <= 1e-9:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - vmin) / rng


__all__ = ["configure_torch_for_infer", "minmax_normalize", "resolve_device"]
