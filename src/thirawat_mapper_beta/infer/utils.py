"""Shared inference utilities for the beta CLIs."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple
import re

import pandas as pd
from thirawat_mapper_beta.scoring import batch_features
from thirawat_mapper_beta.scoring import post_scorer as _ps
from thirawat_mapper_beta.utils import normalize_text_value


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


def rank_candidates(
    df: pd.DataFrame,
    *,
    prefer_brand: bool = True,
    drop_private_brand_col: bool = True,
    extra_tiebreaks: Sequence[Tuple[str, bool]] | None = None,
) -> pd.DataFrame:
    """Return a ranked copy of candidate DataFrame with consistent rules.

    - Prefers higher brand_score first (0 > negatives). Supports either
      "brand_score" or temporary "_brand_score" column.
    - Then sorts by "final_score" (desc) if present; else by "post_score" (desc).
    - Then by "strength_sim" and "jaccard_text" (desc) if present.
    - Accepts additional (column, ascending) tiebreakers via extra_tiebreaks.
    """
    if df is None or df.empty:
        return df

    brand_col = None
    for name in ("brand_score", "_brand_score"):
        if name in df.columns:
            brand_col = name
            break

    order: list[Tuple[str, bool]] = []
    if prefer_brand and brand_col is not None:
        order.append((brand_col, False))

    if "final_score" in df.columns:
        order.append(("final_score", False))
    elif "post_score" in df.columns:
        order.append(("post_score", False))

    if "strength_sim" in df.columns:
        order.append(("strength_sim", False))
    if "jaccard_text" in df.columns:
        order.append(("jaccard_text", False))

    if extra_tiebreaks:
        order.extend(extra_tiebreaks)

    if not order:
        return df

    by = [c for c, _ in order]
    ascending = [asc for _, asc in order]
    ranked = df.sort_values(by=by, ascending=ascending).reset_index(drop=True)
    if drop_private_brand_col and brand_col == "_brand_score" and brand_col in ranked.columns:
        ranked = ranked.drop(columns=brand_col)
    return ranked


__all__.append("rank_candidates")


def _combine_name_profile(name: str | None, profile: str | None) -> str:
    n = (name or "").strip()
    p = (profile or "").strip()
    return (n + " " + p).strip()


def enrich_with_post_scores(
    df: pd.DataFrame,
    query_text_norm: str,
    *,
    post_strength_weight: float,
    post_jaccard_weight: float,
    post_brand_penalty: float,
    post_minmax: bool,
    post_weight: float,
    prefer_brand: bool = True,
) -> pd.DataFrame:
    """Attach post-scoring columns and return a ranked DataFrame.

    Adds columns: strength_sim, jaccard_text, brand_score, post_score, final_score.
    Uses combined raw text "concept_name + profile_text" for brand detection,
    normalized profile text for strength/text, and blends with _relevance_score.
    """
    if df is None or df.empty:
        return df

    raw_names = df["concept_name"].astype(str).tolist() if "concept_name" in df.columns else [""] * len(df)
    raw_profiles = df["profile_text"].astype(str).tolist() if "profile_text" in df.columns else [""] * len(df)
    raw_texts = [_combine_name_profile(n, p) for n, p in zip(raw_names, raw_profiles)]
    cands_text = [normalize_text_value(t) for t in raw_texts]

    feats = batch_features(
        query_text_norm,
        cands_text,
        w_strength=float(post_strength_weight),
        w_jaccard=float(post_jaccard_weight),
        w_brand_penalty=float(post_brand_penalty),
        minmax_within_query=bool(post_minmax),
    )
    df = df.copy()
    df["strength_sim"] = feats["strength_sim"]
    df["jaccard_text"] = feats["jaccard_text"]
    # Recompute brand on raw combined text to preserve bracketed brands
    try:
        df["brand_score"] = [_ps.brand_score(query_text_norm, raw) for raw in raw_texts]
    except Exception:
        df["brand_score"] = feats["brand_score"]
    df["post_score"] = feats.get("post_score", feats.get("simple_score"))

    relevance = df.get("_relevance_score", pd.Series([0.0] * len(df)))
    w = float(post_weight)
    df["final_score"] = (1.0 - w) * relevance.fillna(0.0) + w * df["post_score"].astype(float)

    return rank_candidates(df, prefer_brand=prefer_brand, drop_private_brand_col=True)


__all__.append("enrich_with_post_scores")


def sanitize_query_text(
    text: str,
    *,
    strip_non_latin: bool = False,
    strip_chars: str = "",
) -> str:
    """Optionally remove non‑Latin characters and specific characters from a query.

    - strip_non_latin: drops non‑ASCII codepoints to avoid retrieval noise from other scripts.
    - strip_chars: a string of literal characters to remove (e.g., "()[]{}").
    """
    s = text or ""
    if strip_chars:
        tbl = {ord(c): None for c in strip_chars}
        s = s.translate(tbl)
    if strip_non_latin:
        s = re.sub(r"[^\x00-\x7F]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


__all__.append("sanitize_query_text")
