"""Shared inference utilities for the beta CLIs."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple
import re

import pandas as pd
from thirawat_mapper.scoring import batch_features, extract_strengths_with_spans
from thirawat_mapper.utils import normalize_text_value


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


def tiebreak_rerank(
    df: pd.DataFrame,
    *,
    primary_col: str = "_relevance_score",
    tiebreak_cols: Sequence[str] = (
        "brand_strength_exact",
        "top20_strength_form_exact",
        "brand_score",
        "rerank_top20",
        "strength_exact",
        "strength_sim",
        "form_route_score",
        "release_score",
    ),
    eps: float = 0.01,
    topn: int = 50,
) -> pd.DataFrame:
    """Reorder candidates using post features only within near-ties of the primary score."""

    if df is None or df.empty or primary_col not in df.columns:
        return df

    df = df.sort_values(primary_col, ascending=False, kind="mergesort").reset_index(drop=True)
    if topn <= 0 or len(df) <= 1:
        return df

    head = df.iloc[: min(topn, len(df))].copy()
    tail = df.iloc[min(topn, len(df)) :].copy()

    gaps = head[primary_col].shift(1) - head[primary_col]
    head["_tie_group"] = (gaps > eps).fillna(False).cumsum()

    keys = [c for c in tiebreak_cols if c in head.columns]
    if not keys:
        return df

    sort_cols = list(keys) + [primary_col]
    ascending = [False] * len(sort_cols)
    groups = []
    for _, group in head.groupby("_tie_group", sort=False):
        groups.append(group.sort_values(sort_cols, ascending=ascending, kind="mergesort"))
    head = pd.concat(groups, ignore_index=True).drop(columns=["_tie_group"])

    return pd.concat([head, tail], ignore_index=True)


__all__.append("tiebreak_rerank")


def should_apply_post(post_mode: str, post_weight: float) -> bool:
    if post_mode == "blend":
        return float(post_weight) > 0.0
    return True


__all__.append("should_apply_post")


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
    post_mode: str = "tiebreak",
    tiebreak_eps: float = 0.01,
    tiebreak_topn: int = 50,
    brand_strict: bool = False,
) -> pd.DataFrame:
    """Attach post-score columns and recompute final_score, ranking consistently."""

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
    query_has_strength = bool(extract_strengths_with_spans(query_text_norm)[0])
    df = df.copy()
    df["strength_sim"] = feats["strength_sim"]
    df["jaccard_text"] = feats["jaccard_text"]
    df["brand_score"] = feats["brand_score"]
    if "release_score" in feats:
        df["release_score"] = feats["release_score"]
    if "form_release_score" in feats:
        df["form_release_score"] = feats["form_release_score"]
    if "form_route_score" in feats:
        df["form_route_score"] = feats["form_route_score"]
    df["post_score"] = feats.get("post_score", feats.get("simple_score"))
    df["strength_exact"] = ((df["strength_sim"] >= 0.99) & query_has_strength).astype(int)
    if "form_route_score" in df.columns and "strength_sim" in df.columns:
        exact_strength = (df["strength_sim"] >= 0.99) & query_has_strength
        if exact_strength.any():
            mask = exact_strength & (df["form_route_score"] < 0)
            df.loc[mask, "form_route_score"] = df.loc[mask, "form_route_score"] + 0.5

    base_col = None
    for col in ("_relevance_score", "score", "final_score"):
        if col in df.columns:
            base_col = col
            break
    if base_col is None:
        return df

    base_scores = pd.to_numeric(df[base_col], errors="coerce").fillna(float("-inf"))
    order = base_scores.sort_values(ascending=False, kind="mergesort").index
    rank = pd.Series(range(1, len(df) + 1), index=order)
    df["rerank_top20"] = (rank <= 20).astype(int)
    df["brand_strength_exact"] = ((df["brand_score"] > 0) & (df["strength_exact"] == 1)).astype(int)
    if "form_route_score" in df.columns:
        df["top20_strength_form_exact"] = (
            (df["rerank_top20"] == 1) & (df["strength_exact"] == 1) & (df["form_route_score"] >= 0.5)
        ).astype(int)
    else:
        df["top20_strength_form_exact"] = 0

    if brand_strict and re.search(r"\[([^\[\]]+)\]", query_text_norm or "") and "brand_score" in df.columns:
        mask = df["brand_score"] >= 0
        if mask.any():
            df = df.loc[mask].copy()

    if post_mode == "blend":
        relevance = df[base_col].astype(float).fillna(0.0)
        w = float(post_weight)
        df["final_score"] = (1.0 - w) * relevance + w * df["post_score"].astype(float)
        return rank_candidates(df, prefer_brand=prefer_brand, drop_private_brand_col=True)

    df["final_score"] = df[base_col].astype(float)

    if post_mode == "lex":
        sort_cols = [
            base_col,
            "brand_strength_exact",
            "top20_strength_form_exact",
            "brand_score",
            "rerank_top20",
            "strength_exact",
            "strength_sim",
            "form_route_score",
            "release_score",
        ]
        sort_cols = [c for c in sort_cols if c in df.columns]
        df = df.sort_values(sort_cols, ascending=[False] * len(sort_cols), kind="mergesort").reset_index(drop=True)
        return df

    if post_mode == "tiebreak":
        tiebreak_cols = [
            "brand_strength_exact",
            "top20_strength_form_exact",
            "brand_score",
            "rerank_top20",
            "strength_exact",
            "strength_sim",
            "form_route_score",
            "release_score",
        ]
        df = tiebreak_rerank(
            df,
            primary_col=base_col,
            tiebreak_cols=tuple(tiebreak_cols),
            eps=float(tiebreak_eps),
            topn=int(tiebreak_topn),
        )
        return df

    return df


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


_STRENGTH_UNITS_PATTERN = re.compile(
    r"(?<=\d)(?=(mg|mcg|g|ml|l|iu|meq|unit|units|unt)\b)", re.IGNORECASE
)


def normalize_strength_spacing(text: str) -> str:
    """Insert a space between magnitudes and common strength units when missing."""

    if not text:
        return text
    return _STRENGTH_UNITS_PATTERN.sub(" ", text)


__all__.append("normalize_strength_spacing")
