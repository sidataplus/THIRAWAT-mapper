"""Interactive query CLI."""

from __future__ import annotations

import argparse
from typing import Sequence

import pandas as pd

from thirawat_mapper_beta.models import SapBERTEmbedder, ThirawatReranker
from thirawat_mapper_beta.scoring import batch_features
from thirawat_mapper_beta.utils import connect_table, normalize_text_value


def _resolve_device(name: str | None) -> str:
    """Return best-available device: cuda → mps → cpu, with runtime sanity checks."""
    try:
        import torch
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


def _configure_torch_for_infer(device: str) -> None:
    """Set fast matmul/TF32 knobs when safe to improve GPU utilization."""
    try:
        import torch
        torch.set_float32_matmul_precision("high")
        if device == "cuda":
            try:
                torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception:
        pass


def _format_row(row: pd.Series) -> str:
    concept_id = row.get("concept_id")
    name = row.get("concept_name") or row.get("profile_text")
    score = row.get("final_score")
    sim = row.get("strength_sim")
    return f"{concept_id:<12} | {score:6.3f} | {sim:5.3f} | {name}" if pd.notna(score) else f"{concept_id:<12} | {name}"


def run(args: argparse.Namespace) -> None:
    table, vector_column = connect_table(args.db, args.table)
    device = _resolve_device(args.device)
    _configure_torch_for_infer(device)
    embedder = SapBERTEmbedder(device=device, batch_size=args.batch_size)
    reranker = ThirawatReranker(device=device, return_score="all")

    print("Type a query (':q' to exit).")
    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):  # pragma: no cover
            print()
            break
        if not query:
            continue
        if query in {":q", ":quit", ":exit"}:
            break

        query_norm = normalize_text_value(query)
        vector = embedder.encode([query_norm])[0]
        builder = table.search(
            vector.astype(float).tolist(),
            vector_column_name=vector_column,
            query_type="vector",
        )
        try:
            result_table = (
                builder.distance_type("cosine")
                .limit(args.candidate_topk)
                .rerank(reranker=reranker, query_string=query_norm)
                .limit(args.candidate_topk)
                .to_arrow()
            )
            df = result_table.to_pandas()
        except Exception as exc:  # pragma: no cover - interactive error path
            print(f"Error running search: {exc}")
            continue

        if df.empty:
            print("No matches found.")
            continue

        keep_cols = [
            col
            for col in ["concept_id", "concept_name", "domain_id", "profile_text", "_relevance_score"]
            if col in df.columns
        ]
        if keep_cols:
            df = df.loc[:, keep_cols]

        cands_text = [normalize_text_value(t) for t in df["profile_text"].astype(str).tolist()]
        features = batch_features(query_norm, cands_text)
        df["strength_sim"] = [feat["strength_sim"] for feat in features]
        df["jaccard_text"] = [feat["jaccard_text"] for feat in features]
        # Per-query min-max normalization for simple features
        def _minmax(col: pd.Series) -> pd.Series:
            vmin = float(col.min())
            vmax = float(col.max())
            rng = vmax - vmin
            if rng <= 1e-9:
                return pd.Series([0.0] * len(col), index=col.index)
            return (col - vmin) / rng

        s_norm = _minmax(df["strength_sim"].astype(float))
        j_norm = _minmax(df["jaccard_text"].astype(float))
        df["simple_score"] = 0.6 * s_norm + 0.4 * j_norm
        relevance = df.get("_relevance_score", pd.Series([0.0] * len(df)))
        df["final_score"] = 0.7 * relevance.fillna(0.0) + 0.3 * df["simple_score"]
        df = df.sort_values(
            ["final_score", "strength_sim", "jaccard_text"], ascending=[False, False, False]
        ).reset_index(drop=True)

        print("concept_id   | score  | s_sim | name")
        print("-" * 80)
        for _, row in df.head(args.show_topk).iterrows():
            print(_format_row(row))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive terminology lookup")
    parser.add_argument("--db", required=True, help="Path to LanceDB directory")
    parser.add_argument("--table", required=True, help="LanceDB table name")
    parser.add_argument("--candidate-topk", type=int, default=100, help="Candidate pool size")
    parser.add_argument("--show-topk", type=int, default=10, help="Number of rows to display")
    parser.add_argument("--device", default="auto", help="Device: auto|cuda|mps|cpu (default: auto)")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":  # pragma: no cover
    main()
