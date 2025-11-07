"""Interactive query CLI."""

from __future__ import annotations

import argparse
from typing import Sequence

import pandas as pd

from thirawat_mapper_beta.models import SapBERTEmbedder, ThirawatReranker
from thirawat_mapper_beta.scoring import batch_features
from thirawat_mapper_beta.utils import connect_table, normalize_text_value
from .utils import configure_torch_for_infer, minmax_normalize, resolve_device


def _format_row(row: pd.Series) -> str:
    concept_id = row.get("concept_id")
    name = row.get("concept_name") or row.get("profile_text")
    score = row.get("final_score")
    sim = row.get("strength_sim")
    return f"{concept_id:<12} | {score:6.3f} | {sim:5.3f} | {name}" if pd.notna(score) else f"{concept_id:<12} | {name}"


def run(args: argparse.Namespace) -> None:
    table, vector_column = connect_table(args.db, args.table)
    device = resolve_device(args.device)
    configure_torch_for_infer(device)
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
        s_norm = minmax_normalize(df["strength_sim"].astype(float))
        j_norm = minmax_normalize(df["jaccard_text"].astype(float))
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
