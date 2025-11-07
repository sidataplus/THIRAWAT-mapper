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
    final = row.get("final_score")
    strength = row.get("strength_sim")
    jac = row.get("jaccard_text")
    brand = row.get("brand_score")
    if pd.notna(final):
        return (
            f"{concept_id:<12} | {final:6.3f} | {strength:5.3f} | {jac:5.3f} | {brand:5.2f} | {name}"
        )
    return f"{concept_id:<12} | {name}"


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
        features = batch_features(
            query_norm,
            cands_text,
            w_strength=float(args.post_strength_weight),
            w_jaccard=float(args.post_jaccard_weight),
            w_brand_penalty=float(args.post_brand_penalty),
            minmax_within_query=False,
        )
        df["strength_sim"] = features["strength_sim"]
        df["jaccard_text"] = features["jaccard_text"]
        df["brand_score"] = features["brand_score"]
        # Optional per-query min-max normalization for simple features
        if args.post_minmax:
            s_norm = minmax_normalize(df["strength_sim"].astype(float))
            j_norm = minmax_normalize(df["jaccard_text"].astype(float))
        else:
            s_norm = df["strength_sim"].astype(float)
            j_norm = df["jaccard_text"].astype(float)
        denom = max(float(args.post_strength_weight) + float(args.post_jaccard_weight), 1e-9)
        blended = (float(args.post_strength_weight) * s_norm + float(args.post_jaccard_weight) * j_norm) / denom
        df["simple_score"] = blended + float(args.post_brand_penalty) * df["brand_score"].astype(float)
        relevance = df.get("_relevance_score", pd.Series([0.0] * len(df)))
        df["final_score"] = (1.0 - float(args.post_weight)) * relevance.fillna(0.0) + float(args.post_weight) * df["simple_score"]
        sort_order = ["brand_score", "final_score", "strength_sim", "jaccard_text"]
        ascending = [False, False, False, False]
        available_sort = [col for col in sort_order if col in df.columns]
        asc = [ascending[sort_order.index(col)] for col in available_sort]
        df = df.sort_values(available_sort, ascending=asc).reset_index(drop=True)

        print("concept_id   | final  | s_sim | jacc | brand | name")
        print("-" * 80)
        for _, row in df.head(args.show_topk).iterrows():
            print(_format_row(row))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive terminology lookup")
    parser.add_argument("--db", required=True, help="Path to LanceDB directory")
    parser.add_argument("--table", required=True, help="LanceDB table name")
    parser.add_argument("--candidate-topk", type=int, default=200, help="Candidate pool size")
    parser.add_argument("--show-topk", type=int, default=20, help="Number of rows to display")
    parser.add_argument("--device", default="auto", help="Device: auto|cuda|mps|cpu (default: auto)")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--post-weight", type=float, default=0.3, help="Weight for simple post-score in final blend (0.0 = ML only)")
    parser.add_argument("--post-strength-weight", type=float, default=0.6, help="Weight for strength feature within simple score")
    parser.add_argument("--post-jaccard-weight", type=float, default=0.4, help="Weight for jaccard feature within simple score")
    parser.add_argument(
        "--post-brand-penalty",
        type=float,
        default=0.3,
        help="Penalty weight applied when candidate brand conflicts with query text",
    )
    parser.add_argument(
        "--post-minmax",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable per-query min-max normalization of simple features (default: enabled)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":  # pragma: no cover
    main()
