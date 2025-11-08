"""Interactive query CLI."""

from __future__ import annotations

import argparse
from typing import Sequence

import pandas as pd

from thirawat_mapper_beta.models import SapBERTEmbedder, ThirawatReranker
from thirawat_mapper_beta.scoring import batch_features
from thirawat_mapper_beta.scoring import post_scorer as ps  # for --debug internals
from thirawat_mapper_beta.utils import connect_table, normalize_text_value
from .conversion import convert_inn_ban_to_usan
from .utils import (
    configure_torch_for_infer,
    minmax_normalize,
    resolve_device,
    rank_candidates,
    enrich_with_post_scores,
    sanitize_query_text,
)


def _format_row(row: pd.Series) -> str:
    concept_id = row.get("concept_id")
    name = row.get("concept_name") or row.get("profile_text")
    final = row.get("final_score")
    retr = row.get("_relevance_score")
    strength = row.get("strength_sim")
    jac = row.get("jaccard_text")
    brand = row.get("brand_score")
    simple = row.get("post_score") if row.get("post_score") is not None else row.get("simple_score")
    if pd.notna(final):
        retr_s = f"{float(retr):6.3f}" if retr is not None and pd.notna(retr) else "  n/a "
        simple_s = f"{float(simple):6.3f}" if simple is not None and pd.notna(simple) else "  n/a "
        return (
            f"{concept_id:<12} | {retr_s} | {simple_s} | {final:6.3f} | {strength:5.3f} | {jac:5.3f} | {brand:5.2f} | {name}"
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

        sanitized = sanitize_query_text(
            query,
            strip_non_latin=bool(args.strip_non_latin),
            strip_chars=str(args.strip_chars or ""),
        )
        if args.convert_inn_to_usan:
            sanitized = convert_inn_ban_to_usan(sanitized)
        query_norm = normalize_text_value(sanitized)
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

        df = enrich_with_post_scores(
            df,
            query_norm,
            post_strength_weight=float(args.post_strength_weight),
            post_jaccard_weight=float(args.post_jaccard_weight),
            post_brand_penalty=float(args.post_brand_penalty),
            post_minmax=bool(args.post_minmax),
            post_weight=float(args.post_weight),
            prefer_brand=True,
        )

        print("concept_id   |  retr  |  post  | final  | s_sim | jacc | brand | name")
        print("-" * 80)
        shown = df.head(args.show_topk)
        for _, row in shown.iterrows():
            print(_format_row(row))

        if args.debug:
            print("\n# Debug details\n")
            q_comp, _ = ps.extract_strengths_with_spans(query_norm)
            print(f"Query strengths: {[ (c.kind, c.value, c.unit, c.denom_value, c.denom_unit) for c in q_comp ]}")
            for _, row in shown.iterrows():
                cid = row.get("concept_id")
                name = row.get("concept_name") or row.get("profile_text")
                raw = ((row.get("concept_name") or "").strip() + " " + (row.get("profile_text") or "").strip()).strip()
                cand_text = normalize_text_value(raw)
                d_comp, _ = ps.extract_strengths_with_spans(cand_text)
                qb = ps._unit_bucket(q_comp)
                db = ps._unit_bucket(d_comp)
                s_dose, p_extra = ps._dose_gate_and_extra(qb, db, tau=0.6, kappa_extra=0.7)
                brand = ps.brand_score(query_norm, raw)
                jacc = ps.jaccard_remainder(query_norm, cand_text)
                ssim = ps.strength_sim(query_norm, cand_text)
                mix = ps.simple_strength_plus_jaccard(query_norm, cand_text)
                post = mix.get("post_score", mix.get("simple_score"))
                final = row.get("final_score")
                retr = row.get("_relevance_score")
                print("-" * 80)
                print(f"CID {cid} | {name}")
                print(f"  strengths(doc): {[ (c.kind, c.value, c.unit, c.denom_value, c.denom_unit) for c in d_comp ]}")
                print(f"  s_dose={s_dose:.3f}  p_extra={p_extra:.3f}  strength_sim={ssim:.3f}")
                print(f"  jacc={jacc:.3f}  brand_score={brand:.3f}  post={float(post):.3f}  retr={float(retr) if retr is not None else float('nan'):.3f}  final={float(final):.3f}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive terminology lookup")
    parser.add_argument("--db", required=True, help="Path to LanceDB directory")
    parser.add_argument("--table", required=True, help="LanceDB table name")
    parser.add_argument("--candidate-topk", type=int, default=200, help="Candidate pool size")
    parser.add_argument("--show-topk", type=int, default=20, help="Number of rows to display")
    parser.add_argument("--device", default="auto", help="Device: auto|cuda|mps|cpu (default: auto)")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    parser.add_argument(
        "--strip-non-latin",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Remove non‑Latin characters from query before retrieval/rerank",
    )
    parser.add_argument(
        "--strip-chars",
        default="",
        help="Characters to remove from query before retrieval/rerank (e.g., '()[]{}').",
    )
    parser.add_argument(
        "--convert-inn-to-usan",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Normalize INN/BAN terms to USAN before performing the lookup.",
    )
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False, help="Print per-candidate scoring details for the shown rows")
    parser.add_argument("--post-weight", type=float, default=0.05, help="Weight for simple post-score in final blend (0.0 = ML only)")
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
