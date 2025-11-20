"""Interactive query CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from thirawat_mapper_beta.models import SapBERTEmbedder, ThirawatReranker
from thirawat_mapper_beta.scoring import post_scorer as ps  # for --debug internals
from thirawat_mapper_beta.utils import connect_table, normalize_text_value
from .conversion import convert_inn_ban_to_usan
from .shared_filters import ConceptClassResolver, safe_int, to_exclusion_set
from .utils import (
    configure_torch_for_infer,
    resolve_device,
    enrich_with_post_scores,
    sanitize_query_text,
    normalize_strength_spacing,
)


DEFAULT_TOPK = 100


def _prepare_query_text(text: str, args: argparse.Namespace) -> str:
    value = text
    if args.strip_non_latin or args.strip_chars:
        value = sanitize_query_text(
            value,
            strip_non_latin=bool(args.strip_non_latin),
            strip_chars=str(args.strip_chars or ""),
        )
    if args.convert_inn_to_usan:
        value = convert_inn_ban_to_usan(value)
    value = normalize_strength_spacing(value)
    return value


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

    manifest_path = Path(args.db) / f"{args.table}_manifest.json"
    duckdb_path: str | None = None
    concepts_table: str | None = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            duckdb_path = manifest.get("duckdb")
            concepts_table = manifest.get("concepts_table")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[warn] Failed to read manifest metadata: {exc}")

    exclude_concept_class_ids = to_exclusion_set(getattr(args, "exclude_concept_class_id", None))
    concept_class_resolver: ConceptClassResolver | None = None
    if exclude_concept_class_ids:
        if duckdb_path and concepts_table:
            concept_class_resolver = ConceptClassResolver(Path(duckdb_path), str(concepts_table))
        else:
            print("[warn] Concept class exclusions unavailable (missing duckdb manifest info).")

    def _execute_query(raw_query: str) -> tuple[str, pd.DataFrame | None]:
        sanitized = _prepare_query_text(raw_query, args)
        query_norm = normalize_text_value(sanitized)
        vector = embedder.encode([query_norm])[0]
        builder = table.search(
            vector.astype(float).tolist(),
            vector_column_name=vector_column,
            query_type="vector",
        )
        if args.where:
            try:
                builder = builder.where(args.where)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[warn] Ignoring where clause due to error: {exc}")
        try:
            retrieval_topk = max(args.candidate_topk, args.retrieval_topk)
            result_builder = builder.distance_type("cosine").limit(retrieval_topk)
            if not args.no_rerank:
                result_builder = result_builder.rerank(reranker=reranker, query_string=query_norm)
            result_table = result_builder.limit(args.candidate_topk).to_arrow()
            df = result_table.to_pandas()
        except Exception as exc:  # pragma: no cover - interactive error path
            print(f"Error running search: {exc}")
            return query_norm, None

        keep_cols = [
            col
            for col in ["concept_id", "concept_name", "domain_id", "concept_class_id", "vocabulary_id", "profile_text", "_relevance_score"]
            if col in df.columns
        ]
        if keep_cols:
            df = df.loc[:, keep_cols]

        if exclude_concept_class_ids and not df.empty:
            if "concept_class_id" not in df.columns and concept_class_resolver is not None:
                concept_ids_list = [cid for cid in df["concept_id"].apply(safe_int).tolist() if cid is not None]
                if concept_ids_list:
                    mapping = concept_class_resolver.lookup(concept_ids_list)
                    df["concept_class_id"] = [mapping.get(safe_int(cid)) for cid in df["concept_id"].tolist()]
            if "concept_class_id" in df.columns:
                df = df[
                    ~df["concept_class_id"].astype(str).str.strip().isin(exclude_concept_class_ids)
                ].reset_index(drop=True)

        if df.empty:
            return query_norm, df

        df = enrich_with_post_scores(
            df,
            query_norm,
            post_strength_weight=float(args.post_strength_weight),
            post_jaccard_weight=float(args.post_jaccard_weight),
            post_brand_penalty=float(args.post_brand_penalty),
            post_minmax=bool(args.post_minmax),
            post_weight=float(args.post_weight),
            prefer_brand=True,
            rank_output=False,
        )
        return query_norm, df

    def _print_results(query_norm: str, df: pd.DataFrame) -> pd.DataFrame:
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
                print(
                    f"  jacc={jacc:.3f}  brand_score={brand:.3f}  post={float(post):.3f}  retr={float(retr) if retr is not None else float('nan'):.3f}  final={float(final):.3f}"
                )
        return shown

    def _check_expectation(df: pd.DataFrame) -> bool:
        if not args.expect_name:
            return True
        top = df.iloc[0]
        top_name = str(top.get("concept_name") or top.get("profile_text") or "").strip()
        expected = args.expect_name.strip()
        if top_name.lower() == expected.lower():
            print(f"[ok] Top concept '{top_name}' matched expectation '{expected}'.")
            return True
        print(f"[fail] Top concept '{top_name}' did not match expected '{expected}'.")
        return False

    if args.query:
        query_norm, df = _execute_query(args.query)
        if df is None:
            raise SystemExit(1)
        if df.empty:
            print("No matches found.")
            raise SystemExit(1 if args.expect_name else 0)
        _print_results(query_norm, df)
        raise SystemExit(0 if _check_expectation(df) else 1)

    if args.expect_name:
        print("[warn] --expect-name is ignored without --query.")

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

        query_norm, df = _execute_query(query)
        if df is None:
            continue
        if df.empty:
            print("No matches found.")
            continue
        _print_results(query_norm, df)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive terminology lookup")
    parser.add_argument("--db", required=True, help="Path to LanceDB directory")
    parser.add_argument("--table", required=True, help="LanceDB table name")
    parser.add_argument("--candidate-topk", type=int, default=DEFAULT_TOPK, help="Candidate pool size")
    parser.add_argument(
        "--retrieval-topk",
        type=int,
        default=200,
        help="Number of vector candidates fetched before rerank (>= candidate-topk).",
    )
    parser.add_argument(
        "--no-rerank",
        action=argparse.BooleanOptionalAction,
        help="Skip reranking and rely on vector similarity only.",
    )
    parser.add_argument("--show-topk", type=int, default=20, help="Number of rows to display")
    parser.add_argument("--device", default="cpu", help="Device: auto|cuda|mps|cpu (default: cpu for stability)")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--where", default=None, help="Optional LanceDB filter expression, e.g. vocabulary_id = 'RxNorm'")
    parser.add_argument("--query", default=None, help="Run a single lookup and exit (non-interactive mode)")
    parser.add_argument(
        "--expect-name",
        default=None,
        help="When used with --query, require the top concept name to match this value (case-insensitive).",
    )
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
        default=True,
        help="Normalize INN/BAN terms to USAN before performing the lookup.",
    )
    parser.add_argument(
        "--exclude-concept-class-id",
        action="append",
        default=[],
        help="Concept class IDs to exclude from candidate results (comma-separated or repeat flag).",
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
