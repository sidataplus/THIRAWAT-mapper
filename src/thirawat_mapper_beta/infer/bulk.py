"""Bulk inference CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import pandas as pd

from thirawat_mapper_beta.io import export_relabel_csv
from thirawat_mapper_beta.models import SapBERTEmbedder, ThirawatReranker
from thirawat_mapper_beta.scoring import batch_features
from thirawat_mapper_beta.utils import connect_table


DEFAULT_TOPK = 100
EVAL_K = (1, 2, 5, 10, 20, 50, 100)


def _load_input(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported input format: {suffix}")


def _prepare_queries(df: pd.DataFrame, name_col: str, code_col: str | None) -> List[str]:
    queries: List[str] = []
    for _, row in df.iterrows():
        name = str(row.get(name_col, "") or "").strip()
        if not name:
            queries.append("")
            continue
        code = str(row.get(code_col, "") or "").strip() if code_col else ""
        if code:
            queries.append(f"{name} ({code})")
        else:
            queries.append(name)
    return queries


def _compute_metrics(
    rows: Iterable[Mapping[str, object]],
    topk: int,
) -> Dict[str, float]:
    rows_list = list(rows)
    hits = {k: 0 for k in EVAL_K}
    mrr = 0.0
    labeled = 0
    coverage = 0

    for row in rows_list:
        candidates = row.get("candidates")
        if isinstance(candidates, pd.DataFrame) and not candidates.empty:
            coverage += 1
        gold = row.get("gold_concept_id")
        if gold is None or (isinstance(gold, float) and pd.isna(gold)):
            continue
        if isinstance(gold, str) and not gold.strip():
            continue
        labeled += 1
        try:
            gold_int = int(gold)  # type: ignore[arg-type]
        except Exception:
            continue
        preds: List[int] = []
        if isinstance(candidates, pd.DataFrame):
            preds = [
                int(v)
                for v in candidates["concept_id"].head(topk).tolist()
                if pd.notna(v)
            ]
        rank = None
        for idx, cid in enumerate(preds, start=1):
            if cid == gold_int:
                rank = idx
                break
        for k in EVAL_K:
            if rank is not None and rank <= k:
                hits[k] += 1
        if rank is not None and rank <= 100:
            mrr += 1.0 / rank

    metrics: Dict[str, float] = {
        "n_rows": float(len(rows_list)),
        "coverage": coverage / max(len(rows_list), 1),
    }
    if labeled:
        for k in EVAL_K:
            metrics[f"hit@{k}"] = hits[k] / labeled
        metrics["mrr@100"] = mrr / labeled
        metrics["n_labeled"] = labeled
    else:
        metrics["n_labeled"] = 0.0
    return metrics


def run(args: argparse.Namespace) -> None:
    table, vector_column = connect_table(args.db, args.table)
    df = _load_input(Path(args.input))
    if df.empty:
        raise SystemExit("Input file contained no rows")

    queries = _prepare_queries(df, args.source_name_column, args.source_code_column)

    embedder = SapBERTEmbedder(device=args.device, batch_size=args.batch_size)
    vectors = embedder.encode(queries)

    reranker = ThirawatReranker(device=args.device, return_score="all", pooling="bms", temperature=20.0)

    results: List[Dict[str, object]] = []

    status_series = df[args.status_column] if args.status_column in df.columns else None

    for idx, query_text in enumerate(queries):
        row = df.iloc[idx]
        query_vec = vectors[idx]
        source_code = str(row.get(args.source_code_column, "") or "") if args.source_code_column else None

        builder = table.search(
            query_vec.astype(float).tolist(),
            vector_column_name=vector_column,
            query_type="vector",
        )
        try:
            arrow_table = (
                builder.distance_type("cosine")
                .limit(args.candidate_topk)
                .rerank(reranker=reranker, query_string=query_text)
                .limit(args.candidate_topk)
                .to_arrow()
            )
            df_candidates = arrow_table.to_pandas()
        except Exception:
            df_candidates = pd.DataFrame(columns=["concept_id", "profile_text", "concept_name"])

        keep_cols = [
            col
            for col in [
                "concept_id",
                "concept_name",
                "domain_id",
                "profile_text",
                "_relevance_score",
            ]
            if col in df_candidates.columns
        ]
        df_candidates = df_candidates.loc[:, keep_cols] if keep_cols else df_candidates

        if not df_candidates.empty:
            features = batch_features(query_text, df_candidates["profile_text"].astype(str).tolist())
            df_candidates["strength_sim"] = [feat["strength_sim"] for feat in features]
            df_candidates["jaccard_text"] = [feat["jaccard_text"] for feat in features]
            # Per-query min-max normalization for simple features
            def _minmax(col: pd.Series) -> pd.Series:
                vmin = float(col.min())
                vmax = float(col.max())
                rng = vmax - vmin
                if rng <= 1e-9:
                    return pd.Series([0.0] * len(col), index=col.index)
                return (col - vmin) / rng

            s_norm = _minmax(df_candidates["strength_sim"].astype(float))
            j_norm = _minmax(df_candidates["jaccard_text"].astype(float))
            df_candidates["simple_score"] = 0.6 * s_norm + 0.4 * j_norm
            relevance = df_candidates.get("_relevance_score", pd.Series([0.0] * len(df_candidates)))
            df_candidates["final_score"] = 0.7 * relevance.fillna(0.0) + 0.3 * df_candidates["simple_score"]
            df_candidates = df_candidates.sort_values(
                ["final_score", "strength_sim", "jaccard_text"], ascending=[False, False, False]
            ).reset_index(drop=True)
        else:
            df_candidates = pd.DataFrame(columns=["concept_id", "concept_name", "profile_text", "final_score"])

        gold = None
        if args.label_column and args.label_column in df.columns:
            value = row.get(args.label_column)
            if pd.notna(value) and str(value).strip():
                gold = value
        if gold is None and status_series is not None and args.status_column in df.columns:
            status_value = str(row.get(args.status_column, "") or "").strip()
            if status_value.upper() == args.approved_value.upper():
                fallback_col = args.label_column if args.label_column in df.columns else "conceptId"
                gold = row.get(fallback_col)

        record: Dict[str, object] = {
            "source_name": row.get(args.source_name_column),
            "source_code": source_code,
            "candidates": df_candidates.head(args.candidate_topk),
            "input_row": row.to_dict(),
        }
        if gold is not None and gold != "":
            record["gold_concept_id"] = gold

        results.append(record)

    metrics = _compute_metrics(results, args.candidate_topk)

    export_relabel_csv(
        results,
        args.out,
        topk=args.candidate_topk,
        metrics=metrics,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bulk terminology inference")
    parser.add_argument("--db", required=True, help="Path to LanceDB directory")
    parser.add_argument("--table", required=True, help="LanceDB table name")
    parser.add_argument("--input", required=True, help="Input CSV/TSV/Parquet file")
    parser.add_argument("--out", required=True, help="Directory for outputs")
    parser.add_argument("--candidate-topk", type=int, default=DEFAULT_TOPK, help="Candidates before rerank")
    parser.add_argument("--device", default=None, help="torch device (cuda or cpu)")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--source-name-column", default="sourceName", help="Column containing source names")
    parser.add_argument("--source-code-column", default="sourceCode", help="Column containing source codes")
    parser.add_argument(
        "--label-column",
        default="conceptId",
        help="Column with gold concept IDs (optional)",
    )
    parser.add_argument(
        "--status-column",
        default="mappingStatus",
        help="Column containing Usagi mapping status (optional)",
    )
    parser.add_argument(
        "--approved-value",
        default="APPROVED",
        help="Value in status column treated as gold label",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":  # pragma: no cover
    main()
