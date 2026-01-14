"""CLI for building LanceDB indexes from Athena→DuckDB exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa

from thirawat_mapper.io import read_concept_profiles
from thirawat_mapper.models import SapBERTEmbedder
from thirawat_mapper.models.embedder import DEFAULT_MODEL_ID
from thirawat_mapper.utils import normalize_text_value

EXCLUDED_CONCEPT_CLASSES = (
    "Gene DNA Variant",
    "Gene Protein Variant",
    "Gene RNA Variant",
    "Variant",
)


def _normalize_multi_value(values: Sequence[str]) -> list[str]:
    items: list[str] = []
    for value in values:
        if value is None:
            continue
        parts = [part.strip() for part in str(value).split(",")]
        items.extend([part for part in parts if part])
    return items


def _fixed_size_list(vectors: np.ndarray) -> pa.FixedSizeListArray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("vectors array must be 2-dimensional")
    dim = vectors.shape[1]
    flat = pa.array(vectors.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, dim)


def _quote_literals(values: Sequence[str]) -> str:
    escaped: list[str] = []
    for value in values:
        text = str(value).replace("'", "''")
        escaped.append(f"'{text}'")
    return ", ".join(escaped)


def _build_concept_filters(
    domain_ids: Sequence[str],
    concept_class_ids: Sequence[str],
    exclude_concept_class_ids: Sequence[str],
) -> list[str]:
    conditions: list[str] = ["c.standard_concept = 'S'", "c.invalid_reason IS NULL"]

    if domain_ids:
        conditions.append(f"c.domain_id IN ({_quote_literals(domain_ids)})")
    if concept_class_ids:
        conditions.append(f"c.concept_class_id IN ({_quote_literals(concept_class_ids)})")

    combined_excludes = list(EXCLUDED_CONCEPT_CLASSES)
    combined_excludes.extend(exclude_concept_class_ids)
    if combined_excludes:
        conditions.append(f"c.concept_class_id NOT IN ({_quote_literals(combined_excludes)})")

    return conditions


def _duckdb_table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    try:
        con.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
    except duckdb.CatalogException:
        return False
    return True


def _build_profile_sql(
    concepts_table: str,
    filters: list[str],
    *,
    include_synonyms: bool,
    synonyms_table: str,
) -> str:
    base_query = f"""
        SELECT c.concept_id,
               c.concept_name,
               c.concept_code,
               c.domain_id,
               c.concept_class_id,
               c.vocabulary_id
        FROM {concepts_table} c
        WHERE {' AND '.join(filters)}
    """

    if include_synonyms:
        return f"""
            WITH base AS ({base_query}),
            syn AS (
                SELECT concept_id,
                       LIST(DISTINCT concept_synonym_name) AS synonyms
                FROM {synonyms_table}
                WHERE concept_synonym_name IS NOT NULL
                GROUP BY 1
            )
            SELECT base.*, COALESCE(syn.synonyms, []) AS synonyms
            FROM base
            LEFT JOIN syn USING (concept_id)
        """

    return f"""
        WITH base AS ({base_query})
        SELECT base.*, [] AS synonyms
        FROM base
    """


def _normalize_optional_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return normalize_text_value(value)


def _prepare_synonym_names(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, tuple):
        items = list(raw)
    else:
        try:
            items = list(raw)
        except TypeError:
            items = [raw]

    names: list[str] = []
    for item in items:
        name = _normalize_optional_text(item)
        if name:
            names.append(name)
    return names


def _dedupe_synonyms(names: Sequence[str], primary_name: str) -> list[str]:
    seen = {primary_name} if primary_name else set()
    deduped: list[str] = []
    for name in names:
        if not name or name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def _compose_profile_text(
    name: str,
    synonyms: Sequence[str],
    max_synonyms: int,
    include_code: bool,
    code: str,
) -> str:
    parts: list[str] = []
    if name:
        parts.append(name)

    trimmed_synonyms: list[str] = []
    limit = max(0, max_synonyms)
    if limit > 0 and synonyms:
        trimmed_synonyms = list(synonyms[:limit])
        if trimmed_synonyms:
            parts.append("; ".join(trimmed_synonyms))

    if include_code and code:
        parts.append(code)

    return " | ".join(parts)


def _build_profiles_inline(
    duckdb_path: Path | str,
    *,
    concepts_table: str | None,
    domain_ids: Sequence[str],
    concept_class_ids: Sequence[str],
    exclude_concept_class_ids: Sequence[str],
    max_synonyms: int,
    include_codes_in_text: bool,
) -> pd.DataFrame:
    table_name = concepts_table or "concept"
    con = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        has_synonyms = _duckdb_table_exists(con, "concept_synonym")
        filters = _build_concept_filters(domain_ids, concept_class_ids, exclude_concept_class_ids)
        sql = _build_profile_sql(
            table_name,
            filters,
            include_synonyms=has_synonyms,
            synonyms_table="concept_synonym",
        )
        df = con.execute(sql).fetch_df()
    finally:
        con.close()

    if df.empty:
        df["profile_text"] = pd.Series(dtype="string")
        return df

    df["concept_id"] = df["concept_id"].astype("int64", copy=False)
    df["concept_name"] = df["concept_name"].apply(_normalize_optional_text)
    df["concept_code"] = df["concept_code"].apply(_normalize_optional_text)
    df["synonyms"] = df["synonyms"].apply(_prepare_synonym_names)
    df["synonyms"] = df.apply(
        lambda row: _dedupe_synonyms(row["synonyms"], row["concept_name"]),
        axis=1,
    )

    df["profile_text"] = df.apply(
        lambda row: _compose_profile_text(
            row["concept_name"],
            row["synonyms"],
            max_synonyms=max_synonyms,
            include_code=include_codes_in_text,
            code=row["concept_code"],
        ),
        axis=1,
    )

    df = df[df["profile_text"].astype(str).str.len() > 0].copy()
    df.drop(columns=["synonyms"], inplace=True)
    df["profile_text"] = df["profile_text"].astype("string", copy=False)
    return df


def _load_profiles(
    args: argparse.Namespace,
    *,
    domain_ids: Sequence[str],
    concept_class_ids: Sequence[str],
    exclude_concept_class_ids: Sequence[str],
    extra_columns: Sequence[str],
) -> pd.DataFrame:
    try:
        return read_concept_profiles(
            args.duckdb,
            args.profiles_table,
            concepts_table=args.concepts_table,
            domain_ids=domain_ids,
            concept_class_ids=concept_class_ids,
            exclude_concept_class_ids=exclude_concept_class_ids,
            extra_profile_columns=extra_columns,
        )
    except (duckdb.CatalogException, ValueError):
        print(
            f"profiles_table '{args.profiles_table}' not found; building profiles inline from concept data"
        )
        return _build_profiles_inline(
            args.duckdb,
            concepts_table=args.concepts_table,
            domain_ids=domain_ids,
            concept_class_ids=concept_class_ids,
            exclude_concept_class_ids=exclude_concept_class_ids,
            max_synonyms=args.max_synonyms,
            include_codes_in_text=args.include_codes_in_text,
        )


def build_index(args: argparse.Namespace) -> None:
    domain_ids = _normalize_multi_value(args.domain_id)
    concept_class_ids = _normalize_multi_value(args.concept_class_id)
    exclude_concept_class_ids = _normalize_multi_value(args.exclude_concept_class_id)
    extra_columns = _normalize_multi_value(args.extra_column)

    df = _load_profiles(
        args,
        domain_ids=domain_ids,
        concept_class_ids=concept_class_ids,
        exclude_concept_class_ids=exclude_concept_class_ids,
        extra_columns=extra_columns,
    )

    if df.empty:
        raise SystemExit("No rows matched the provided filters; nothing to index")

    # Normalize profile_text for indexing
    df["profile_text"] = df["profile_text"].astype("string").apply(normalize_text_value)

    embedder = SapBERTEmbedder(
        model_id=str(args.model_id or DEFAULT_MODEL_ID),
        device=args.device,
        batch_size=args.batch_size,
        max_length=int(args.max_length),
        pooling=str(args.pooling),
        trust_remote_code=bool(args.trust_remote_code),
    )
    vectors = embedder.encode(df["profile_text"].tolist(), progress=True)

    table_data = {
        "concept_id": pa.array(df["concept_id"].astype("int64")),
        "profile_text": pa.array(df["profile_text"].astype("string")),
        "vector": _fixed_size_list(vectors),
    }

    for column in extra_columns:
        if column in df.columns:
            table_data[column] = pa.array(df[column])

    table = pa.table(table_data)

    db_path = Path(args.out_db)
    db_path.mkdir(parents=True, exist_ok=True)

    import lancedb

    db = lancedb.connect(str(db_path))
    db.create_table(args.table, data=table, mode="overwrite")

    manifest = {
        "duckdb": str(Path(args.duckdb).resolve()),
        "profiles_table": args.profiles_table,
        "concepts_table": args.concepts_table,
        "domain_id": list(domain_ids),
        "concept_class_id": list(concept_class_ids),
        "exclude_concept_class_id": list(exclude_concept_class_ids),
        "extra_columns": list(extra_columns),
        "max_synonyms": args.max_synonyms,
        "include_codes_in_text": bool(args.include_codes_in_text),
        "vector_dim": vectors.shape[1],
        "count": len(df),
        "model_id": embedder.model_id,
        "pooling": embedder.pooling,
        "max_length": embedder.max_length,
        "trust_remote_code": embedder.trust_remote_code,
    }
    manifest_path = db_path / f"{args.table}_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a LanceDB index from DuckDB profiles")
    parser.add_argument("--duckdb", required=True, help="Path to duckdb file produced by athena2duckdb")
    parser.add_argument(
        "--profiles-table",
        required=True,
        help="Table containing concept_id and profile_text (auto-built inline if missing)",
    )
    parser.add_argument("--concepts-table", help="OMOP concept table for applying filters")
    parser.add_argument("--out-db", required=True, help="Directory to create or update LanceDB database")
    parser.add_argument("--table", required=True, help="LanceDB table name to create")
    parser.add_argument(
        "--domain-id",
        action="append",
        default=[],
        help="Domain ID filter (comma-separated or repeat flag for multiples)",
    )
    parser.add_argument(
        "--concept-class-id",
        action="append",
        default=[],
        help="Concept class ID filter (comma-separated or repeat flag for multiples)",
    )
    parser.add_argument(
        "--exclude-concept-class-id",
        action="append",
        default=[],
        help="Concept class ID exclusion list (comma-separated or repeat flag)",
    )
    parser.add_argument(
        "--extra-column",
        action="append",
        default=[],
        help="Additional profile columns (comma-separated or repeat flag) to carry into the LanceDB table",
    )
    parser.add_argument(
        "--max-synonyms",
        type=int,
        default=3,
        help="Maximum number of synonyms appended when building profile_text inline (default: 3)",
    )
    parser.add_argument(
        "--include-codes-in-text",
        action="store_true",
        default=False,
        help="Append concept_code to profile_text when profiles are built inline",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Embedding batch size")
    parser.add_argument("--device", default=None, help="torch device, e.g. cuda or cpu")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model id for the encoder")
    parser.add_argument(
        "--pooling",
        choices=["cls", "mean"],
        default="cls",
        help="Pooling strategy for encoder outputs (default: cls for SapBERT)",
    )
    parser.add_argument("--max-length", type=int, default=128, help="Maximum token length for encoder inputs")
    parser.add_argument("--trust-remote-code", action="store_true", help="Enable trust_remote_code for HF model loading")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    build_index(args)


if __name__ == "__main__":  # pragma: no cover
    main()
