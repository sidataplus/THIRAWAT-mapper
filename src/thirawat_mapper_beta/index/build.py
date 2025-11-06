"""CLI for building LanceDB indexes from Athena→DuckDB exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pyarrow as pa

from thirawat_mapper_beta.io import read_concept_profiles
from thirawat_mapper_beta.models import SapBERTEmbedder
from thirawat_mapper_beta.utils import normalize_text_value


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


def build_index(args: argparse.Namespace) -> None:
    domain_ids = _normalize_multi_value(args.domain_id)
    concept_class_ids = _normalize_multi_value(args.concept_class_id)
    extra_columns = _normalize_multi_value(args.extra_column)

    df = read_concept_profiles(
        args.duckdb,
        args.profiles_table,
        concepts_table=args.concepts_table,
        domain_ids=domain_ids,
        concept_class_ids=concept_class_ids,
        extra_profile_columns=extra_columns,
    )

    if df.empty:
        raise SystemExit("No rows matched the provided filters; nothing to index")

    # Normalize profile_text for indexing
    df["profile_text"] = df["profile_text"].astype("string").apply(normalize_text_value)

    embedder = SapBERTEmbedder(device=args.device, batch_size=args.batch_size)
    vectors = embedder.encode(df["profile_text"].tolist())

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
        "extra_columns": list(extra_columns),
        "vector_dim": vectors.shape[1],
        "count": len(df),
        "model_id": embedder.model_id,
    }
    manifest_path = db_path / f"{args.table}_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a LanceDB index from DuckDB profiles")
    parser.add_argument("--duckdb", required=True, help="Path to duckdb file produced by athena2duckdb")
    parser.add_argument("--profiles-table", required=True, help="Table containing concept_id and profile_text")
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
        "--extra-column",
        action="append",
        default=[],
        help="Additional profile columns (comma-separated or repeat flag) to carry into the LanceDB table",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Embedding batch size")
    parser.add_argument("--device", default=None, help="torch device, e.g. cuda or cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    build_index(args)


if __name__ == "__main__":  # pragma: no cover
    main()
