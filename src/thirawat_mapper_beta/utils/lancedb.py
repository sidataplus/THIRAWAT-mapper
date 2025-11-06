"""Helpers for interacting with LanceDB tables."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import lancedb
import pyarrow as pa


def _is_fixed_size_vector(field: pa.Field) -> bool:
    return pa.types.is_fixed_size_list(field.type) and pa.types.is_float32(
        field.type.value_type
    )


def connect_table(db_dir: Path | str, table_name: str):
    """Return a LanceDB table ensuring a compatible vector column exists."""

    db_path = Path(db_dir)
    if not db_path.exists():
        raise FileNotFoundError(f"LanceDB directory not found: {db_path}")

    db = lancedb.connect(str(db_path))
    table = db.open_table(table_name)

    schema = table.schema
    vector_field: Tuple[str, pa.Field] | None = None
    for field in schema:
        if _is_fixed_size_vector(field):
            vector_field = (field.name, field)
            break

    if vector_field is None:
        raise ValueError(
            "LanceDB table must contain a FixedSizeList<float32> vector column"
        )

    return table, vector_field[0]
