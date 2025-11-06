"""Utilities for reading Athena→DuckDB exports."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import duckdb
import pandas as pd


def _as_list(values: Optional[Iterable[str]]) -> List[str]:
    if not values:
        return []
    if isinstance(values, (str, bytes)):
        return [str(values)]
    return [str(v) for v in values]


def read_concept_profiles(
    duckdb_path: Path | str,
    profiles_table: str,
    *,
    concepts_table: Optional[str] = None,
    domain_ids: Optional[Iterable[str]] = None,
    concept_class_ids: Optional[Iterable[str]] = None,
    extra_profile_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Return concept profiles with optional domain/class filters.

    Parameters
    ----------
    duckdb_path:
        Path to the DuckDB file created via ``athena2duckdb``.
    profiles_table:
        Table containing ``concept_id`` and ``profile_text`` columns.
    concepts_table:
        Optional OMOP ``concept`` table enabling ``domain_id`` and
        ``concept_class_id`` filtering.
    domain_ids / concept_class_ids:
        Iterable of allowed values. When provided, ``concepts_table`` must be
        set. Values are compared case-sensitively.
    extra_profile_columns:
        Additional columns to select from ``profiles_table`` if present.

    Returns
    -------
    pandas.DataFrame
        Columns: ``concept_id``, ``profile_text``, plus any available extras.
    """

    domain_ids_list = _as_list(domain_ids)
    class_ids_list = _as_list(concept_class_ids)

    if (domain_ids_list or class_ids_list) and not concepts_table:
        raise ValueError(
            "concepts_table must be provided when domain_id or concept_class_id "
            "filters are requested"
        )

    duckdb_path = Path(duckdb_path)
    if not duckdb_path.exists():
        raise FileNotFoundError(f"DuckDB database not found: {duckdb_path}")

    select_cols = ["p.concept_id", "p.profile_text"]
    for col in extra_profile_columns:
        select_cols.append(f"p.{col}")

    query = ["SELECT ", ", ".join(select_cols), f" FROM {profiles_table} p"]

    filters: List[str] = []
    if concepts_table:
        query.append(f" JOIN {concepts_table} c USING(concept_id)")
        if domain_ids_list:
            values = ",".join(f"'{v}'" for v in domain_ids_list)
            filters.append(f"c.domain_id IN ({values})")
        if class_ids_list:
            values = ",".join(f"'{v}'" for v in class_ids_list)
            filters.append(f"c.concept_class_id IN ({values})")

    if filters:
        query.append(" WHERE ")
        query.append(" AND ".join(filters))

    sql = "".join(query)

    con = duckdb.connect(str(duckdb_path), read_only=True)
    with contextlib.closing(con):
        try:
            df = con.execute(sql).df()
        except duckdb.CatalogException as exc:  # table or column missing
            raise ValueError(f"Failed to execute query: {sql}") from exc

    # Ensure required columns exist
    if "concept_id" not in df.columns or "profile_text" not in df.columns:
        raise ValueError(
            "profiles_table must contain concept_id and profile_text columns"
        )
    df["concept_id"] = df["concept_id"].astype("int64", copy=False)
    df["profile_text"] = df["profile_text"].astype("string", copy=False)
    return df
