"""Shared filtering helpers reused across inference CLIs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Set

import pandas as pd


def safe_int(value: object) -> Optional[int]:
    """Convert values to ``int`` while tolerating pandas NA/object noise."""

    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def to_exclusion_set(values: Sequence[str] | None) -> Set[str]:
    """Normalize a sequence of comma-separated strings into a set."""

    excluded: Set[str] = set()
    if not values:
        return excluded
    for raw in values:
        if not raw:
            continue
        for piece in str(raw).split(","):
            item = piece.strip()
            if item:
                excluded.add(item)
    return excluded


class ConceptClassResolver:
    """Resolve ``concept_class_id`` values via DuckDB metadata when needed."""

    def __init__(self, duckdb_path: Path, concepts_table: str) -> None:
        self.duckdb_path = duckdb_path
        self.concepts_table = concepts_table
        self._conn = None
        self._failed = False
        self._cache: Dict[int, Optional[str]] = {}

    def _ensure_conn(self) -> None:
        if self._conn is None:
            try:
                import duckdb  # type: ignore
            except ImportError:  # pragma: no cover - optional dependency
                raise RuntimeError("duckdb is required for this feature.")
            self._conn = duckdb.connect(str(self.duckdb_path))

    def lookup(self, concept_ids: Sequence[int]) -> Dict[int, Optional[str]]:
        pending = [cid for cid in concept_ids if cid not in self._cache]
        if pending and not self._failed:
            try:
                self._ensure_conn()
                if not pending:
                    pass
                placeholders = ",".join(str(int(cid)) for cid in pending)
                query = f"SELECT concept_id, concept_class_id FROM {self.concepts_table} WHERE concept_id IN ({placeholders})"
                df = self._conn.execute(query).df()  # type: ignore[union-attr]
                fetched = {
                    int(row["concept_id"]): (str(row["concept_class_id"]) if row["concept_class_id"] is not None else None)
                    for _, row in df.iterrows()
                }
                for cid in pending:
                    self._cache[cid] = fetched.get(cid)
            except Exception as exc:  # pragma: no cover - defensive logging in callers
                print(f"[warn] Failed to resolve concept_class_id from DuckDB: {exc}")
                self._failed = True
                for cid in pending:
                    self._cache[cid] = None
        return {cid: self._cache.get(cid) for cid in concept_ids}

__all__ = ["safe_int", "to_exclusion_set", "ConceptClassResolver"]
