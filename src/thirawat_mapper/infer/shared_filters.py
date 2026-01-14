"""Shared filtering helpers reused across inference CLIs."""

from __future__ import annotations

import re
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


def _parse_atc_cell(value: object, *, as_int: bool) -> list[str] | list[int]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    items: list[object]
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        text = str(value).strip()
        if not text:
            return []
        items = [part for part in re.split(r"[,\s;|]+", text) if part]

    out: list[str] | list[int] = []
    for item in items:
        if item is None:
            continue
        raw = str(item).strip()
        if not raw:
            continue
        if as_int:
            if raw.isdigit():
                out.append(int(raw))  # type: ignore[arg-type]
        else:
            out.append(raw.upper())  # type: ignore[arg-type]
    return out


class AtcScopeResolver:
    """Resolve per-row ATC allowlists via DuckDB vocab tables."""

    def __init__(self, vocab_path: Path) -> None:
        self.vocab_path = Path(vocab_path)
        self._conn = None
        self._failed = False
        self._code_to_id: dict[str, int] | None = None
        self._valid_ids: set[int] | None = None
        self._desc_cache: dict[int, set[int]] = {}

    def _ensure_conn(self) -> None:
        if self._conn is None:
            import duckdb  # type: ignore

            self._conn = duckdb.connect(str(self.vocab_path), read_only=True)

    def _load_atc_vocab(self) -> tuple[dict[str, int], set[int]]:
        if self._code_to_id is not None and self._valid_ids is not None:
            return self._code_to_id, self._valid_ids
        self._ensure_conn()
        df = self._conn.execute(
            """
            SELECT concept_id, concept_code
            FROM concept
            WHERE vocabulary_id = 'ATC'
              AND invalid_reason IS NULL
            """
        ).fetch_df()
        code_to_id = {str(code).strip().upper(): int(cid) for cid, code in zip(df["concept_id"], df["concept_code"])}
        valid_ids = set(int(cid) for cid in df["concept_id"].tolist())
        self._code_to_id = code_to_id
        self._valid_ids = valid_ids
        return code_to_id, valid_ids

    def _load_atc_descendants(self, atc_ids: set[int]) -> dict[int, set[int]]:
        missing = sorted({int(i) for i in atc_ids if int(i) not in self._desc_cache})
        if not missing:
            return {int(i): set(self._desc_cache.get(int(i), set())) for i in atc_ids}
        self._ensure_conn()
        id_list = ",".join(str(int(i)) for i in missing)
        sql = f"""
        SELECT ca.ancestor_concept_id AS atc_id,
               ca.descendant_concept_id AS concept_id
        FROM concept_ancestor ca
        JOIN concept d ON d.concept_id = ca.descendant_concept_id
        WHERE ca.ancestor_concept_id IN ({id_list})
          AND d.standard_concept = 'S'
          AND d.invalid_reason IS NULL
          AND d.domain_id = 'Drug'
        """
        df = self._conn.execute(sql).fetch_df()
        for atc_id in missing:
            self._desc_cache[int(atc_id)] = set()
        for atc_id, concept_id in zip(df["atc_id"], df["concept_id"]):
            self._desc_cache[int(atc_id)].add(int(concept_id))
        return {int(i): set(self._desc_cache.get(int(i), set())) for i in atc_ids}

    def resolve_atc_ids(self, df: pd.DataFrame) -> dict[int, list[int]]:
        if "atc_ids" not in df.columns and "atc_codes" not in df.columns:
            return {}
        try:
            code_to_id, valid_ids = self._load_atc_vocab()
        except Exception as exc:
            print(f"[warn] Failed to load ATC vocab from DuckDB: {exc}")
            self._failed = True
            return {}

        resolved: dict[int, list[int]] = {}
        for idx, row in enumerate(df.itertuples(index=False), start=0):
            ids: list[int] = []
            if "atc_ids" in df.columns:
                ids.extend(_parse_atc_cell(getattr(row, "atc_ids", None), as_int=True))  # type: ignore[arg-type]
            if "atc_codes" in df.columns:
                codes = _parse_atc_cell(getattr(row, "atc_codes", None), as_int=False)  # type: ignore[arg-type]
                ids.extend([code_to_id[c] for c in codes if c in code_to_id])  # type: ignore[arg-type]
            cleaned = sorted({int(i) for i in ids if int(i) in valid_ids})
            if cleaned:
                resolved[idx] = cleaned
        return resolved

    def build_allowlist(self, df: pd.DataFrame, *, allowlist_max_ids: int = 1000) -> dict[int, set[int]]:
        atc_ids_by_row = self.resolve_atc_ids(df)
        if not atc_ids_by_row:
            return {}
        all_atc_ids: set[int] = set()
        for ids in atc_ids_by_row.values():
            all_atc_ids.update(ids)
        try:
            descendants = self._load_atc_descendants(all_atc_ids)
        except Exception as exc:
            print(f"[warn] Failed to resolve ATC descendants from DuckDB: {exc}")
            self._failed = True
            return {}

        allowlist: dict[int, set[int]] = {}
        for row_idx, ids in atc_ids_by_row.items():
            merged: set[int] = set()
            for atc_id in ids:
                merged.update(descendants.get(int(atc_id), set()))
            if not merged:
                continue
            if allowlist_max_ids > 0 and len(merged) > int(allowlist_max_ids):
                print(f"[warn] ATC allowlist too large for row {row_idx} ({len(merged)} ids); skipping ATC scope.")
                continue
            allowlist[row_idx] = merged
        return allowlist


__all__ = ["safe_int", "to_exclusion_set", "ConceptClassResolver", "AtcScopeResolver"]
