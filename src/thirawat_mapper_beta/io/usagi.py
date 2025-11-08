"""Helpers for working with Usagi-formatted rows."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

USAGI_REQUIRED_COLUMNS: tuple[str, ...] = (
    "sourceName",
    "sourceCode",
    "mappingStatus",
    "matchScore",
)


class UsagiRow(BaseModel):
    """Minimal Usagi schema derived from the reference CSV sample."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True, extra="allow")

    sourceCode: str = Field(..., alias="sourceCode")
    sourceName: str = Field(..., alias="sourceName")
    sourceFrequency: float | int | None = None
    sourceAutoAssignedConceptIds: str | None = ""
    matchScore: float = 0.0
    mappingStatus: str = "UNCHECKED"
    equivalence: str | None = "EQUIVALENT"
    statusSetBy: str | None = None
    statusSetOn: float | int | None = None
    conceptId: int | None = None
    conceptName: str | None = None
    domainId: str | None = None
    mappingType: str = "MAPS_TO"
    comment: str | None = None
    createdBy: str | None = None
    createdOn: float | int | None = None
    assignedReviewer: str | None = None

    @field_validator("sourceCode", "sourceName", mode="before")
    @classmethod
    def _require_non_empty(cls, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("value must be a non-empty string")
        return text

    @field_validator("mappingStatus", mode="before")
    @classmethod
    def _default_mapping_status(cls, value: Any) -> str:
        text = str(value or "UNCHECKED").strip()
        return text or "UNCHECKED"

    @field_validator("matchScore", mode="before")
    @classmethod
    def _coerce_match_score(cls, value: Any) -> float:
        if value is None or value == "":
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError("matchScore must be numeric") from exc


def is_usagi_format(columns: Sequence[Any]) -> bool:
    """Return True if the dataframe columns look like a Usagi export."""

    normalized = {str(col) for col in columns}
    return set(USAGI_REQUIRED_COLUMNS).issubset(normalized)


def validate_usagi_frame(df: pd.DataFrame, sample_size: int = 50) -> bool:
    """Validate up to ``sample_size`` rows using the Usagi schema.

    Returns True when the dataframe satisfies the schema, otherwise raises a
    ValueError detailing the first failing row.
    """

    if df.empty:
        return False
    if not is_usagi_format(df.columns):
        return False
    sample = df.head(min(len(df), sample_size))
    for idx, row in sample.iterrows():
        try:
            cleaned = {k: _clean_scalar(v) for k, v in row.to_dict().items()}
            UsagiRow.model_validate(cleaned)
        except ValidationError as exc:  # pragma: no cover - validation path
            raise ValueError(f"Row {idx} is not a valid Usagi record: {exc}") from exc
    return True


def _clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return None
    except Exception:
        pass
    return value


def _coerce_text(value: Any) -> str:
    cleaned = _clean_scalar(value)
    if cleaned is None:
        return ""
    return str(cleaned).strip()


def coerce_usagi_row(
    raw_row: Mapping[str, Any],
    *,
    row_index: int,
    source_name: Any,
    source_code: Any,
    source_code_field: str | None = None,
) -> dict[str, Any]:
    """Ensure a mapping has the minimal Usagi fields using fallbacks.

    ``row_index`` is only used to manufacture synthetic identifiers when the
    source data does not provide a usable ``sourceCode`` or ``sourceName``.
    """

    payload: dict[str, Any] = {k: _clean_scalar(v) for k, v in raw_row.items()}
    name_value = source_name if _coerce_text(source_name) else payload.get("sourceName")
    if not _coerce_text(name_value):
        name_value = f"Query {row_index + 1}"
    payload["sourceName"] = _coerce_text(name_value)

    code_candidates = [source_code]
    if source_code_field:
        code_candidates.append(payload.get(source_code_field))
    code_candidates.append(payload.get("sourceCode"))

    code_value = next((val for val in code_candidates if _coerce_text(val)), None)
    if not _coerce_text(code_value):
        code_value = f"ROW_{row_index + 1:05d}"
    payload["sourceCode"] = _coerce_text(code_value)

    payload.setdefault("mappingStatus", "UNCHECKED")
    payload.setdefault("matchScore", 0.0)
    payload.setdefault("equivalence", "EQUIVALENT")
    payload.setdefault("mappingType", "MAPS_TO")
    payload.setdefault("sourceAutoAssignedConceptIds", "")

    return UsagiRow.model_validate(payload).model_dump()
