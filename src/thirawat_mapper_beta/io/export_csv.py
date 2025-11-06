"""CSV export helpers for bulk inference outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Optional

import pandas as pd


def export_relabel_csv(
    rows: Iterable[Mapping[str, object]],
    out_dir: Path | str,
    *,
    topk: int = 100,
    metrics: Optional[Mapping[str, float]] = None,
    metrics_filename: str = "metrics.json",
    results_filename: str = "results.csv",
    appended_filename: str = "results_with_input.csv",
    usagi_filename: str = "results_usagi.csv",
) -> tuple[Path, Path, Optional[Path]]:
    """Write inference results in two wide CSV formats.

    The primary CSV mirrors the classic relabel layout (source fields + topK
    columns). The secondary CSV appends candidate columns to the original input
    row so reviewers can keep the full source spreadsheet structure.
    """

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    relabel_records: list[MutableMapping[str, object]] = []
    appended_records: list[MutableMapping[str, object]] = []
    usagi_records: list[MutableMapping[str, object]] = []

    for row in rows:
        record: MutableMapping[str, object] = {}
        record["sourceName"] = row.get("source_name")
        if "source_code" in row:
            record["sourceCode"] = row.get("source_code")
        if "gold_concept_id" in row:
            record["gold_concept_id"] = row.get("gold_concept_id")

        candidates = row.get("candidates")
        if candidates is None:
            candidates = pd.DataFrame()

        candidate_columns: MutableMapping[str, object] = {}
        has_concept_name = (
            "concept_name" in candidates.columns if not candidates.empty else False
        )
        has_domain_id = "domain_id" in candidates.columns if not candidates.empty else False

        for idx in range(1, topk + 1):
            prefix = f"top{idx}"
            if idx <= len(candidates):
                candidate_row = candidates.iloc[idx - 1]
                candidate_columns[f"{prefix}_concept_id"] = candidate_row.get("concept_id")
                if has_concept_name:
                    candidate_columns[f"{prefix}_concept_name"] = candidate_row.get("concept_name")
                candidate_columns[f"{prefix}_score"] = candidate_row.get("final_score")
                if has_domain_id:
                    candidate_columns[f"{prefix}_domain_id"] = candidate_row.get("domain_id")
            else:
                candidate_columns[f"{prefix}_concept_id"] = None
                if has_concept_name:
                    candidate_columns[f"{prefix}_concept_name"] = None
                candidate_columns[f"{prefix}_score"] = None
                if has_domain_id:
                    candidate_columns[f"{prefix}_domain_id"] = None

        record.update(candidate_columns)
        relabel_records.append(record)

        input_row = row.get("input_row")
        if isinstance(input_row, Mapping):
            appended_record: MutableMapping[str, object] = dict(input_row)
        else:
            appended_record = {}
        appended_record.update(candidate_columns)
        appended_records.append(appended_record)

        if isinstance(input_row, Mapping) and {"sourceName", "mappingStatus", "matchScore"}.issubset(input_row.keys()):
            if candidate_columns.get("top1_concept_id") is not None:
                usagi_record: MutableMapping[str, object] = dict(input_row)
                usagi_record["matchScore"] = candidate_columns.get("top1_score")
                usagi_record["mappingStatus"] = "UNCHECKED"
                usagi_record["statusSetBy"] = "THIRAWAT-mapper"
                usagi_record["conceptId"] = candidate_columns.get("top1_concept_id")
                usagi_record["conceptName"] = candidate_columns.get("top1_concept_name")
                domain_candidate = candidate_columns.get("top1_domain_id")
                if domain_candidate is None and isinstance(input_row, Mapping):
                    domain_candidate = input_row.get("domainId")
                usagi_record["domainId"] = domain_candidate
                usagi_record["mappingType"] = "MAPS_TO"
                usagi_records.append(usagi_record)

    df_relabel = pd.DataFrame(relabel_records)
    csv_path = out_path / results_filename
    df_relabel.to_csv(csv_path, index=False)

    df_appended = pd.DataFrame(appended_records)
    appended_path = out_path / appended_filename
    df_appended.to_csv(appended_path, index=False)

    if metrics is not None:
        metrics_path = out_path / metrics_filename
        with metrics_path.open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, sort_keys=True)

    usagi_path: Optional[Path] = None
    if usagi_records:
        df_usagi = pd.DataFrame(usagi_records)
        usagi_path = out_path / usagi_filename
        df_usagi.to_csv(usagi_path, index=False)

    return csv_path, appended_path, usagi_path
