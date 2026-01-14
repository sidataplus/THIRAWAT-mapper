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
    preserve_input_order: bool = False,
) -> tuple[Path, Path, Optional[Path]]:
    """Write inference results in two wide CSV formats.

    The primary CSV mirrors the classic relabel layout (source fields + topK
    columns). The secondary CSV appends candidate columns to the original input
    row so reviewers can keep the full source spreadsheet structure.
    """

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Classic relabel layout (wide, block-per-query) following the reference format
    # Build eval_df, predictions mapping, and profiles lookup from the provided rows
    eval_rows: list[MutableMapping[str, object]] = []
    predictions: dict[str, list[int]] = {}
    profiles_accum: dict[int, MutableMapping[str, object]] = {}

    relabel_records: list[MutableMapping[str, object]] = []  # kept for appended/usagi composition below
    appended_records: list[MutableMapping[str, object]] = []
    usagi_records: list[MutableMapping[str, object]] = []

    for _idx, row in enumerate(rows):
        record: MutableMapping[str, object] = {}
        record["sourceName"] = row.get("source_name")
        if "source_code" in row:
            record["sourceCode"] = row.get("source_code")
        if "gold_concept_id" in row:
            record["gold_concept_id"] = row.get("gold_concept_id")

        candidates = row.get("candidates")
        if candidates is None:
            candidates = pd.DataFrame()

        # Accumulate predictions and candidate metadata for the relabel layout
        key = str(row.get("source_code") or f"row{_idx}")
        cand_ids: list[int] = []
        if not candidates.empty and "concept_id" in candidates.columns:
            # keep up to topk ids
            cand_ids = [int(v) for v in candidates["concept_id"].head(topk).tolist() if pd.notna(v)]
            # accumulate concept_name / concept_code if available
            has_name = "concept_name" in candidates.columns
            code_col = "concept_code" if "concept_code" in candidates.columns else None
            for _, crow in candidates.head(topk).iterrows():
                try:
                    cid = int(crow.get("concept_id"))
                except Exception:
                    continue
                if cid not in profiles_accum:
                    profiles_accum[cid] = {
                        "concept_id": cid,
                        "concept_name": str(crow.get("concept_name") or "") if has_name else "",
                        "concept_code": str(crow.get(code_col) or "") if code_col else "",
                    }
        predictions[key] = cand_ids

        # Build eval_df row
        gold = row.get("gold_concept_id")
        try:
            gold_id = int(gold) if gold is not None and str(gold).strip() else -1
        except Exception:
            gold_id = -1
        eval_rows.append(
            {
                "query_key": key,
                "gold_id": gold_id,
                "query_text": str(row.get("source_name") or ""),
            }
        )

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

        record.update(candidate_columns)  # keep for appended + usagi
        relabel_records.append(record)

        input_row = row.get("input_row")
        if isinstance(input_row, Mapping):
            appended_record: MutableMapping[str, object] = dict(input_row)
        else:
            appended_record = {}
        appended_record.update(candidate_columns)
        appended_records.append(appended_record)

        usagi_payload: Optional[MutableMapping[str, object]] = None
        if isinstance(row.get("usagi_row"), Mapping):
            usagi_payload = dict(row["usagi_row"])  # type: ignore[index]
        elif isinstance(input_row, Mapping) and {"sourceName", "mappingStatus", "matchScore"}.issubset(input_row.keys()):
            usagi_payload = dict(input_row)

        if usagi_payload is not None:
            top1_id = candidate_columns.get("top1_concept_id")
            if top1_id is not None:
                usagi_payload["conceptId"] = top1_id
                usagi_payload["conceptName"] = candidate_columns.get("top1_concept_name")
                usagi_payload["matchScore"] = candidate_columns.get("top1_score")
                domain_candidate = candidate_columns.get("top1_domain_id")
                if domain_candidate is None:
                    domain_candidate = usagi_payload.get("domainId")
                usagi_payload["domainId"] = domain_candidate
            usagi_payload["mappingStatus"] = "UNCHECKED"
            usagi_payload["statusSetBy"] = "THIRAWAT-mapper"
            usagi_payload["mappingType"] = usagi_payload.get("mappingType") or "MAPS_TO"
            usagi_records.append(usagi_payload)

    # Construct classic relabel CSV frame (wide blocks) per reference format
    classic_csv = out_path / results_filename
    try:
        eval_df = pd.DataFrame(eval_rows)
        # profiles table with required columns; ensure concept_code exists
        if profiles_accum:
            profiles_df = pd.DataFrame(profiles_accum.values())
        else:
            profiles_df = pd.DataFrame(columns=["concept_id", "concept_name", "concept_code"])  # empty

        # Determine target rows (per query) and build blocks
        max_len = max((len(v) for v in predictions.values()), default=0)
        if max_len == 0:
            # Write an empty file to keep previous behavior
            pd.DataFrame().to_csv(classic_csv, index=False)
        else:
            target_rows = min(topk if topk and topk > 0 else max_len, max_len)
            target_rows = max(target_rows, 1)

            # Build lookup dict
            if not profiles_df.empty:
                lookup = profiles_df.set_index("concept_id")[
                    [c for c in ["concept_name", "concept_code"] if c in profiles_df.columns]
                ].to_dict("index")
            else:
                lookup = {}

            blocks = []
            for ed in eval_df.itertuples(index=False):
                key = str(getattr(ed, "query_key"))
                gold_id = int(getattr(ed, "gold_id")) if pd.notna(getattr(ed, "gold_id")) else -1
                query_text = str(getattr(ed, "query_text"))
                source_code = key
                cand_list = list(predictions.get(key, []))[:target_rows]
                gold_rank = next((i + 1 for i, cid in enumerate(cand_list) if cid == gold_id and gold_id >= 0), None)
                rank_header = str(gold_rank) if gold_rank is not None else "unmatched"

                rows_block = []
                for i in range(target_rows):
                    if i < len(cand_list):
                        cid = int(cand_list[i])
                        info = lookup.get(cid, {}) or {}
                        cname = info.get("concept_name") or ""
                        match_flag = "X" if gold_rank is not None and cid == gold_id else ""
                        rows_block.append([match_flag, cname, str(cid)])
                    else:
                        rows_block.append(["", "", ""])
                block_df = pd.DataFrame(rows_block, columns=[rank_header, query_text, source_code])
                blocks.append({
                    "gold_rank": gold_rank,
                    "source_code": source_code,
                    "frame": block_df,
                    "order": len(blocks),
                })

            if preserve_input_order:
                ordered_blocks = blocks
            else:
                def sort_key(item: dict) -> tuple[int, float, str]:
                    gr = item["gold_rank"]
                    if gr is None:
                        return (1, float("inf"), item["source_code"])  # unmatched last
                    return (0, float(gr), item["source_code"])  # matched first, lower rank first

                ordered_blocks = sorted(blocks, key=sort_key)

            frames = [b["frame"] for b in ordered_blocks] if ordered_blocks else []
            if frames:
                wide = pd.concat(frames, axis=1)
                # Add rank column on the left
                rank_col = pd.Series([str(i + 1) for i in range(target_rows)], name="rank")
                wide = pd.concat([rank_col, wide], axis=1)
                wide.to_csv(classic_csv, index=False)
            else:
                pd.DataFrame().to_csv(classic_csv, index=False)
    except Exception:
        # Fallback: write the previous simple table if something goes wrong
        pd.DataFrame(relabel_records).to_csv(classic_csv, index=False)

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

    # Report the classic CSV path as the primary result
    csv_path = classic_csv
    return csv_path, appended_path, usagi_path
