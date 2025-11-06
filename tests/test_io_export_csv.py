import json
from pathlib import Path

import pandas as pd

from thirawat_mapper_beta.io.export_csv import export_relabel_csv


def test_export_relabel_csv_writes_expected_layout(tmp_path: Path):
    rows = [
        {
            "source_name": "Query A",
            "source_code": "A1",
            "gold_concept_id": 11,
            "candidates": pd.DataFrame(
                {
                    "concept_id": [11, 22],
                    "concept_name": ["Concept A", "Concept B"],
                    "final_score": [0.9, 0.5],
                    "domain_id": ["Drug", "Drug"],
                }
            ),
            "input_row": {
                "sourceName": "Query A",
                "mappingStatus": "APPROVED",
                "matchScore": 0.0,
                "conceptId": 11,
            },
        },
        {
            "source_name": "Query B",
            "source_code": "B1",
            "gold_concept_id": 33,
            "candidates": pd.DataFrame(
                {
                    "concept_id": [44],
                    "concept_name": ["Concept C"],
                    "final_score": [0.8],
                }
            ),
            "input_row": {
                "sourceName": "Query B",
                "mappingStatus": "PENDING",
                "matchScore": 0.0,
                "conceptId": 33,
            },
        },
    ]

    csv_path, appended_path, usagi_path = export_relabel_csv(
        rows,
        out_dir=tmp_path,
        topk=3,
        metrics={"hit@1": 0.5},
    )

    df_relabel = pd.read_csv(csv_path)
    # topk=3 but predictions lengths => target_rows=2
    assert df_relabel.shape == (2, 1 + len(rows) * 3)
    assert df_relabel.columns[0] == "rank"

    # Map blocks by source code column name (last column in each block)
    block_map = {}
    for i in range(len(rows)):
        block_cols = df_relabel.columns[1 + i * 3 : 1 + (i + 1) * 3]
        assert len(block_cols) == 3
        block_map[block_cols[-1]] = block_cols

    # Query A block -> top1 match flag should be "X" and concept id "11"
    flag_col, text_col, code_col = block_map["A1"]
    assert df_relabel.at[0, flag_col] == "X"
    assert str(df_relabel.at[0, code_col]) == "11"
    assert df_relabel.at[1, text_col] == "Concept B"

    # Query B block -> first row has candidate 44, second row empty
    flag_col_b, text_col_b, code_col_b = block_map["B1"]
    assert int(float(df_relabel.at[0, code_col_b])) == 44
    assert pd.isna(df_relabel.at[1, code_col_b])

    # Appended CSV should contain original input columns plus candidate columns
    df_appended = pd.read_csv(appended_path)
    assert "sourceName" in df_appended.columns
    assert "top1_concept_id" in df_appended.columns
    assert df_appended.loc[df_appended["sourceName"] == "Query B", "top1_concept_id"].iat[0] == 44

    # Usagi CSV should not be None because first row looked like Usagi input
    assert usagi_path is not None and usagi_path.exists()
    df_usagi = pd.read_csv(usagi_path)
    assert set(["conceptId", "matchScore"]).issubset(df_usagi.columns)


def test_export_relabel_csv_without_usagi(tmp_path: Path):
    rows = [
        {
            "source_name": "Query C",
            "source_code": "C1",
            "gold_concept_id": 55,
            "candidates": pd.DataFrame({
                "concept_id": [55, 66],
                "concept_name": ["Concept X", "Concept Y"],
                "final_score": [0.75, 0.5],
            }),
            # input_row lacks mappingStatus/matchScore -> should not produce Usagi export
            "input_row": {"sourceName": "Query C"},
        }
    ]

    csv_path, appended_path, usagi_path = export_relabel_csv(
        rows,
        out_dir=tmp_path,
        topk=2,
    )

    assert csv_path.exists()
    assert appended_path.exists()
    assert usagi_path is None

    metrics_path = tmp_path / "metrics.json"
    assert not metrics_path.exists()
