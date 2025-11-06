import pandas as pd

from thirawat_mapper_beta.infer.bulk import _prepare_queries, _compute_metrics


def test_prepare_queries_handles_missing_and_codes():
    df = pd.DataFrame(
        {
            "sourceName": ["Drug A", "", "Drug C"],
            "sourceCode": ["123", "456", None],
        }
    )
    queries = _prepare_queries(df, "sourceName", "sourceCode")
    assert queries == ["Drug A (123)", "", "Drug C"]


def test_compute_metrics_counts_hits_and_mrr():
    rows = [
        {  # hit@1 -> yes
            "gold_concept_id": 11,
            "candidates": pd.DataFrame({"concept_id": [11, 22]}),
        },
        {  # hit@5 -> yes, @1 no
            "gold_concept_id": 33,
            "candidates": pd.DataFrame({"concept_id": [44, 55, 33]}),
        },
        {  # gold missing => ignored in labeled counts
            "gold_concept_id": None,
            "candidates": pd.DataFrame({"concept_id": [99]}),
        },
        {  # empty candidates -> no coverage increment
            "gold_concept_id": 77,
            "candidates": pd.DataFrame({"concept_id": []}),
        },
    ]

    metrics = _compute_metrics(rows, topk=5)

    assert metrics["n_rows"] == pytest.approx(4.0)
    assert metrics["coverage"] == pytest.approx(0.75)  # 3 rows had candidates
    assert metrics["n_labeled"] == 3
    # hits: first row hit@1, second row hit within top3
    assert metrics["hit@1"] == pytest.approx(1 / 3)
    assert metrics["hit@5"] == pytest.approx(2 / 3)
    assert metrics["mrr@100"] == pytest.approx((1 + 1 / 3) / 3)


import pytest  # noqa: E402 (after function definitions for readability)
