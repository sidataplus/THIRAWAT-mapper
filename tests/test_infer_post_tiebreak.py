import pandas as pd

from thirawat_mapper.infer.utils import (
    enrich_with_post_scores,
    should_apply_post,
    tiebreak_rerank,
)


def test_tiebreak_rerank_reorders_within_eps():
    df = pd.DataFrame(
        {
            "_relevance_score": [1.0, 0.995, 0.98],
            "post_score": [0.1, 0.9, 0.2],
        }
    )
    out = tiebreak_rerank(
        df,
        primary_col="_relevance_score",
        tiebreak_cols=("post_score",),
        eps=0.01,
        topn=3,
    )
    assert out["post_score"].tolist() == [0.9, 0.1, 0.2]


def test_tiebreak_rerank_respects_gaps_and_topn():
    df = pd.DataFrame(
        {
            "_relevance_score": [1.0, 0.97, 0.969],
            "post_score": [0.1, 0.9, 0.8],
        }
    )
    out = tiebreak_rerank(
        df,
        primary_col="_relevance_score",
        tiebreak_cols=("post_score",),
        eps=0.01,
        topn=1,
    )
    assert out["_relevance_score"].tolist() == [1.0, 0.97, 0.969]


def test_tiebreak_prefers_rerank_top20():
    df = pd.DataFrame(
        {
            "_relevance_score": [1.0] * 25,
            "brand_strength_exact": [0] * 25,
            "top20_strength_form_exact": [0] * 25,
            "brand_score": [0.0] * 25,
            "rerank_top20": [1] * 20 + [0] * 5,
            "strength_exact": [0] * 25,
            "strength_sim": [0.0] * 25,
            "form_route_score": [0.0] * 25,
            "release_score": [0.0] * 25,
        }
    )
    out = tiebreak_rerank(
        df,
        primary_col="_relevance_score",
        tiebreak_cols=(
            "brand_strength_exact",
            "top20_strength_form_exact",
            "brand_score",
            "rerank_top20",
            "strength_exact",
            "strength_sim",
            "form_route_score",
            "release_score",
        ),
        eps=0.01,
        topn=25,
    )
    assert out["rerank_top20"].tolist()[:20] == [1] * 20


def test_tiebreak_prefers_brand_strength_exact_over_top20_form():
    df = pd.DataFrame(
        {
            "_relevance_score": [1.0, 1.0],
            "brand_strength_exact": [1, 0],
            "top20_strength_form_exact": [0, 1],
            "brand_score": [1.0, 1.0],
            "rerank_top20": [1, 1],
            "strength_exact": [1, 1],
            "strength_sim": [1.0, 1.0],
            "form_route_score": [1.0, 1.0],
            "release_score": [0.0, 0.0],
        }
    )
    out = tiebreak_rerank(
        df,
        primary_col="_relevance_score",
        tiebreak_cols=(
            "brand_strength_exact",
            "top20_strength_form_exact",
            "brand_score",
            "rerank_top20",
            "strength_exact",
            "strength_sim",
            "form_route_score",
            "release_score",
        ),
        eps=0.01,
        topn=2,
    )
    assert out["brand_strength_exact"].tolist()[0] == 1


def test_brand_strict_filters_mismatches_with_fallback():
    df = pd.DataFrame(
        {
            "concept_name": ["foo [Acme]", "foo [Other]", "foo"],
            "profile_text": ["", "", ""],
            "_relevance_score": [1.0, 0.99, 0.98],
        }
    )
    out = enrich_with_post_scores(
        df,
        "foo [Acme]",
        post_strength_weight=0.6,
        post_jaccard_weight=0.4,
        post_brand_penalty=0.3,
        post_minmax=False,
        post_weight=0.0,
        prefer_brand=True,
        post_mode="tiebreak",
        tiebreak_eps=0.01,
        tiebreak_topn=50,
        brand_strict=True,
    )
    assert out["concept_name"].tolist() == ["foo [Acme]", "foo"]

    df_all_bad = pd.DataFrame(
        {
            "concept_name": ["foo [Other]"],
            "profile_text": [""],
            "_relevance_score": [1.0],
        }
    )
    out_all_bad = enrich_with_post_scores(
        df_all_bad,
        "foo [Acme]",
        post_strength_weight=0.6,
        post_jaccard_weight=0.4,
        post_brand_penalty=0.3,
        post_minmax=False,
        post_weight=0.0,
        prefer_brand=True,
        post_mode="tiebreak",
        tiebreak_eps=0.01,
        tiebreak_topn=50,
        brand_strict=True,
    )
    assert out_all_bad["concept_name"].tolist() == ["foo [Other]"]


def test_exact_strength_unpenalizes_form_mismatch():
    df = pd.DataFrame(
        {
            "concept_name": [
                "drug 24 mg/mL oral solution",
                "drug 24 mg/mL injection",
            ],
            "profile_text": ["", ""],
            "_relevance_score": [1.0, 1.0],
        }
    )
    out = enrich_with_post_scores(
        df,
        "drug 24 mg/mL oral solution",
        post_strength_weight=0.6,
        post_jaccard_weight=0.4,
        post_brand_penalty=0.3,
        post_minmax=False,
        post_weight=0.0,
        prefer_brand=True,
        post_mode="tiebreak",
        tiebreak_eps=0.01,
        tiebreak_topn=50,
        brand_strict=False,
    )
    inj = out[out["concept_name"].str.contains("injection")].iloc[0]
    assert inj["form_route_score"] == -0.5
    assert inj["strength_exact"] == 1


def test_should_apply_post_respects_mode():
    assert should_apply_post("tiebreak", 0.0) is True
    assert should_apply_post("lex", 0.0) is True
    assert should_apply_post("blend", 0.0) is False
    assert should_apply_post("blend", 0.1) is True
