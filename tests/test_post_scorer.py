import math

from thirawat_mapper_beta.scoring import post_scorer as ps


def test_extract_strengths_parses_ratio_and_percent():
    text = "Insulin 100 units/mL (10 mL)"
    comps, spans = ps.extract_strengths_with_spans(text)
    assert spans, "expected spans for parsed strengths"
    ratio = [c for c in comps if c.kind == "ratio"]
    assert ratio, "ratio component missing"
    assert math.isclose(ratio[0].normalized(), 100.0, rel_tol=1e-3)


def test_strength_sim_handles_unit_normalization():
    q = "albuterol 90 mcg per actuation"
    d = "90 mcg/actuation albuterol HFA"
    assert ps.strength_sim(q, d) > 0.95


def test_batch_features_minmax_option():
    q = "acetaminophen 500 mg tablet"
    docs = [
        "acetaminophen 500 mg oral tablet",
        "acetaminophen 325 mg tablet",
        "ibuprofen 200 mg tablet",
    ]
    features = ps.batch_features(q, docs, minmax_within_query=True)
    assert list(features.keys()) == ["strength_sim", "jaccard_text", "brand_score", "simple_score"]
    assert len(features["strength_sim"]) == len(docs)
    # Scores can dip negative due to brand penalties but should remain bounded
    assert all(-1.0 <= s <= 1.0 for s in features["simple_score"])


def test_brand_score_prefers_matching_brand():
    query = "insulin isophane inj 10 ml [Gensulin N]"
    match_doc = "insulin isophane inj 10 ml [gensulin n]"
    mismatch_doc = "insulin isophane inj 10 ml [humulin n]"
    no_brand = "insulin isophane inj 10 ml"

    match = ps.brand_score(query, match_doc)
    mismatch = ps.brand_score(query, mismatch_doc)
    missing = ps.brand_score(query, no_brand)

    assert match == 0.0
    assert missing == 0.0
    assert mismatch < 0
