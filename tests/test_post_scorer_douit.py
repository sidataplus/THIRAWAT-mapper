import math

from thirawat_mapper.scoring import post_scorer as ps


def test_unitless_combo_alignment_ok():
    q = "Linagliptin+Metformin 2.5/500"
    d = "Metformin 500 mg / Linagliptin 2.5 mg oral tablet"
    s = ps.strength_sim(q, d)
    assert s > 0.95


def test_brand_neutral_remainder():
    q = "acetaminophen 500 mg tablet"
    d1 = "acetaminophen 500 mg [MYBRAND] oral tablet"
    d2 = "acetaminophen 500 mg oral tablet"
    j1 = ps.jaccard_remainder(q, d1)
    j2 = ps.jaccard_remainder(q, d2)
    # Brand token in brackets should not hurt remainder similarity
    assert abs(j1 - j2) < 1e-6


def test_extra_active_penalty_applies():
    q = "metformin 500 mg"
    clean = "metformin 500 mg tablet"
    extra = "metformin 500 mg + vitamin c 100 mg tablet"
    s_clean = ps.strength_sim(q, clean)
    s_extra = ps.strength_sim(q, extra)
    # Extra unmatched active should reduce strength similarity due to extra penalty
    assert s_clean >= s_extra
