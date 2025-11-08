from thirawat_mapper_beta.scoring import post_scorer as ps


def test_dose_gate_zero_simple_score_on_large_mismatch():
    # Big dose mismatch should trip the S_dose < tau gate and yield 0 simple_score
    q = "metformin 500 mg tablet"
    d = "metformin 5 mg tablet"
    out = ps.simple_strength_plus_jaccard(q, d)
    assert out["post_score"] == 0.0
