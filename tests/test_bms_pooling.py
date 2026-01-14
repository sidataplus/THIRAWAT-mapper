import torch
import torch.nn.functional as F

from thirawat_mapper.models.bms_pooling import bms_scores


def _ref_bms_scores(q, d, qmask=None, dmask=None):
    q = F.normalize(q, p=2, dim=-1)
    d = F.normalize(d, p=2, dim=-1)
    sim = torch.einsum("ash,bth->abst", q, d)  # [A,B,S,T]
    neg_inf = torch.finfo(sim.dtype).min / 4
    if dmask is not None:
        sim = sim.masked_fill(dmask.bool().logical_not().unsqueeze(0).unsqueeze(2), neg_inf)
    if qmask is not None:
        sim = sim.masked_fill(qmask.bool().logical_not().unsqueeze(1).unsqueeze(3), neg_inf)
    # Guard fully masked
    if dmask is not None and dmask.numel():
        docs_all_masked = (~dmask.bool()).all(dim=1)
        if docs_all_masked.any():
            sim[:, docs_all_masked, :, :] = -1.0
    if qmask is not None and qmask.numel():
        q_all_masked = (~qmask.bool()).all(dim=1)
        if q_all_masked.any():
            sim[q_all_masked, :, :, :] = -1.0

    per_q = sim.max(dim=-1).values  # [A,B,S]
    if qmask is not None:
        qm = qmask.bool().unsqueeze(1)
        q2d = (per_q * qm).sum(dim=-1) / qm.sum(dim=-1).clamp(min=1).float()
    else:
        q2d = per_q.mean(dim=-1)

    per_d = sim.max(dim=-2).values  # [A,B,T]
    if dmask is not None:
        dm = dmask.bool().unsqueeze(0)
        d2q = (per_d * dm).sum(dim=-1) / dm.sum(dim=-1).clamp(min=1).float()
    else:
        d2q = per_d.mean(dim=-1)
    return 0.5 * (q2d + d2q)


def test_bms_no_masks_matches_reference():
    torch.manual_seed(0)
    A, S, B, T, H = 2, 5, 3, 7, 8
    q = torch.randn(A, S, H, dtype=torch.float32)
    d = torch.randn(B, T, H, dtype=torch.float32)
    out_ref = _ref_bms_scores(q, d)
    out_opt = bms_scores(q, d)
    assert out_ref.shape == out_opt.shape == (A, B)
    assert torch.allclose(out_ref, out_opt, atol=1e-5, rtol=1e-5)


def test_bms_with_masks_matches_reference():
    torch.manual_seed(1)
    A, S, B, T, H = 2, 6, 3, 5, 16
    q = torch.randn(A, S, H, dtype=torch.float32)
    d = torch.randn(B, T, H, dtype=torch.float32)
    # random masks with at least 1 true per row
    qmask = (torch.rand(A, S) > 0.2)
    dmask = (torch.rand(B, T) > 0.2)
    if qmask.sum(dim=1).min() == 0:
        qmask[:, 0] = True
    if dmask.sum(dim=1).min() == 0:
        dmask[:, 0] = True
    out_ref = _ref_bms_scores(q, d, qmask, dmask)
    out_opt = bms_scores(q, d, qmask, dmask)
    assert out_ref.shape == out_opt.shape == (A, B)
    assert torch.allclose(out_ref, out_opt, atol=1e-5, rtol=1e-5)


def test_bms_fully_masked_rows_are_handled_and_match():
    torch.manual_seed(2)
    A, S, B, T, H = 2, 4, 3, 5, 12
    q = torch.randn(A, S, H, dtype=torch.float32)
    d = torch.randn(B, T, H, dtype=torch.float32)
    # Fully mask query tokens for the first query, and doc tokens for the last doc
    qmask = torch.ones(A, S, dtype=torch.bool)
    dmask = torch.ones(B, T, dtype=torch.bool)
    qmask[0, :] = False
    dmask[-1, :] = False
    out_ref = _ref_bms_scores(q, d, qmask, dmask)
    out_opt = bms_scores(q, d, qmask, dmask)
    assert torch.isfinite(out_opt).all(), "optimized output contains non-finite values"
    assert torch.allclose(out_ref, out_opt, atol=1e-5, rtol=1e-5)


def test_bms_degenerate_lengths():
    torch.manual_seed(3)
    # Degenerate token lengths (S=1 or T=1)
    for S, T in [(1, 7), (6, 1), (1, 1)]:
        A, B, H = 2, 3, 16
        q = torch.randn(A, S, H, dtype=torch.float32)
        d = torch.randn(B, T, H, dtype=torch.float32)
        qmask = torch.ones(A, S, dtype=torch.bool)
        dmask = torch.ones(B, T, dtype=torch.bool)
        out_ref = _ref_bms_scores(q, d, qmask, dmask)
        out_opt = bms_scores(q, d, qmask, dmask)
        assert out_ref.shape == out_opt.shape == (A, B)
        assert torch.allclose(out_ref, out_opt, atol=1e-5, rtol=1e-5)
