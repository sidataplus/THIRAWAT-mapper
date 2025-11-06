"""Token‑interaction pooling utility for BMS‑style reranking."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bms_scores(
    queries_embeddings: torch.Tensor,
    documents_embeddings: torch.Tensor,
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
) -> torch.Tensor:

    q = F.normalize(queries_embeddings, p=2, dim=-1)
    d = F.normalize(documents_embeddings, p=2, dim=-1)
    q, d = q, d

    sim = torch.einsum("ash,bth->abst", q, d)
    neg_inf = torch.finfo(sim.dtype).min / 4

    if documents_mask is not None:
        dmask = documents_mask.bool()
        sim = sim.masked_fill(dmask.logical_not().unsqueeze(0).unsqueeze(2), neg_inf)
    if queries_mask is not None:
        qmask = queries_mask.bool()
        sim = sim.masked_fill(qmask.logical_not().unsqueeze(1).unsqueeze(3), neg_inf)

    floor = sim.new_tensor(-1.0)
    if documents_mask is not None and documents_mask.numel():
        docs_all_masked = (~documents_mask.bool()).all(dim=1)
        if docs_all_masked.any():
            sim[:, docs_all_masked, :, :] = floor
    if queries_mask is not None and queries_mask.numel():
        q_all_masked = (~queries_mask.bool()).all(dim=1)
        if q_all_masked.any():
            sim[q_all_masked, :, :, :] = floor

    per_q = sim.max(dim=-1).values  # [A,B,Tq]
    if queries_mask is not None:
        qmask = queries_mask.bool().unsqueeze(1)  # [A,1,Tq]
        denom_q = qmask.sum(dim=-1).clamp(min=1).float()
        q2d = (per_q * qmask).sum(dim=-1) / denom_q  # [A,B]
    else:
        q2d = per_q.mean(dim=-1)

    per_d = sim.max(dim=-2).values  # [A,B,Td]
    if documents_mask is not None:
        dmask = documents_mask.bool().unsqueeze(0)  # [1,B,Td]
        denom_d = dmask.sum(dim=-1).clamp(min=1).float()
        d2q = (per_d * dmask).sum(dim=-1) / denom_d  # [A,B]
    else:
        d2q = per_d.mean(dim=-1)

    return 0.5 * (q2d + d2q)


__all__ = ["bms_scores"]
