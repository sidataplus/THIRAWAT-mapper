"""Token‑interaction pooling utilities for BMS‑style reranking.

Implements symmetric, length‑neutral pooling over token interaction matrices
with optional temperature and n‑gram span augmentation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bms_scores(
    queries_embeddings: torch.Tensor,
    documents_embeddings: torch.Tensor,
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
    *,
    pooling: str = "symmetric_max",
    temperature: float = 20.0,
    topk: int | None = None,
    bidir_alpha: float = 0.5,
    len_b: float = 0.5,
    symmetric_merge: str = "avg",
    ngrams: list[int] | tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Compute pooled scores with symmetric options; returns [A, B]."""

    q = F.normalize(queries_embeddings, p=2, dim=-1)
    d = F.normalize(documents_embeddings, p=2, dim=-1)

    def _augment_with_ngrams(
        emb: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not ngrams:
            return emb, mask
        toks = emb
        m = mask
        ns = sorted({int(n) for n in ngrams if int(n) >= 2})
        if len(ns) == 0:
            return emb, mask
        parts = [toks]
        span_masks: list[torch.Tensor] = []
        x = toks.permute(0, 2, 1)
        for n in ns:
            if toks.size(1) - n + 1 <= 0:
                continue
            span = F.avg_pool1d(x, kernel_size=n, stride=1).permute(0, 2, 1)
            span = F.normalize(span, p=2, dim=-1)
            parts.append(span)
            if m is not None:
                mf = m.float().unsqueeze(1)
                mspan = F.avg_pool1d(mf, kernel_size=n, stride=1)
                mvalid = mspan.squeeze(1).eq(1.0)
                span_masks.append(mvalid)
        out = torch.cat(parts, dim=1)
        if m is None:
            return out, None
        out_mask = torch.cat([m] + span_masks, dim=1) if span_masks else m
        return out, out_mask

    q, queries_mask = _augment_with_ngrams(q, queries_mask)
    d, documents_mask = _augment_with_ngrams(d, documents_mask)

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

    def _q2d(kind: str) -> torch.Tensor:
        if kind.endswith("max"):
            per_q = sim.max(dim=-1).values
        elif kind.endswith("lnmax"):
            per_q = sim.max(dim=-1).values
            if documents_mask is not None:
                dlen = documents_mask.sum(dim=1).clamp(min=1).float()
            else:
                dlen = torch.full((sim.size(1),), float(sim.size(-1)), device=sim.device)
            per_q = per_q / dlen.view(1, -1, 1)
        elif kind.endswith("softmax"):
            per_q = (sim * float(temperature)).logsumexp(dim=-1) / float(temperature)
        elif kind.endswith("topk"):
            k = int(topk or max(1, int(sim.size(-1) ** 0.5)))
            per_q = sim.topk(k=k, dim=-1).values.mean(dim=-1)
        else:
            raise ValueError(f"Unknown pooling kind: {kind}")
        if queries_mask is not None:
            qmask = queries_mask.bool().unsqueeze(1)
            denom = qmask.sum(dim=-1).clamp(min=1).float()
            return (per_q * qmask).sum(dim=-1) / denom
        return per_q.mean(dim=-1)

    prim = str(pooling).strip().lower()
    if prim.startswith("symmetric_"):
        base = prim.split("symmetric_")[-1]
        qd = _q2d(base)
        dq = _q2d(base)
        merge = str(symmetric_merge).lower()
        alpha = float(bidir_alpha)
        if merge in {"hmean", "harmonic"}:
            eps = 1e-6
            qd_pos = torch.clamp((qd + 1.0) * 0.5, min=eps, max=1.0)
            dq_pos = torch.clamp((dq + 1.0) * 0.5, min=eps, max=1.0)
            hm = 1.0 / ((1.0 - alpha) / qd_pos + alpha / dq_pos)
            return hm.mul(2.0).sub(1.0)
        elif merge in {"gmean", "geometric"}:
            eps = 1e-6
            qd_pos = torch.clamp((qd + 1.0) * 0.5, min=eps, max=1.0)
            dq_pos = torch.clamp((dq + 1.0) * 0.5, min=eps, max=1.0)
            gm = torch.exp((1.0 - alpha) * torch.log(qd_pos) + alpha * torch.log(dq_pos))
            return gm.mul(2.0).sub(1.0)
        else:
            return (1.0 - alpha) * qd + alpha * dq
    else:
        return _q2d(prim)


__all__ = ["bms_scores"]

