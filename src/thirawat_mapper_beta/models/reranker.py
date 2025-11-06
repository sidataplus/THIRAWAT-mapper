"""LanceDB-compatible wrapper for the THIRAWAT reranker (beta).

Implements an internal BMS-style token interaction scorer and adapts it to
the LanceDB reranker protocol. Vector-only in this beta.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pyarrow as pa
import torch

from .bms_pooling import bms_scores

DEFAULT_RERANKER_ID = "na399/THIRAWAT-reranker-beta"


class _ThirawatPylateScorer:
    def __init__(self, model_id: str, device: Optional[str], max_query_len: int = 128, max_doc_len: int = 128):
        from pylate.models import ColBERT  # type: ignore
        self.model = ColBERT(
            model_name_or_path=model_id,
            device=device or None,
            query_length=int(max_query_len),
            document_length=int(max_doc_len),
        )
        try:
            if device:
                self.model.to(torch.device(device))
        except Exception:
            pass

    def encode_query(self, text: str):
        tok = self.model.tokenize([text], is_query=True, pad=True)
        try:
            device = next(self.model.parameters()).device
            tok = {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in tok.items()}
        except Exception:
            pass
        out = self.model(tok)
        return out["token_embeddings"], out.get("attention_mask")

    def encode_docs(self, texts: List[str]):
        tok = self.model.tokenize(texts, is_query=False, pad=False)
        try:
            device = next(self.model.parameters()).device
            tok = {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in tok.items()}
        except Exception:
            pass
        out = self.model(tok)
        return out["token_embeddings"], out.get("attention_mask")


class ThirawatReranker:
    def __init__(
        self,
        model_id: str = DEFAULT_RERANKER_ID,
        *,
        device: str | None = None,
        return_score: str = "all",
        column: str = "profile_text",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.return_score = return_score
        # LanceDB inspects `.score` to decide output columns
        self.score = return_score
        self.column = column

        # Lazy-load scorer to avoid import cost on module import
        self._scorer: Optional[_ThirawatPylateScorer] = None

    @property
    def scorer(self) -> _ThirawatPylateScorer:
        if self._scorer is None:
            self._scorer = _ThirawatPylateScorer(self.model_id, self.device)
        return self._scorer

    # LanceDB expects these methods when used as a reranker
    def rerank(self, query: Optional[str], results: pa.Table) -> pa.Table:
        return self.rerank_vector(query, results)

    def rerank_vector(self, query: Optional[str], vector_results: pa.Table) -> pa.Table:
        # Resolve text column
        col = self.column if self.column in vector_results.column_names else None
        if col is None and "profile_text_norm" in vector_results.column_names:
            col = "profile_text_norm"
        if col is None:
            raise ValueError(f"Candidate table missing '{self.column}' (or 'profile_text_norm') column.")

        texts: List[str] = vector_results[col].to_pylist()
        qtext = str(query or "").strip()
        if not qtext:
            raise ValueError("A text query is required for reranking.")

        q_emb, q_mask = self.scorer.encode_query(qtext)
        d_emb, d_mask = self.scorer.encode_docs(texts)
        with torch.no_grad():
            scores = bms_scores(q_emb, d_emb, q_mask, d_mask)
        scores = scores.detach().float().cpu().numpy().reshape(-1)
        df = vector_results.to_pandas()
        df["_relevance_score"] = scores.astype(np.float32)
        df = df.sort_values("_relevance_score", ascending=False, kind="mergesort").reset_index(drop=True)
        return pa.Table.from_pandas(df, preserve_index=False)

    def rerank_fts(self, query: Optional[str], fts_results: pa.Table) -> pa.Table:
        # Vector-only beta: fall back to vector rerank path
        return self.rerank_vector(query, fts_results)

    def rerank_hybrid(self, query: Optional[str], vector_results: pa.Table, fts_results: pa.Table) -> pa.Table:
        # Vector-only beta: ignore FTS and rerank vector results only
        return self.rerank_vector(query, vector_results)
