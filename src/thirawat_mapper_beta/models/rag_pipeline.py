"""Coordination utilities for LLM-based reranking."""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .rag_llm import BaseLLMClient
from .rag_prompt import RAGPromptBuilder, RagCandidate

LOGGER = logging.getLogger(__name__)


@dataclass
class RagRanking:
    """Structured output from the RAG pipeline."""

    concept_ids: List[int]
    scores: List[float]
    prompt: str
    response: str


def _normalize(text: str) -> str:
    lowered = text.strip().lower()
    lowered = re.sub(r"\s+", " ", lowered)
    lowered = lowered.replace("µg", "mcg")
    lowered = lowered.replace("’", "'").replace("–", "-")
    lowered = lowered.strip(",.;")
    return lowered


def _extract_json_payload(text: str) -> Optional[Iterable]:
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            answer = parsed.get("answer")
            if isinstance(answer, (list, tuple)):
                return answer
            if isinstance(answer, (str, int, float)):
                return [answer]
            result = parsed.get("result")
            if isinstance(result, (list, tuple)):
                return result
            if isinstance(result, dict):
                inner = result.get("answer") or result.get("output")
                if isinstance(inner, (list, tuple)):
                    return inner
                if isinstance(inner, (str, int, float)):
                    return [inner]
        return None
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, list):
            return parsed
    return None


def _tokenize_freeform(text: str) -> List[str]:
    tokens: List[str] = []
    # Replace common separators with newline for easier splitting
    normalized = re.sub(r"[;,]", "\n", text)
    for raw in normalized.splitlines():
        piece = raw.strip()
        if not piece:
            continue
        if piece.lower().startswith("option "):
            # Allow formats like "Option 1: <text>"
            _, _, tail = piece.partition(":")
            piece = tail.strip() or piece
        if piece and piece[0].isdigit() and ". " in piece[:6]:
            piece = piece.split(". ", 1)[1].strip()
        if piece:
            tokens.append(piece)
    if not tokens:
        # Fallback to comma split if previous logic produced nothing
        for raw in text.split(","):
            piece = raw.strip()
            if piece:
                tokens.append(piece)
    return tokens


def _parse_candidates_from_sequence(
    payload: Iterable,
    option_strings: Sequence[str],
    concept_ids: Sequence[int],
) -> Tuple[List[int], Dict[int, float]]:
    option_norm = {_normalize(opt): cid for opt, cid in zip(option_strings, concept_ids)}
    score_map: Dict[int, float] = {}
    ordered: List[int] = []
    seen: set[int] = set()

    def _push(cid: int, score: Optional[float]) -> None:
        if cid in seen:
            return
        seen.add(cid)
        ordered.append(cid)
        if score is not None:
            score_map[cid] = score

    for item in payload:
        candidate_id: Optional[int] = None
        score_val: Optional[float] = None
        if isinstance(item, dict):
            value = item.get("concept_id") or item.get("id") or item.get("conceptId")
            if value is not None:
                try:
                    candidate_id = int(value)
                except (TypeError, ValueError):
                    candidate_id = None
            score = item.get("score") or item.get("confidence") or item.get("weight")
            if score is not None:
                try:
                    score_val = float(score)
                except (TypeError, ValueError):
                    score_val = None
            if candidate_id is None and "option" in item:
                option_text = str(item["option"])
                candidate_id = option_norm.get(_normalize(option_text))
        elif isinstance(item, (int, float)):
            candidate_id = int(item)
        elif isinstance(item, str):
            token_norm = _normalize(item)
            if token_norm in option_norm:
                candidate_id = option_norm[token_norm]
            else:
                if token_norm.isdigit():
                    idx = int(token_norm) - 1
                    if 0 <= idx < len(concept_ids):
                        candidate_id = concept_ids[idx]
                    else:
                        numeric_id = int(token_norm)
                        if numeric_id in concept_ids:
                            candidate_id = numeric_id
                else:
                    match = re.search(r"\((\d+)\)$", item)
                    if match:
                        numeric = int(match.group(1))
                        if numeric in concept_ids:
                            candidate_id = numeric
        if candidate_id is not None and candidate_id in concept_ids:
            _push(candidate_id, score_val)
    return ordered, score_map


def _order_from_response(
    response: str,
    option_strings: Sequence[str],
    concept_ids: Sequence[int],
) -> Tuple[List[int], Dict[int, float]]:
    payload = _extract_json_payload(response)
    if payload is not None:
        ordered, scores = _parse_candidates_from_sequence(payload, option_strings, concept_ids)
        if ordered:
            return ordered, scores

    # Try parsing free-form output
    tokens = _tokenize_freeform(response)
    ordered_ids: List[int] = []
    scores: Dict[int, float] = {}
    option_norm = {_normalize(opt): cid for opt, cid in zip(option_strings, concept_ids)}
    seen: set[int] = set()
    for token in tokens:
        cid: Optional[int] = None
        norm = _normalize(token)
        if norm in option_norm:
            cid = option_norm[norm]
        else:
            match = re.search(r"\((\d+)\)$", token)
            if match:
                num = int(match.group(1))
                if num in concept_ids:
                    cid = num
        if cid is None and norm and norm.isdigit():
            rank_idx = int(norm) - 1
            if 0 <= rank_idx < len(concept_ids):
                cid = concept_ids[rank_idx]
            else:
                num = int(norm)
                if num in concept_ids:
                    cid = num
        if cid is None:
            continue
        if cid in seen:
            continue
        seen.add(cid)
        ordered_ids.append(cid)
    return ordered_ids, scores


class RAGPipeline:
    """End-to-end helper that builds prompts, calls the LLM, and parses results."""

    def __init__(
        self,
        llm: BaseLLMClient,
        *,
        prompt_builder: Optional[RAGPromptBuilder] = None,
        fallback_on_error: bool = True,
    ) -> None:
        self.llm = llm
        self.prompt_builder = prompt_builder or RAGPromptBuilder()
        self.fallback_on_error = bool(fallback_on_error)

    def rerank(
        self,
        query_text: str,
        candidates: Sequence[RagCandidate],
        *,
        extra_context: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop_sequences: Optional[Sequence[str]] = None,
    ) -> RagRanking:
        if not candidates:
            raise ValueError("RAGPipeline requires at least one candidate.")
        prompt, option_strings = self.prompt_builder.build(
            query_text,
            candidates,
            extra_context=extra_context,
        )
        concept_ids = [cand.concept_id for cand in candidates]
        try:
            response = self.llm.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop_sequences,
            )
        except Exception as exc:
            LOGGER.warning("LLM generation failed (%s); falling back to baseline order.", exc)
            if not self.fallback_on_error:
                raise
            response = ""

        ordered, score_map = _order_from_response(response, option_strings, concept_ids)
        if not ordered:
            LOGGER.debug("No candidates parsed from LLM response; using baseline order.")
            ordered = concept_ids
        else:
            # Append remaining candidates preserving baseline order
            remaining = [cid for cid in concept_ids if cid not in ordered]
            ordered.extend(remaining)

        scores: List[float] = []
        total = len(ordered)
        for idx, cid in enumerate(ordered):
            if cid in score_map:
                scores.append(score_map[cid])
            else:
                # Backfill with a simple monotonic curve
                scores.append(float(total - idx) / float(max(total, 1)))
        return RagRanking(concept_ids=ordered, scores=scores, prompt=prompt, response=response)


__all__ = ["RAGPipeline", "RagRanking"]
