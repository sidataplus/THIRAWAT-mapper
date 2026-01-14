"""Coordination utilities for LLM-based reranking."""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .rag_llm import BaseLLMClient
from .rag_prompt import RAGPromptBuilder, RagCandidate

LOGGER = logging.getLogger(__name__)

_STRUCTURED_INSTRUCTIONS = (
    '- Return JSON with field "concept_ids", an array of candidate concept_ids ordered best → worst.\n'
    '- Example: {"concept_ids":[111111,222222,333333]}\n'
    "- No commentary, quotes, code fences, or additional text."
)


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


_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", flags=re.DOTALL | re.IGNORECASE)


def _extract_json_blob(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return ""
    m = _CODE_FENCE_RE.search(stripped)
    if m:
        return m.group(1).strip()
    return stripped


def _parse_structured_concept_ids(response: str) -> List[int]:
    blob = _extract_json_blob(response)
    if not blob:
        raise ValueError("empty LLM response")
    try:
        payload = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}; response={response!r}") from exc

    if isinstance(payload, dict):
        value = payload.get("concept_ids")
    elif isinstance(payload, list):
        value = payload
    else:
        value = None

    if not isinstance(value, list) or not value:
        raise ValueError(f'JSON must contain key "concept_ids" with a non-empty array of integers; response={response!r}')

    out: List[int] = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError("concept_ids must be integers (bool not allowed)")
        try:
            cid = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"concept_ids must be integers: {item}") from exc
        out.append(cid)
    return out


class RAGPipeline:
    """End-to-end helper that builds prompts, calls the LLM, and parses results."""

    def __init__(
        self,
        llm: BaseLLMClient,
        *,
        prompt_builder: Optional[RAGPromptBuilder] = None,
    ) -> None:
        self.llm = llm
        self.prompt_builder = prompt_builder or RAGPromptBuilder()

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
            format_instructions=_STRUCTURED_INSTRUCTIONS,
        )
        concept_ids = [cand.concept_id for cand in candidates]
        response = self.llm.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop_sequences,
        )

        # Structured output required: parse JSON and validate IDs against the candidate set.
        parsed_order = _parse_structured_concept_ids(response)
        candidate_set = set(concept_ids)
        ordered: List[int] = []
        seen: set[int] = set()
        for cid in parsed_order:
            if cid not in candidate_set or cid in seen:
                continue
            seen.add(cid)
            ordered.append(cid)
        if not ordered:
            raise ValueError(f"LLM returned no valid concept_ids from the candidate set: {response!r}")
        remaining = [cid for cid in concept_ids if cid not in seen]
        ordered.extend(remaining)

        score_map: Dict[int, float] = {}

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
