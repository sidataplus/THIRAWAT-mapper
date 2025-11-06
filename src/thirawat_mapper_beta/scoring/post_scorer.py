"""Lightweight strength-aware post-scorer."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


RATIO = re.compile(
    r"""(?P<num>\d+(?:\.\d+)?)\s*(?P<num_unit>mg|mcg|g|iu|meq|unit|unt)\s*
        /\s*(?P<den>\d+(?:\.\d+)?)\s*(?P<den_unit>ml|l|actuation|actuat|puff|spray|drop|g|hr|h|dose)""",
    re.IGNORECASE | re.VERBOSE,
)
PERCENT = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s?%", re.IGNORECASE)
SINGLE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mg|mcg|g|iu|unit|unt|meq|ml|l)\b", re.IGNORECASE)
CONNECTOR = re.compile(r"(?:\bwith\b|\band\b|\+|&|/)\s*", re.IGNORECASE)
STOP_WORDS = {"and", "with", "of", "the", "to", "for", "in", "by", "per"}


def _canon_unit(amount: float, unit: str) -> tuple[float, str]:
    u = unit.lower()
    if u in {"µg", "μg", "ug"}:
        return amount, "mcg"
    if u == "g":
        return amount * 1000.0, "mg"
    if u == "l":
        return amount * 1000.0, "ml"
    if u in {"iu", "unit", "unt"}:
        return amount, "unit"
    if u == "actuat":
        return amount, "actuation"
    if u == "h":
        return amount, "hr"
    return amount, u


def _canon_den_unit(amount: float, unit: str) -> tuple[float, str]:
    value, canonical = _canon_unit(amount, unit)
    if canonical == "actuation":
        return value, "actuation"
    return value, canonical


@dataclass
class StrengthComponent:
    key: tuple[str, str | None]
    value: float


def _extract_strengths(text: str) -> tuple[List[StrengthComponent], List[tuple[int, int]]]:
    components: List[StrengthComponent] = []
    spans: List[tuple[int, int]] = []

    def _free(start: int, end: int) -> bool:
        return all(end <= s or start >= e for s, e in spans)

    for match in RATIO.finditer(text):
        if not _free(match.start(), match.end()):
            continue
        raw_den = float(match.group("den"))
        num, num_unit = _canon_unit(float(match.group("num")), match.group("num_unit"))
        den, den_unit = _canon_den_unit(raw_den, match.group("den_unit"))
        key = (num_unit, den_unit)
        value = num / max(den, 1e-9)
        components.append(StrengthComponent(key, value))
        spans.append((match.start(), match.end()))


    for match in PERCENT.finditer(text):
        if not _free(match.start(), match.end()):
            continue
        value = float(match.group("value"))
        components.append(StrengthComponent(("percent", None), value))
        spans.append((match.start(), match.end()))

    for match in SINGLE.finditer(text):
        if not _free(match.start(), match.end()):
            continue
        amount, unit = _canon_unit(float(match.group("value")), match.group("unit"))
        components.append(StrengthComponent((unit, None), amount))
        spans.append((match.start(), match.end()))

    return components, spans


def _strip_spans(text: str, spans: Sequence[tuple[int, int]]) -> str:
    if not spans:
        return text
    spans = sorted(spans)
    chunks: List[str] = []
    cursor = 0
    for start, end in spans:
        chunks.append(text[cursor:start])
        cursor = end
    chunks.append(text[cursor:])
    return " ".join(chunks)


def _greedy_match(query: List[StrengthComponent], doc: List[StrengthComponent]) -> float:
    if not query:
        return 0.0
    doc_used = [False] * len(doc)
    total = 0.0
    for comp in query:
        best = 0.0
        best_idx = -1
        for idx, candidate in enumerate(doc):
            if doc_used[idx] or comp.key != candidate.key:
                continue
            qv, dv = comp.value, candidate.value
            if qv <= 0 or dv <= 0:
                continue
            score = math.exp(-abs(math.log(qv / dv)))
            if score > best:
                best = score
                best_idx = idx
        if best_idx >= 0:
            doc_used[best_idx] = True
        total += best
    return total / len(query)


def _tokenize(text: str) -> List[str]:
    tokens = re.split(r"[^a-z0-9]+", text.lower())
    return [tok for tok in tokens if tok and tok not in STOP_WORDS and not tok.isdigit()]


def _jaccard(xs: List[str], ys: List[str]) -> float:
    if not xs and not ys:
        return 0.0
    set_x, set_y = set(xs), set(ys)
    union = set_x | set_y
    if not union:
        return 0.0
    inter = set_x & set_y
    return len(inter) / len(union)


def strength_sim(query: str, candidate: str) -> float:
    q_comp, _ = _extract_strengths(query)
    d_comp, _ = _extract_strengths(candidate)
    return _greedy_match(q_comp, d_comp)


def jaccard_remainder(query: str, candidate: str) -> float:
    q_comp, q_spans = _extract_strengths(query)
    d_comp, d_spans = _extract_strengths(candidate)
    q_rem = _strip_spans(query, q_spans)
    d_rem = _strip_spans(candidate, d_spans)
    tokens_q = _tokenize(q_rem)
    tokens_d = _tokenize(d_rem)
    return _jaccard(tokens_q, tokens_d)


def simple_strength_plus_jaccard(query: str, candidate: str) -> Dict[str, float]:
    s = strength_sim(query, candidate)
    j = jaccard_remainder(query, candidate)
    combined = 0.6 * s + 0.4 * j
    return {
        "strength_sim": float(s),
        "jaccard_text": float(j),
        "simple_score": float(combined),
    }


def batch_features(query: str, candidates: Sequence[str]) -> List[Dict[str, float]]:
    return [simple_strength_plus_jaccard(query, cand) for cand in candidates]
