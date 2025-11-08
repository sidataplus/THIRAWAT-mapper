"""Strength-aware post-scorer with unit/ratio parsing and text fallback."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

RATIO = re.compile(
    r"""
    (?P<a>\d+(?:\.\d+)?)\s*(?P<u>mg|mcg|g|iu|meq|unit|units|unt)\s*/\s*
    (?:(?P<b>\d+(?:\.\d+)?)\s*)?(?P<per>
        ml|mls|l|
        actuation|actuations|actuat|puff|puffs|spray|sprays|drop|drops|
        g|gram|grams|dose|doses|unit|units|unt|
        hr|hour|hours|h|day|days
    )(?![A-Za-z])
    """,
    re.IGNORECASE | re.VERBOSE,
)
PER = re.compile(
    r"""
    (?P<a>\d+(?:\.\d+)?)\s*(?P<u>mg|mcg|g|iu|meq|unit|units|unt)\s*per\s*
    (?:(?P<b>\d+(?:\.\d+)?)\s*)?(?P<per>
        ml|mls|l|
        actuation|actuations|actuat|puff|puffs|spray|sprays|drop|drops|
        g|gram|grams|dose|doses|unit|units|unt|
        hr|hour|hours|h|day|days
    )(?![A-Za-z])
    """,
    re.IGNORECASE | re.VERBOSE,
)
EXPLICIT_COMBO = re.compile(
    r"""
    (?P<a>\d+(?:\.\d+)?)\s*(?P<u>mg|mcg|g|iu|meq)\s*
    (?:/|\+|&|with)\s*
    (?P<b>\d+(?:\.\d+)?)\s*(?P<u2>mg|mcg|g|iu|meq)
    """,
    re.IGNORECASE | re.VERBOSE,
)
HYPHEN_COMBO = re.compile(r"(?P<a>\d+(?:\.\d+)?)\s*[-–]\s*(?P<b>\d+(?:\.\d+)?)\s*(?P<u>mg|mcg|g|iu)\b", re.IGNORECASE)
SINGLE = re.compile(r"(?P<a>\d+(?:\.\d+)?)\s*(?P<u>mg|mcg|g|iu|meq|unit|units|unt)\b", re.IGNORECASE)
PERCENT = re.compile(r"(?P<a>\d+(?:\.\d+)?)\s?%(?=$|\s)", re.IGNORECASE)
STOP_WORDS = {"and", "with", "of", "the", "to", "for", "in", "by", "per"}
BRAND_IN_BRACKETS = re.compile(r"\[([^\[\]]+)\]")
UNITLESS_CHAIN = re.compile(
    r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)(?:\s*(?:/|\+|&|and)\s*(\d+(?:\.\d+)?))+",
    re.IGNORECASE,
)

# No hard-coded INN/USAN alias folding per guidance.


@dataclass(frozen=True)
class StrengthComponent:
    """Normalized representation of a strength element."""

    kind: str
    value: float
    unit: str | None = None
    denom_value: float | None = None
    denom_unit: str | None = None

    def bucket(self) -> tuple[str | None, str | None]:  # grouping key
        if self.kind == "single":
            if self.unit in {None, "mg", "mcg", "g"}:
                return ("mass_single", None)
            return (self.unit, None)
        if self.kind == "ratio":
            return (self.unit, self.denom_unit)
        return ("percent", None)

    def normalized(self) -> float:
        if self.kind == "ratio":
            denom = self.denom_value if self.denom_value else 1.0
            return self.value / max(denom, 1e-9)
        return self.value


def _canon_num_unit(amount: float, unit: str) -> tuple[float, str]:
    u = unit.lower()
    if u in {"µg", "μg", "ug"}:
        return amount, "mcg"
    if u in {"gram", "grams"}:
        return amount * 1000.0, "mg"
    if u == "g":
        return amount * 1000.0, "mg"
    if u == "l":
        return amount * 1000.0, "ml"
    if u == "mls":
        return amount, "ml"
    if u in {"unit", "units", "unt"}:
        return amount, "unit"
    if u in {"puff", "puffs", "actuations", "actuation", "actuat", "spray", "sprays"}:
        return amount, "actuation"
    if u == "drops":
        return amount, "drop"
    if u == "doses":
        return amount, "dose"
    if u in {"hr", "hour", "hours", "h"}:
        return amount, "hr"
    return amount, u


def _canon_den_unit(amount: float, unit: str) -> tuple[float, str]:
    u = unit.lower()
    if u in {"l"}:
        return amount * 1000.0, "ml"
    if u in {"ml", "mls"}:
        return amount, "ml"
    if u in {"gram", "grams", "g"}:
        return amount, "g"
    if u == "mg":
        return amount / 1000.0, "g"
    if u == "mcg":
        return amount / 1_000_000.0, "g"
    if u in {"puff", "puffs", "actuation", "actuations", "actuat", "spray", "sprays"}:
        return amount, "actuation"
    if u in {"drop", "drops"}:
        return amount, "drop"
    if u in {"dose", "doses"}:
        return amount, "dose"
    if u in {"unit", "units", "unt"}:
        return amount, "unit"
    if u in {"hr", "hour", "hours", "h"}:
        return amount, "hr"
    if u in {"day", "days"}:
        return amount, "day"
    return amount, unit.lower()


def _add_percent_if_possible(
    components: List[StrengthComponent],
    num_value: float,
    num_unit: str,
    denom: float,
    denom_unit: str,
) -> None:
    if denom <= 0:
        return
    grams = None
    if num_unit == "mg":
        grams = num_value / 1000.0
    elif num_unit == "mcg":
        grams = num_value / 1_000_000.0
    elif num_unit == "g":
        grams = num_value
    if grams is None:
        return
    if denom_unit not in {"ml", "g"}:
        return
    pct = (grams / denom) * 100.0
    components.append(StrengthComponent("percent", pct))


def extract_strengths_with_spans(text: str) -> tuple[List[StrengthComponent], List[tuple[int, int]]]:
    spans: List[tuple[int, int]] = []
    components: List[StrengthComponent] = []

    def free(start: int, end: int) -> bool:
        return all(end <= s or start >= e for s, e in spans)

    def handle_ratio(match: re.Match[str]) -> None:
        if not free(match.start(), match.end()):
            return
        num, num_unit = _canon_num_unit(float(match.group("a")), match.group("u"))
        raw_den = match.group("b")
        den_value = float(raw_den) if raw_den is not None else 1.0
        den, den_unit = _canon_den_unit(den_value, match.group("per"))
        components.append(
            StrengthComponent("ratio", num, unit=num_unit, denom_value=den, denom_unit=den_unit)
        )
        spans.append((match.start(), match.end()))
        _add_percent_if_possible(components, num, num_unit, den, den_unit)

    for matcher in (RATIO, PER):
        for m in matcher.finditer(text):
            handle_ratio(m)

    for m in EXPLICIT_COMBO.finditer(text):
        if not free(m.start(), m.end()):
            continue
        a, ua = _canon_num_unit(float(m.group("a")), m.group("u"))
        b, ub = _canon_num_unit(float(m.group("b")), m.group("u2"))
        components.extend(
            [StrengthComponent("single", a, unit=ua), StrengthComponent("single", b, unit=ub)]
        )
        spans.append((m.start(), m.end()))

    for m in HYPHEN_COMBO.finditer(text):
        if not free(m.start(), m.end()):
            continue
        a, unit = _canon_num_unit(float(m.group("a")), m.group("u"))
        b, _ = _canon_num_unit(float(m.group("b")), m.group("u"))
        components.extend(
            [StrengthComponent("single", a, unit=unit), StrengthComponent("single", b, unit=unit)]
        )
        spans.append((m.start(), m.end()))

    for m in PERCENT.finditer(text):
        if not free(m.start(), m.end()):
            continue
        components.append(StrengthComponent("percent", float(m.group("a"))))
        spans.append((m.start(), m.end()))

    for m in SINGLE.finditer(text):
        if not free(m.start(), m.end()):
            continue
        value, unit = _canon_num_unit(float(m.group("a")), m.group("u"))
        components.append(StrengthComponent("single", value, unit=unit))
        spans.append((m.start(), m.end()))

    # Capture unitless magnitude chains like "2.5/500" only when no explicit
    # mass singles (mg/mcg/g) were found. This prevents noisy slashed codes in
    # candidate strings from swamping the assignment.
    has_mass_single = any(c.kind == "single" and c.unit in {"mg", "mcg", "g"} for c in components)
    if not has_mass_single:
        for m in UNITLESS_CHAIN.finditer(text):
            if free(m.start(), m.end()):
                chain = text[m.start(): m.end()]
                nums = re.findall(r"\d+(?:\.\d+)?", chain)
                if len(nums) >= 2:
                    for n in nums:
                        try:
                            components.append(StrengthComponent("single", float(n), unit=None))
                        except Exception:
                            pass
                    spans.append((m.start(), m.end()))

    return components, spans


def strip_spans(text: str, spans: Sequence[tuple[int, int]]) -> str:
    if not spans:
        return text
    sorted_spans = sorted(spans)
    buffer: List[str] = []
    cursor = 0
    for start, end in sorted_spans:
        buffer.append(text[cursor:start])
        cursor = end
    buffer.append(text[cursor:])
    return " ".join(buffer)


def _unit_bucket(values: List[StrengthComponent]) -> Dict[tuple[str | None, str | None], List[float]]:
    buckets: Dict[tuple[str | None, str | None], List[float]] = {}
    for comp in values:
        buckets.setdefault(comp.bucket(), []).append(comp.normalized())
    return buckets


def _min_cost_assignment(cost: List[List[float]], dummy_cost: float) -> tuple[float, int]:
    """Bitmask-DP linear assignment for small matrices; returns (min_cost, real_matches).

    cost: rectangular matrix (m x k). We pad to n = max(m, k) with dummy costs.
    A pair assigned to a dummy column/row gets cost = dummy_cost and is not counted
    as a real match in the returned real_matches.
    """
    if not cost:
        return 0.0, 0
    m = len(cost)
    k = max(len(row) for row in cost) if m else 0
    n = max(m, k)
    # Build square matrix
    sq = [[dummy_cost] * n for _ in range(n)]
    for i in range(m):
        row = cost[i]
        for j in range(len(row)):
            sq[i][j] = row[j]

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dp(i: int, used_mask: int) -> float:
        if i == n:
            return 0.0
        best = float("inf")
        r = sq[i]
        for j in range(n):
            if not (used_mask & (1 << j)):
                cand = r[j] + dp(i + 1, used_mask | (1 << j))
                if cand < best:
                    best = cand
        return best

    # Recover count of real matches using a backtrack guided by dp values.
    total = dp(0, 0)
    real = 0
    i, mask = 0, 0
    while i < n:
        r = sq[i]
        chosen = None
        for j in range(n):
            if mask & (1 << j):
                continue
            if abs((r[j] + dp(i + 1, mask | (1 << j))) - total) <= 1e-9:
                chosen = j
                total -= r[j]
                mask |= 1 << j
                break
        if chosen is None:
            break
        if r[chosen] < dummy_cost - 1e-9:
            real += 1
        i += 1

    return dp(0, 0), real


def _dose_gate_and_extra(
    q_buckets: Dict[tuple[str | None, str | None], List[float]],
    d_buckets: Dict[tuple[str | None, str | None], List[float]],
    *,
    tau: float = 0.6,
    kappa_extra: float = 0.7,
) -> tuple[float, float]:
    """Compute dose gate metric S_dose and extra-active penalty P_extra.

    - S_dose: geometric mean of per-pair exp(-|log ratio|) across all real matches
    - P_extra: exp(-kappa_extra * count_extras) where extras are unmatched doc values
    """
    eps = 1e-12
    total_real = 0
    sum_cost_real = 0.0
    extras_total = 0
    def _dedupe(vals: List[float], *, log_tol: float = 0.01, eps: float = 1e-12) -> List[float]:
        if not vals:
            return vals
        vals = sorted(vals)
        out = [vals[0]]
        for v in vals[1:]:
            prev = out[-1]
            if abs(math.log(max(v, eps) / max(prev, eps))) > log_tol:
                out.append(v)
        return out

    for bucket in q_buckets.keys() | d_buckets.keys():
        q_vals = _dedupe(list(q_buckets.get(bucket, [])))
        d_vals = _dedupe(list(d_buckets.get(bucket, [])))
        if not q_vals or not d_vals:
            continue
        cost: List[List[float]] = []
        for qv in q_vals:
            row = [abs(math.log(max(qv, eps) / max(dv, eps))) for dv in d_vals]
            cost.append(row)
        dummy_cost = 8.0
        total_cost, real = _min_cost_assignment(cost, dummy_cost)
        if real > 0:
            n = max(len(q_vals), len(d_vals))
            real_cost_sum = total_cost - (n - real) * dummy_cost
            sum_cost_real += max(real_cost_sum, 0.0)
            total_real += real
            extras_total += max(len(d_vals) - real, 0)
    if total_real <= 0:
        return 0.0, 1.0
    mean_cost = sum_cost_real / total_real
    s_dose = math.exp(-mean_cost)
    p_extra = math.exp(-kappa_extra * extras_total)
    if s_dose < tau:
        return 0.0, p_extra
    return s_dose, p_extra


def _sim(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return 0.0
    return math.exp(-abs(math.log(a / b)))


def strength_sim(query: str, candidate: str) -> float:
    q_comp, _ = extract_strengths_with_spans(query)
    d_comp, _ = extract_strengths_with_spans(candidate)
    q_buckets = _unit_bucket(q_comp)
    d_buckets = _unit_bucket(d_comp)
    # Use the same assignment-based measure as the dose gate (without gating)
    s_dose, p_extra = _dose_gate_and_extra(q_buckets, d_buckets, tau=0.0, kappa_extra=0.7)
    return float(s_dose * p_extra)


def _tokenize(text: str) -> List[str]:
    lower = text.lower()
    # Remove brand chunks to avoid penalizing branded candidates in remainder.
    lower = re.sub(r"\[[^\]]+\]", " ", lower)
    lower = re.sub(r"[\(\)\[\],;:]", " ", lower)
    lower = re.sub(r"[-_/]", " ", lower)
    tokens = re.split(r"[^a-z0-9]+", lower)
    return [tok for tok in tokens if tok and tok not in STOP_WORDS and not tok.isdigit()]


def _bigrams(tokens: Sequence[str]) -> set[str]:
    return {f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)} if len(tokens) > 1 else set()


def jaccard_remainder(query: str, candidate: str) -> float:
    _, q_spans = extract_strengths_with_spans(query)
    _, d_spans = extract_strengths_with_spans(candidate)
    q_rem = strip_spans(query, q_spans)
    d_rem = strip_spans(candidate, d_spans)
    q_tokens = _tokenize(q_rem)
    d_tokens = _tokenize(d_rem)
    set_q = set(q_tokens)
    set_d = set(d_tokens)
    union = set_q | set_d
    j1 = (len(set_q & set_d) / len(union)) if union else 0.0
    b_q = _bigrams(q_tokens)
    b_d = _bigrams(d_tokens)
    b_union = b_q | b_d
    j2 = (len(b_q & b_d) / len(b_union)) if b_union else 0.0
    return 0.7 * j1 + 0.3 * j2


def _normalize_brand_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _extract_document_brands(candidate: str) -> List[str]:
    return [
        normalized
        for raw in BRAND_IN_BRACKETS.findall(candidate)
        if (normalized := _normalize_brand_text(raw))
    ]


def brand_score(query: str, candidate: str) -> float:
    """Return a non-positive penalty for brand mismatches.

    - Candidate brand(s) (from bracketed segments) must appear in the normalized query text.
    - If a candidate exposes a brand but the query text never mentions it, assign -1.0.
    - Missing brands on the candidate (or matching brands) produce 0.0 (no penalty).
    """

    candidate_brands = _extract_document_brands(candidate)
    if not candidate_brands:
        return 0.0

    query_norm = _normalize_brand_text(query)
    for brand in candidate_brands:
        if brand and brand in query_norm:
            return 0.0

    return -1.0


def simple_strength_plus_jaccard(
    query: str,
    candidate: str,
    *,
    w_strength: float = 0.6,
    w_jaccard: float = 0.4,
    w_brand_penalty: float = 0.3,
) -> Dict[str, float]:
    # Dose-first: compute S_dose (gated) and P_extra from assignment; fall back to
    # overall strength_sim for logging only.
    q_comp, _ = extract_strengths_with_spans(query)
    d_comp, _ = extract_strengths_with_spans(candidate)
    q_buckets = _unit_bucket(q_comp)
    d_buckets = _unit_bucket(d_comp)
    s_dose, p_extra = _dose_gate_and_extra(q_buckets, d_buckets, tau=0.6, kappa_extra=0.7)
    strength = strength_sim(query, candidate)
    jaccard = jaccard_remainder(query, candidate)
    brand = brand_score(query, candidate)
    # If gated off, score is zero regardless of other parts
    if s_dose <= 0.0:
        return {
            "strength_sim": float(strength),
            "jaccard_text": float(jaccard),
            "brand_score": float(brand),
            "post_score": 0.0,
            # Back-compat alias
            "simple_score": 0.0,
        }
    pos_weight = max(w_strength + w_jaccard, 1e-9)
    # Geometric blend of (S_dose * P_extra) with remainder similarity
    base = (max(s_dose * p_extra, 1e-12) ** w_strength) * (max(jaccard, 1e-12) ** w_jaccard)
    base = base ** (1.0 / pos_weight)
    brand_factor = math.exp(w_brand_penalty * brand)
    post_score = base * brand_factor
    return {
        "strength_sim": float(strength),
        "jaccard_text": float(jaccard),
        "brand_score": float(brand),
        "post_score": float(post_score),
        # Back-compat alias
        "simple_score": float(post_score),
    }


def _minmax(values: List[float]) -> List[float]:
    if not values:
        return values
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def batch_features(
    query: str,
    candidates: Sequence[str],
    *,
    w_strength: float = 0.6,
    w_jaccard: float = 0.4,
    w_brand_penalty: float = 0.3,
    minmax_within_query: bool = False,
) -> Dict[str, List[float]]:
    strength_scores: List[float] = []
    jaccard_scores: List[float] = []
    brand_scores: List[float] = []
    for candidate in candidates:
        features = simple_strength_plus_jaccard(
            query,
            candidate,
            w_strength=w_strength,
            w_jaccard=w_jaccard,
            w_brand_penalty=w_brand_penalty,
        )
        strength_scores.append(features["strength_sim"])
        jaccard_scores.append(features["jaccard_text"])
        brand_scores.append(features["brand_score"])

    if minmax_within_query:
        strength_norm = _minmax(strength_scores)
        jaccard_norm = _minmax(jaccard_scores)
    else:
        strength_norm = strength_scores
        jaccard_norm = jaccard_scores
    pos_weight = max(w_strength + w_jaccard, 1e-9)
    post = []
    for s, j, b in zip(strength_norm, jaccard_norm, brand_scores):
        gm = (max(s, 1e-12) ** w_strength) * (max(j, 1e-12) ** w_jaccard)
        gm = gm ** (1.0 / pos_weight)
        brand_factor = math.exp(w_brand_penalty * b)
        post.append(gm * brand_factor)
    return {
        "strength_sim": strength_scores,
        "jaccard_text": jaccard_scores,
        "brand_score": brand_scores,
        "post_score": post,
        # Back-compat alias
        "simple_score": post,
    }


__all__ = [
    "extract_strengths_with_spans",
    "strip_spans",
    "strength_sim",
    "jaccard_remainder",
    "brand_score",
    "simple_strength_plus_jaccard",
    "batch_features",
]
