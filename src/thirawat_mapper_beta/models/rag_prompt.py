"""Prompt construction helpers for LLM-based reranking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

DEFAULT = """You are ranking candidate RxNorm drug concepts for a given query. Each candidate string ends with its numeric concept_id in parentheses — keep those IDs intact.

## Assumptions:
- Options are RxNorm strings.
- Brand indicators often appear in brackets (e.g., `[...]`), but they may also be part of the main text (prefix/suffix) without brackets. Treat case-insensitive matches in either position as potential brand mentions.
- The Query can be any language (non-English allowed).
- Return ONLY exact strings from Options.

## Inputs:
- Query: may include ingredient, dose (value + unit), dosage form, route, release modifiers, and possibly a brand.
- Options: list of RxNorm candidate strings.

## Output:
- Return a comma-separated list of candidate strings, ordered best → worst.
- Copy each option exactly as shown (including casing and the trailing `(concept_id)`).
- If none qualify, return an empty string.

## Normalization for matching (do not alter output strings):
- Case-insensitive; trim/normalize whitespace; ignore minor punctuation.
- Units: “mg” ≡ “MG”; “mcg” ≡ “µg”; “g” ≡ “gram”; “mL” ≡ “milliliter”.
- Decimal separators “,” and “.” equivalent (2,5 mg ≡ 2.5 mg). Ignore trailing zeros (2.50 ≡ 2.5).
- Dosage form includes route and release modifiers (e.g., “Oral Tablet”, “Extended-Release Capsule”, “Delayed-Release Tablet”, “Transdermal Patch”).
- When an Option has a trailing bracketed brand, use only the **non-bracketed head** for ingredient/dose/form matching; treat the bracketed segment as the brand indicator.
- Non-English: allow transliteration for form terms (e.g., “tableta/comprimido/เม็ด” ≡ “Tablet”), but do not invent ingredients.

## INN↔USAN alias pairs (ingredient-level only; do NOT cross salts/esters/hydrates):
- acetaminophen ≡ paracetamol; epinephrine ≡ adrenaline; norepinephrine ≡ noradrenaline;
  albuterol ≡ salbutamol; nitroglycerin ≡ glyceryl trinitrate; lidocaine ≡ lignocaine;
  furosemide ≡ frusemide; dipyrone ≡ metamizole; levothyroxine ↔ thyroxine (base-only contexts).

## Selection rules (strict → relaxed):
1) 4/4 exact match (brand queries only): ingredient + exact dose (numeric + unit) + exact dosage form (including route/modifier) + exact brand (if the query explicitly names a brand). If no brand is supplied in the query, skip directly to Rule 2. A candidate whose bracketed brand does not match the query's brand is NOT a 4/4; treat its brand component as missing and demote it behind exact or unbranded matches from subsequent rules.

2) 3/3 exact match: ingredient + exact dose + exact dosage form.
   - If the query specifies a specific salt/ester/hydrate/base, match that exact variant.
   - If the Option states “as base/equivalent to …” and numerically equals the specified base/salt, accept as exact.
   - If the query names a brand but no candidate contains that brand, treat the brand component as unmet: select the best unbranded or neutral candidate that satisfies ingredient + dose + form, and place any mismatched-brand options after those unbranded matches.

3) 2/3 match (only if no higher tier exists). Rank:
   3a) Ingredient + Dosage Form (dose missing/unspecified).
   3b) Ingredient + Dose (form missing/unspecified).

4) 1/3 match (ingredient only) — only if no higher tier exists.

- **Generic query** (no brand mentioned): Prefer unbranded SCD/IN/PIN/DF surfaces; when the same clinical core appears with branded wording, list the unbranded or neutral option before any branded variants.
- **Brand-specific query**: Prefer candidates whose brand wording (bracketed or inline) matches the query brand (case/punct-insensitive). If the exact brand is missing, fall back to the best 3/3 unbranded SCD and rank mismatched-brand options after those unbranded ones. A mismatched brand should never outrank an otherwise identical unbranded candidate. When choosing the fallback, treat the brand component as absent and rely on ingredient + dose + dosage form equality, even if the unbranded string omits the volume or other packaging hints.
- When a candidate contains a bracketed or inline brand that does not match the query brand, move it to the end of your answer (after all unbranded or brand-neutral options that satisfy the clinical components).
- The candidates below are shown from highest to lowest score; earlier options are stronger matches.
- Packs/Kits (GPCK/BPCK): exclude unless the query clearly requests a pack/kit (“dose pack”, “starter pack”, blister count).
- Combination products: all actives named in the query must be present; exclude any Option with extra unmentioned actives.

## Form and packaging clarifications (apply strictly):
- Tablet vs Chewable: If the query states a plain “Tablet”, prefer “Oral Tablet” and avoid “Chewable Tablet” unless no exact tablet exists.
- Volumes for injections/solutions: When the query specifies a volume (e.g., “5 mL”, “100 mL bag”), prefer options with the exact same volume; avoid generic solutions lacking volume.
- Pack counts: If the query specifies strength/dose but not a pack/quantity, demote or exclude options that indicate a pack count (for example, strings containing “x” quantities like “10 x …”).

## Dose, units, conversions:
- Dose must match exactly when specified. No rounding or near values.
- Prefer exact same unit. Consider conversions only when the Option string itself provides an unambiguous mapping yielding exact equality (e.g., mg↔g scaling; mg/mL↔mg/5 mL when both numerator/denominator explicit; per actuation/patch/hour when explicit).
- Disallow IU↔mg unless the Option explicitly provides exact equivalence.
- If a required concentration is absent from the Option, do not convert.

## Form and route:
- If the query specifies a form, require exact form (including route and release modifier). “Oral Tablet” ≠ “Oral Capsule”; accept “Chewable Tablet” only if no exact “Tablet” match exists and you are already in a lower tier.

## Ordering (deterministic):
- By match tier (3/3 > 2/3 > 1/3),
- then by term-type preference (unbranded SCD > bracketed-brand SBD > IN/PIN > DF > BN > GPCK/BPCK unless the query is explicitly brand/pack),
- then exact-unit > converted-unit,
- then shorter string length,
- then smaller edit distance to the normalized query.
- Remove exact duplicate strings from the final list.

## Response format:
- Example: `Acetaminophen 500 MG Oral Tablet (123456), acetaminophen Oral Tablet (345678)`
- No commentary, quotes, code fences, or additional text.

## Examples

Example A (generic query excludes bracketed brand)
Query: Paracetamol 500 mg Oral Tablet
Options:
  Acetaminophen 500 MG Oral Tablet (123456)
  Acetaminophen 500 MG Oral Tablet [Tylenol] (789012)
  acetaminophen Oral Tablet (345678)
  acetaminophen (901234)
Answer (illustrative):
  Acetaminophen 500 MG Oral Tablet (123456), acetaminophen Oral Tablet (345678), acetaminophen (901234)

Example B (brand query)
Query: Ventolin HFA 90 mcg/actuation Inhalation Aerosol
Options:
  albuterol 90 MCG/ACTUATION Inhalation Aerosol (555111)
  albuterol 90 MCG/ACTUATION Inhalation Aerosol [Ventolin HFA] (555222)
  albuterol 90 MCG/ACTUATION Inhalation Aerosol [ProAir HFA] (555333)
Answer (illustrative):
  albuterol 90 MCG/ACTUATION Inhalation Aerosol [Ventolin HFA] (555222), albuterol 90 MCG/ACTUATION Inhalation Aerosol (555111)
(If that brand Option were absent, fall back to the unbranded in tier 2.)

Example C (brand query without matching candidate)
Query: Insulin isophane (NPH) Inj 10 mL [Gensulin N]
Options:
  10 ML insulin, isophane 100 UNT/ML Injection [Humulin N] (43518547)
  3 ML insulin isophane, human 100 UNT/ML Pen Injector (46234237)
  insulin isophane, human 100 UNT/ML Injectable Suspension (19078552)
Answer (illustrative):
  insulin isophane, human 100 UNT/ML Injectable Suspension (19078552), 3 ML insulin isophane, human 100 UNT/ML Pen Injector (46234237)
(Prefer the unbranded injectable suspension first because the query brand is unavailable; list mismatched brands last even if their scores are higher. Do not place `[Humulin N]` ahead of the unbranded options when the query brand is `[Gensulin N]`.)

## Placeholders:
{format_instructions}

## QUESTION:
Select the term(s) from the RxNorm Options that most closely match the query by ingredient, dose, and dosage form. Do not invent terms.

Query: {source}
Options (ranked best → worst):
-----------------------------------------------
{candidates}
-----------------------------------------------

## Guidance:
- The Options above are ordered from highest to lowest score; however, the selection rules override score ordering. Move lower-scoring options ahead of higher-scoring ones when the rules require it. If you are uncertain, prefer earlier options after applying the rules.
- Do not simply echo the candidate order. Reorder candidates whenever the rules demand it (for example, placing unbranded matches ahead of mismatched brands even if the unbranded option appears later in the list).
- Prioritize exact ingredient, dose, and dosage form matches in that order.
- Ignore the numeric bullet (`1.`, `2.` …) when copying; return only the option text with its `(concept_id)`.
- If you are confident that none of the candidates are correct, return an empty string.
- When multiple candidates are equivalent, keep the earlier one(s) first."""

@dataclass(frozen=True)
class RagCandidate:
    """Container describing a candidate concept passed to the LLM."""

    concept_id: int
    concept_name: Optional[str] = None
    domain_id: Optional[str] = None
    vocabulary_id: Optional[str] = None
    concept_code: Optional[str] = None
    concept_class_id: Optional[str] = None
    profile_text: Optional[str] = None
    retrieval_score: Optional[float] = None
    final_score: Optional[float] = None


def _trim_text(text: Optional[str], limit: int) -> str:
    if not text:
        return ""
    stripped = text.strip()
    if limit and len(stripped) > limit:
        return stripped[:limit].rstrip() + " …"
    return stripped


class RAGPromptBuilder:
    """Formats a textual prompt listing the best retrieval candidates."""

    def __init__(
        self,
        *,
        instructions: str = DEFAULT,
        profile_char_limit: int = 512,
        include_retrieval_score: bool = True,
        include_final_score: bool = True,
    ) -> None:
        self.instructions = instructions
        self.profile_char_limit = max(int(profile_char_limit), 0)
        self.include_retrieval_score = bool(include_retrieval_score)
        self.include_final_score = bool(include_final_score)

    def _candidate_lines(self, candidates: Sequence[RagCandidate]) -> Tuple[List[str], List[str]]:
        lines: List[str] = []
        option_strings: List[str] = []
        for idx, cand in enumerate(candidates, start=1):
            name = (cand.concept_name or cand.profile_text or "").strip()
            surface = name if name else f"Concept {cand.concept_id}"
            option_text = f"{surface} ({cand.concept_id})"
            option_strings.append(option_text)

            line = f"{idx}. {option_text}"
            lines.append(line)
        return lines, option_strings

    def build(
        self,
        query_text: str,
        candidates: Sequence[RagCandidate],
        *,
        extra_context: Optional[str] = None,
    ) -> Tuple[str, List[str]]:
        if not candidates:
            raise ValueError("RAG prompt requires at least one candidate.")
        header = self.instructions.strip()
        sections = [header, "", f"Query: {query_text.strip()}"]
        if extra_context:
            sections.extend(["", f"Additional context: {extra_context.strip()}"])
        sections.append("")
        sections.append("Candidates (highest score first):")
        lines, option_strings = self._candidate_lines(candidates)
        sections.extend(lines)
        sections.extend(
            [
                "",
                "Respond with a comma-separated list of the option texts exactly as written. "
                "If unsure, prefer earlier options. Return an empty string if none match.",
            ]
        )
        prompt = "\n".join(sections)
        return prompt, option_strings


class TemplatePromptBuilder:
    """Prompt builder that renders from a markdown template file."""

    def __init__(
        self,
        template_path: str | Path,
        *,
        candidate_field: str = "profile_text",
        profile_char_limit: int = 512,
    ) -> None:
        self.template_path = Path(template_path)
        self.template = self.template_path.read_text(encoding="utf-8")
        self.candidate_field = str(candidate_field)
        self.profile_char_limit = max(int(profile_char_limit), 0)

    def _candidate_line(self, candidate: RagCandidate) -> str:
        display = getattr(candidate, self.candidate_field, None) or candidate.concept_name or ""
        display = display.strip()
        if self.profile_char_limit and len(display) > self.profile_char_limit:
            display = display[: self.profile_char_limit].rstrip() + " …"
        if not display:
            display = f"Concept {candidate.concept_id}"
        return f"{display} ({candidate.concept_id})"

    def build(
        self,
        query_text: str,
        candidates: Sequence[RagCandidate],
        *,
        extra_context: Optional[str] = None,
        format_instructions: str = "",
    ) -> Tuple[str, List[str]]:
        if not candidates:
            raise ValueError("Template prompt requires at least one candidate.")
        option_strings = [self._candidate_line(c) for c in candidates]
        prompt = (
            self.template.replace("{source}", query_text.strip())
            .replace("{candidates}", "\n".join(option_strings))
            .replace("{format_instructions}", format_instructions.strip())
        )
        if extra_context:
            prompt = prompt.replace("{extra_context}", extra_context.strip())
        else:
            prompt = prompt.replace("{extra_context}", "")
        return prompt, option_strings


def to_candidates(rows: Iterable[dict]) -> List[RagCandidate]:
    """Convert a lightweight dict collection into RagCandidate objects."""

    result: List[RagCandidate] = []
    for row in rows:
        if "concept_id" not in row:
            raise ValueError("Candidate rows must include concept_id.")
        try:
            cid = int(row["concept_id"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"concept_id must be coercible to int: {row.get('concept_id')}") from exc
        result.append(
            RagCandidate(
                concept_id=cid,
                concept_name=row.get("concept_name"),
                domain_id=row.get("domain_id"),
                vocabulary_id=row.get("vocabulary_id"),
                concept_code=row.get("concept_code"),
                concept_class_id=row.get("concept_class_id"),
                profile_text=row.get("profile_text"),
                retrieval_score=row.get("_relevance_score") or row.get("retrieval_score"),
                final_score=row.get("final_score") if row.get("final_score") is not None else None,
            )
        )
    return result


__all__ = ["DEFAULT", "RagCandidate", "RAGPromptBuilder", "TemplatePromptBuilder", "to_candidates"]
