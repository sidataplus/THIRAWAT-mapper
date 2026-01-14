"""Prompt construction helpers for LLM-based reranking."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

DEFAULT = """You are ranking candidate RxNorm drug concepts for a given query. Each candidate string ends with its numeric concept_id in parentheses -- keep those IDs intact.

## Inputs
- Query: ingredient(s), dose (value + unit), dosage form/route/release, optional brand.
- Candidates: RxNorm strings (brand often appears in trailing brackets).

## Output
{format_instructions}

## Normalization
- Case-insensitive; normalize whitespace; ignore minor punctuation.
- Units: mg = MG; mcg = ug; g = gram; mL = milliliter.
- Decimal separators \",\" and \".\" equivalent; ignore trailing zeros.
- Form terms can be transliterated for language variants (e.g., tableta/comprimido = tablet).
- When a candidate has a trailing bracketed brand, treat that bracketed segment as the brand indicator.
- Candidates usually use USAN for ingredients, which have equivalents in INN.

## Selection rules (4-part ladder; higher priority -> lower priority)
Score each candidate on 4 parts: ingredient, dose, form/route/release, brand.
1) 4/4 match: all four parts match (brand required only if the query names a brand).
2) 3/4 match: any three parts match; prefer missing brand over missing ingredient/dose/form.
3) 2/4 match: prefer ingredient + (dose or form).
4) 1/4 match: ingredient only (last resort).

Soft matching is allowed for dose and form:
- Dose conversions are ok when clearly equivalent (mg<->g, mcg<->mg, %<->mg/mL, mg/mL<->mg/5 mL, mass<->volume when reasonable).
- Form/route terms can match even if word order differs (e.g., \"tablet chewable\" = \"chewable tablet\").

## Demotions (do not hard-exclude)
- Brand mismatch when query names a brand -> strongly demote.
- Extra unmentioned actives, pack/kit indicators, or extra strength/volume terms -> demote.
- If query is generic, prefer unbranded over branded.

## Ordering (deterministic)
Sort by highest match tier, then by closer form match, then by closer dose match.

## Examples

Example A (generic query; prefer unbranded)
Query: Paracetamol 500 mg Oral Tablet
Candidates:
  Acetaminophen 500 MG Oral Tablet (123456)
  Acetaminophen 500 MG Oral Tablet [Tylenol] (789012)
Answer:
  123456,789012

Example B (brand query; demote mismatched brand)
Query: Ventolin HFA 90 mcg/actuation Inhalation Aerosol
Candidates:
  albuterol 90 MCG/ACTUATION Inhalation Aerosol (555111)
  albuterol 90 MCG/ACTUATION Inhalation Aerosol [Ventolin HFA] (555222)
  albuterol 90 MCG/ACTUATION Inhalation Aerosol [ProAir HFA] (555333)
Answer:
  555222,555111

Example C (soft dose conversion)
Query: Hydrocortisone 2.5% cream
Candidates:
  hydrocortisone 25 MG/ML cream (101010)
  hydrocortisone 2.5 MG/ML cream (202020)
Answer:
  101010,202020

---

## Query
{source}

## Candidates (pre-ranked list)
{candidates}
"""

_QUERY_MARKER = "<<__QUERY__>>"
_CANDIDATES_MARKER = "<<__CANDIDATES__>>"


@dataclass(frozen=True)
class RagCandidate:
    concept_id: int
    concept_name: Optional[str] = None
    domain_id: Optional[str] = None
    vocabulary_id: Optional[str] = None
    concept_code: Optional[str] = None
    concept_class_id: Optional[str] = None
    profile_text: Optional[str] = None
    retrieval_score: Optional[float] = None
    final_score: Optional[float] = None


class RAGPromptBuilder:
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
        candidate_strings: List[str] = []
        for cand in candidates:
            name = (cand.concept_name or cand.profile_text or "").strip()
            surface = name if name else f"Concept {cand.concept_id}"
            candidate_text = f"{surface} ({cand.concept_id})"
            candidate_strings.append(candidate_text)
            parts = [candidate_text]
            if self.include_retrieval_score and cand.retrieval_score is not None:
                parts.append(f"retrieval_score={cand.retrieval_score:.4f}")
            if self.include_final_score and cand.final_score is not None:
                parts.append(f"final_score={cand.final_score:.4f}")
            lines.append(" | ".join(parts))
        return lines, candidate_strings

    def build_parts(
        self,
        query_text: str,
        candidates: Sequence[RagCandidate],
        *,
        extra_context: Optional[str] = None,
        format_instructions: Optional[str] = None,
    ) -> Tuple[List[str], str]:
        if not candidates:
            raise ValueError("RAG prompt requires at least one candidate.")
        lines, _ = self._candidate_lines(candidates)
        candidate_block = "\n".join(lines)
        if extra_context:
            candidate_block_with_context = f"Additional context: {extra_context.strip()}\n\n{candidate_block}"
        else:
            candidate_block_with_context = candidate_block

        template_placeholders = ["{source}", "{candidates}", "{format_instructions}"]
        if any(token in self.instructions for token in template_placeholders):
            prompt = self.instructions
            prompt = prompt.replace("{source}", _QUERY_MARKER)
            prompt = prompt.replace("{candidates}", _CANDIDATES_MARKER)
            prompt = prompt.replace("{format_instructions}", (format_instructions or "").strip())
            candidates_text = candidate_block_with_context
        else:
            sections = [self.instructions.strip(), "", f"Query: {_QUERY_MARKER}"]
            if extra_context:
                sections.extend(["", f"Additional context: {extra_context.strip()}"])
            sections.append("")
            sections.append("Candidates (highest score first):")
            sections.append(_CANDIDATES_MARKER)
            if format_instructions:
                sections.extend(["", "Output:", format_instructions.strip()])
            prompt = "\n".join(sections)
            candidates_text = candidate_block

        prompt = prompt.replace("{format_instructions}", "")
        prompt = re.sub(r"\n{3,}", "\n\n", prompt).strip()

        if _QUERY_MARKER not in prompt or _CANDIDATES_MARKER not in prompt:
            return [prompt], candidates_text
        if prompt.index(_QUERY_MARKER) > prompt.index(_CANDIDATES_MARKER):
            return [prompt], candidates_text
        before, rest = prompt.split(_QUERY_MARKER, 1)
        between, after = rest.split(_CANDIDATES_MARKER, 1)
        return [before, between, after], candidates_text

    def build(
        self,
        query_text: str,
        candidates: Sequence[RagCandidate],
        *,
        extra_context: Optional[str] = None,
        format_instructions: Optional[str] = None,
    ) -> Tuple[str, List[str]]:
        if not candidates:
            raise ValueError("RAG prompt requires at least one candidate.")
        lines, candidate_strings = self._candidate_lines(candidates)
        candidate_block = "\n".join(lines)
        if extra_context:
            candidate_block = f"Additional context: {extra_context.strip()}\n\n{candidate_block}"

        template_placeholders = ["{source}", "{candidates}", "{format_instructions}"]
        if any(token in self.instructions for token in template_placeholders):
            prompt = self.instructions
            prompt = prompt.replace("{source}", query_text.strip())
            prompt = prompt.replace("{candidates}", candidate_block)
            prompt = prompt.replace("{format_instructions}", (format_instructions or "").strip())
        else:
            sections = [self.instructions.strip(), "", f"Query: {query_text.strip()}"]
            if extra_context:
                sections.extend(["", f"Additional context: {extra_context.strip()}"])
            sections.append("")
            sections.append("Candidates (highest score first):")
            sections.extend(lines)
            if format_instructions:
                sections.extend(["", "Output:", format_instructions.strip()])
            prompt = "\n".join(sections)
        prompt = prompt.replace("{format_instructions}", "")
        prompt = re.sub(r"\n{3,}", "\n\n", prompt).strip()
        return prompt, candidate_strings


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
        candidate_strings = [self._candidate_line(c) for c in candidates]
        prompt = (
            self.template.replace("{source}", query_text.strip())
            .replace("{candidates}", "\n".join(candidate_strings))
            .replace("{format_instructions}", format_instructions.strip())
        )
        if extra_context:
            prompt = prompt.replace("{extra_context}", extra_context.strip())
        else:
            prompt = prompt.replace("{extra_context}", "")
        return prompt, candidate_strings


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
