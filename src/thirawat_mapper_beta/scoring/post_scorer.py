"""Strength-aware post-scorer with unit/ratio parsing and text fallback."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

NUM_UNIT_PATTERNS = [
    r"anti[-\s]?xa\s+unit(?:s)?",
    r"d\s*antigen(?:\s+unit)?(?:s)?",
    r"elisa\s*u",
    r"vaccinating\s+dose(?:s)?",
    r"billion\s+organism(?:s)?",
    r"billion\s+spore(?:s)?",
    r"sq[-\s]?hdm",
    r"log10ccid50",
    r"lgccid50",
    r"ccid50",
    r"tcid50",
    r"ld50",
    r"pfu",
    r"cfu",
    r"mbq",
    r"lsu",
    r"lf",
    r"au",
    r"bau",
    r"kiu",
    r"mu",
    r"iu",
    r"meq",
    r"mmol",
    r"microgram(?:s)?",
    r"ug",
    r"milligram(?:s)?",
    r"gram(?:s)?",
    r"mcg",
    r"mg",
    r"g",
    r"mcl",
    r"mls?",
    r"l",
    r"w",
    r"unit(?:s)?",
    r"unt",
]
DEN_UNIT_PATTERNS = [
    r"unit\s+dose(?:s)?",
    r"dropper\s+bottle(?:s)?",
    r"prefilled\s+pen(?:s)?",
    r"prefilled\s+syr(?:inge)?(?:s)?",
    r"sq\.?\s*cm\.?",
    r"d\s*antigen(?:\s+unit)?(?:s)?",
    r"actuation(?:s)?",
    r"actuat",
    r"puff(?:s)?",
    r"spray(?:s)?",
    r"drop(?:s)?",
    r"dose(?:s)?",
    r"tablet(?:s)?",
    r"capsule(?:s)?",
    r"lozenge(?:s)?",
    r"patch(?:es)?",
    r"implant(?:s)?",
    r"bead(?:s)?",
    r"wafer(?:s)?",
    r"suppositor(?:y|ies)",
    r"ampoule(?:s)?",
    r"applicator(?:s)?",
    r"bag(?:s)?",
    r"bar(?:s)?",
    r"bottle(?:s)?",
    r"box(?:es)?",
    r"bucket(?:s)?",
    r"can(?:s)?",
    r"canister(?:s)?",
    r"cartridge(?:s)?",
    r"gallon(?:s)?",
    r"gum",
    r"inhalation(?:s)?",
    r"inhaler(?:s)?",
    r"jar(?:s)?",
    r"measure(?:s)?",
    r"pack(?:s)?",
    r"pen(?:s)?",
    r"piece(?:s)?",
    r"sachet(?:s)?",
    r"set(?:s)?",
    r"sheet(?:s)?",
    r"syringe(?:s)?",
    r"tube(?:s)?",
    r"vial(?:s)?",
    r"gallon(?:s)?",
    r"kg",
    r"g",
    r"gram(?:s)?",
    r"mg",
    r"mcg",
    r"mls?",
    r"mcl",
    r"l",
    r"iu",
    r"meq",
    r"mmol",
    r"mu",
    r"au",
    r"bau",
    r"unit(?:s)?",
    r"unt",
    r"hr|hour|hours|h|day|days",
]
NUM_UNIT_PATTERN = "(?:" + "|".join(NUM_UNIT_PATTERNS) + ")"
DEN_UNIT_PATTERN = "(?:" + "|".join(DEN_UNIT_PATTERNS) + ")"
NUMBER_PATTERN = r"\d+(?:[ ,]\d{3})*(?:\.\d+)?"
COMBO_UNIT_PATTERN = "(?:mg|mcg|ug|g|iu|meq|mmol|au|bau|kiu|mu)"

RATIO = re.compile(
    rf"""
    (?P<a>{NUMBER_PATTERN})\s*(?P<u>{NUM_UNIT_PATTERN})\s*/\s*
    (?:(?P<b>{NUMBER_PATTERN})\s*)?(?P<per>{DEN_UNIT_PATTERN})(?![A-Za-z])
    """,
    re.IGNORECASE | re.VERBOSE,
)
PER = re.compile(
    rf"""
    (?P<a>{NUMBER_PATTERN})\s*(?P<u>{NUM_UNIT_PATTERN})\s*(?:per|in)\s*
    (?:(?P<b>{NUMBER_PATTERN})\s*)?(?P<per>{DEN_UNIT_PATTERN})(?![A-Za-z])
    """,
    re.IGNORECASE | re.VERBOSE,
)
EXPLICIT_COMBO = re.compile(
    rf"""
    (?P<a>{NUMBER_PATTERN})\s*(?P<u>{COMBO_UNIT_PATTERN})\s*
    (?:/|\+|&|with)\s*
    (?P<b>{NUMBER_PATTERN})\s*(?P<u2>{COMBO_UNIT_PATTERN})
    """,
    re.IGNORECASE | re.VERBOSE,
)
HYPHEN_COMBO = re.compile(
    rf"(?P<a>{NUMBER_PATTERN})\s*[-–]\s*(?P<b>{NUMBER_PATTERN})\s*(?P<u>{COMBO_UNIT_PATTERN})\b",
    re.IGNORECASE,
)
SINGLE = re.compile(
    rf"(?P<a>{NUMBER_PATTERN})\s*(?P<u>{NUM_UNIT_PATTERN})(?![A-Za-z])",
    re.IGNORECASE | re.VERBOSE,
)
COUNT = re.compile(
    rf"(?P<a>{NUMBER_PATTERN})\s*(?P<u>{DEN_UNIT_PATTERN})(?![A-Za-z])",
    re.IGNORECASE | re.VERBOSE,
)
PERCENT = re.compile(
    r"""
    (?P<a>""" + NUMBER_PATTERN + r""")\s*(?:%|percent|pct)
    (?:\s*(?P<qual>\(?\s*(?:w\s*/\s*v|w\s*/\s*w|v\s*/\s*v|wv|ww|vv)\s*\)?) )?
    """,
    re.IGNORECASE | re.VERBOSE,
)
STOP_WORDS = {"and", "with", "of", "the", "to", "for", "in", "by", "per"}
FORM_TOKEN_MAP = {
    "aerosol": "spray",
    "aerosols": "spray",
    "cap": "capsule",
    "caps": "capsule",
    "capsula": "capsule",
    "capsulas": "capsule",
    "capsule": "capsule",
    "capsules": "capsule",
    "comprime": "tablet",
    "comprimido": "tablet",
    "comprimidos": "tablet",
    "compressa": "tablet",
    "compresse": "tablet",
    "crema": "cream",
    "creme": "cream",
    "cream": "cream",
    "drop": "solution",
    "drops": "solution",
    "gel": "gel",
    "gota": "solution",
    "gotas": "solution",
    "goutte": "solution",
    "gouttes": "solution",
    "implante": "implant",
    "implantes": "implant",
    "implant": "implant",
    "implants": "implant",
    "inalacao": "inhalation",
    "inhalacao": "inhalation",
    "inhalacion": "inhalation",
    "inhalation": "inhalation",
    "inhaler": "inhalation",
    "inhal": "inhalation",
    "inj": "inject",
    "injeccion": "inject",
    "injecao": "inject",
    "injetavel": "inject",
    "inject": "inject",
    "injection": "inject",
    "injectable": "inject",
    "injectables": "inject",
    "injt": "inject",
    "inyeccion": "inject",
    "inyectable": "inject",
    "loz": "lozenge",
    "lozenge": "lozenge",
    "oral": "oral",
    "orale": "oral",
    "orales": "oral",
    "patch": "patch",
    "patches": "patch",
    "pastilla": "lozenge",
    "pastillas": "lozenge",
    "pastille": "lozenge",
    "plaster": "patch",
    "plasters": "patch",
    "pomada": "ointment",
    "pomata": "ointment",
    "pommade": "ointment",
    "powder": "powder",
    "poudre": "powder",
    "pwdr": "powder",
    "sirup": "syrup",
    "sirop": "syrup",
    "soln": "solution",
    "oph": "ophthalmic",
    "solucion": "solution",
    "solucao": "solution",
    "solution": "solution",
    "soluzione": "solution",
    "spray": "spray",
    "sprays": "spray",
    "nasal": "nasal",
    "susp": "suspension",
    "suspension": "suspension",
    "suspensao": "suspension",
    "syrup": "syrup",
    "tableta": "tablet",
    "tabletas": "tablet",
    "tablet": "tablet",
    "tablets": "tablet",
    "tablette": "tablet",
    "tabletten": "tablet",
    "tab": "tablet",
    "tabs": "tablet",
    "topical": "topical",
    "topico": "topical",
    "transdermal": "patch",
    "troche": "lozenge",
    "unguento": "ointment",
    "unguent": "ointment",
    "ointment": "ointment",
    "jarabe": "syrup",
    "xarope": "syrup",
    "polvo": "powder",
    "pulver": "powder",
    "granule": "granule",
    "granules": "granule",
    "granulo": "granule",
    "granulos": "granule",
    "supp": "suppository",
    "suppositories": "suppository",
    "suppository": "suppository",
    "suppositoire": "suppository",
    "supositorio": "suppository",
    "supositorios": "suppository",
    "infusion": "infusion",
    "infusao": "infusion",
    "irrigation": "irrigation",
    "inj soln": "injectable solution",
    "inj solution": "injectable solution",
    "injectable solution": "injectable solution",
    "solution for injection": "injectable solution",
}

# -----------------------------
# Release + dosage form canonicalization (EU/international -> US RxNorm style)
# -----------------------------

RELEASE_CANONICAL: Dict[str, dict] = {
    "immediate_release": {
        "abbr": ["IR"],
        "aliases": ["immediate release", "immediate-release", "conventional release"],
        "notes": "Releases promptly after administration.",
        "us_equiv": "IR",
        "us_label": "immediate-release",
    },
    "delayed_release": {
        "abbr": ["DR"],
        "aliases": ["delayed release", "delayed-release"],
        "notes": "Release is delayed; mechanism may be pH- or time-dependent.",
        "us_equiv": "DR",
        "us_label": "delayed-release",
    },
    "gastro_resistant": {
        "abbr": ["GR", "EC"],
        "aliases": [
            "gastro-resistant",
            "gastro resistant",
            "enteric-coated",
            "enteric coated",
            "acid-resistant",
            "acid resistant",
        ],
        "notes": "Enteric/acid-resistant; subtype of delayed-release (typically intestinal release).",
        "us_equiv": "DR",
        "us_label": "delayed-release (enteric-coated)",
    },
    "extended_release": {
        "abbr": ["ER", "XR", "XL", "SR", "CR", "PR", "LA", "MR"],
        "aliases": [
            "extended release",
            "extended-release",
            "sustained release",
            "sustained-release",
            "controlled release",
            "controlled-release",
            "long acting",
            "long-acting",
            "prolonged release",
            "prolonged-release",
            "modified release",
            "modified-release",
        ],
        "notes": "Prolonged release over time; many shorthands map here.",
        "us_equiv": "ER",
        "us_label": "extended-release",
    },
}

ABBR_TO_RELEASE: Dict[str, str] = {
    abbr.upper(): canonical
    for canonical, meta in RELEASE_CANONICAL.items()
    for abbr in meta.get("abbr", [])
}


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[\(\)\[\]\{\},;:/\\]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


ALIAS_TO_RELEASE: Dict[str, str] = {}
for canon, meta in RELEASE_CANONICAL.items():
    for a in meta.get("aliases", []):
        ALIAS_TO_RELEASE[_norm(a)] = canon

_ABBR_RE = re.compile(r"\b(IR|DR|ER|XR|XL|SR|CR|PR|LA|MR|EC|GR)\b", re.IGNORECASE)
_BRACKET_CONTENT_RE = re.compile(r"\[[^\[\]]+\]")


def _strip_bracketed(text: str) -> str:
    return _BRACKET_CONTENT_RE.sub(" ", text or "")


def _bracket_spans(text: str) -> List[tuple[int, int]]:
    if not text:
        return []
    return [m.span() for m in _BRACKET_CONTENT_RE.finditer(text)]


def normalize_release(text: str, *, ignore_bracketed: bool = False) -> Optional[str]:
    """Return canonical release key."""

    raw = text or ""
    if ignore_bracketed:
        raw = _strip_bracketed(raw)
    t = _norm(raw)
    abbrs = [m.group(1).upper() for m in _ABBR_RE.finditer(raw)]
    for ab in abbrs:
        canon = ABBR_TO_RELEASE.get(ab)
        if canon == "gastro_resistant":
            return canon
    for alias, canon in ALIAS_TO_RELEASE.items():
        if alias in t:
            return canon
    if abbrs:
        if any(a in {"EC", "GR"} for a in abbrs):
            return "gastro_resistant"
        if any(a in {"DR"} for a in abbrs):
            return "delayed_release"
        if any(a in {"ER", "XR", "XL", "SR", "CR", "PR", "LA", "MR"} for a in abbrs):
            return "extended_release"
        if any(a in {"IR"} for a in abbrs):
            return "immediate_release"
    return None


DOSAGE_FORMS_CANONICAL: Dict[str, dict] = {
    "oral_tablet": {
        "route": "oral",
        "form": "tablet",
        "aliases": ["tablet", "tab", "oral tablet"],
        "us_label": "tablet",
    },
    "oral_tablet_film_coated": {
        "route": "oral",
        "form": "tablet",
        "aliases": ["film-coated tablet", "film coated tablet", "fct"],
        "us_label": "film-coated tablet",
    },
    "oral_tablet_orodispersible": {
        "route": "oral",
        "form": "tablet",
        "aliases": ["orodispersible tablet", "odt", "mouth-dissolving tablet"],
        "us_label": "orally disintegrating tablet",
    },
    "oral_tablet_chewable": {
        "route": "oral",
        "form": "tablet",
        "aliases": ["chewable tablet", "chew tab"],
        "us_label": "chewable tablet",
    },
    "oral_capsule": {
        "route": "oral",
        "form": "capsule",
        "aliases": ["capsule", "cap", "oral capsule", "hard capsule", "hard gelatin capsule"],
        "us_label": "capsule",
    },
    "oral_capsule_softgel": {
        "route": "oral",
        "form": "capsule",
        "aliases": ["soft capsule", "softgel", "softgel capsule", "soft capsule, liquid-filled"],
        "us_label": "capsule, liquid-filled",
    },
    "oral_solution": {
        "route": "oral",
        "form": "solution",
        "aliases": ["oral solution", "oral drop", "oral drops", "oral soln"],
        "us_label": "oral solution",
    },
    "nasal_spray": {
        "route": "nasal",
        "form": "spray",
        "aliases": ["nasal spray", "nasal aerosol"],
        "us_label": "nasal spray",
    },
    "nasal_solution": {
        "route": "nasal",
        "form": "solution",
        "aliases": ["nasal solution", "nasal drops", "nasal drop"],
        "us_label": "nasal solution",
    },
    "injectable_solution": {
        "route": "injectable",
        "form": "solution",
        "aliases": [
            "injectable solution",
            "inj",
            "inject",
            "injection",
            "injection solution",
            "solution for injection",
            "solution for injection/infusion",
            "infusion",
        ],
        "us_label": "injectable solution",
    },
    "ophthalmic_solution": {
        "route": "ophthalmic",
        "form": "solution",
        "aliases": [
            "ophthalmic solution",
            "ophthalmic drops",
            "ophthalmic drop",
            "eye solution",
            "eye drops",
            "eye drop",
            "oph soln",
            "soln oph",
            "oph solution",
            "oph drops",
            "oph drop",
            "ophthalmic soln",
        ],
        "us_label": "ophthalmic solution",
    },
    "otic_solution": {
        "route": "otic",
        "form": "solution",
        "aliases": [
            "otic solution",
            "otic drops",
            "otic drop",
            "ear solution",
            "ear drops",
            "ear drop",
            "otic soln",
            "soln otic",
            "ear soln",
        ],
        "us_label": "otic solution",
    },
    "solution_unspecified": {
        "route": None,
        "form": "solution",
        "aliases": ["solution", "drops", "drop"],
        "us_label": "solution",
    },
    "oral_suspension": {
        "route": "oral",
        "form": "suspension",
        "aliases": ["oral suspension", "suspension", "susp", "oral susp"],
        "us_label": "oral suspension",
    },
    "oral_powder": {
        "route": "oral",
        "form": "powder",
        "aliases": ["oral powder", "powder for oral solution", "powder for oral suspension"],
        "us_label": "oral powder",
    },
    "powder_unspecified": {
        "route": None,
        "form": "powder",
        "aliases": ["powder"],
        "us_label": "powder",
    },
    "topical_powder": {
        "route": "topical",
        "form": "powder",
        "aliases": [
            "topical powder",
            "cutaneous powder",
            "dermal powder",
            "dusting powder",
            "powder for topical",
            "powder for cutaneous",
            "powder for external use",
        ],
        "us_label": "topical powder",
    },
}

ALIAS_TO_DOSAGE_FORM: Dict[str, str] = {}
for canon, meta in DOSAGE_FORMS_CANONICAL.items():
    for a in meta.get("aliases", []):
        ALIAS_TO_DOSAGE_FORM[_norm(a)] = canon
ALIAS_TOKEN_SETS: Dict[str, set[str]] = {
    alias: {t for t in alias.split(" ") if t} for alias in ALIAS_TO_DOSAGE_FORM
}


def normalize_dosage_form(text: str, *, ignore_bracketed: bool = False) -> Optional[str]:
    raw = text or ""
    if ignore_bracketed:
        raw = _strip_bracketed(raw)
    t = _norm(raw)
    tokens = {tok for tok in t.split(" ") if tok}
    for alias, canon in sorted(ALIAS_TO_DOSAGE_FORM.items(), key=lambda kv: -len(kv[0])):
        if alias in t:
            return canon
        alias_tokens = ALIAS_TOKEN_SETS.get(alias)
        if alias_tokens and alias_tokens.issubset(tokens):
            return canon
    return None


def eu_to_us_style(text: str, *, ignore_bracketed: bool = False) -> dict:
    raw = text or ""
    if ignore_bracketed:
        raw = _strip_bracketed(raw)
    release_canon = normalize_release(raw)
    form_canon = normalize_dosage_form(raw)

    us_release = RELEASE_CANONICAL.get(release_canon, {}).get("us_equiv")
    us_release_label = RELEASE_CANONICAL.get(release_canon, {}).get("us_label")
    us_form_label = DOSAGE_FORMS_CANONICAL.get(form_canon, {}).get("us_label")

    combined = None
    if us_release_label and us_form_label:
        combined = f"{us_release_label} {us_form_label}"
    elif us_form_label:
        combined = us_form_label
    elif us_release_label:
        combined = us_release_label

    enteric = (release_canon == "gastro_resistant") or ("enteric" in _norm(raw)) or ("gastro resistant" in _norm(raw))
    modified = release_canon in {"delayed_release", "gastro_resistant", "extended_release"}

    return {
        "input": raw,
        "canonical_dosage_form": form_canon,
        "canonical_release": release_canon,
        "us_release_abbr": us_release,
        "us_combined_label": combined,
        "flags": {
            "enteric_or_gastro_resistant": bool(enteric),
            "modified_release": bool(modified),
        },
    }


def _release_equiv(canon: str | None) -> str | None:
    if not canon:
        return None
    return RELEASE_CANONICAL.get(canon, {}).get("us_equiv") or canon


def release_score(query: str, candidate: str) -> float:
    """Return +0.4 on release match, -0.4 on mismatch, -0.2 if candidate has release but query doesn't."""

    q_release = normalize_release(query)
    c_release = normalize_release(candidate, ignore_bracketed=True)
    q_info = eu_to_us_style(query)
    c_info = eu_to_us_style(candidate, ignore_bracketed=True)
    q_modified = bool(q_info.get("flags", {}).get("modified_release"))
    c_modified = bool(c_info.get("flags", {}).get("modified_release"))
    if not q_release and not c_release:
        if c_modified and not q_modified:
            return -0.2
        return 0.0
    if not q_release and c_release:
        return -0.2
    if q_release and not c_release:
        return -0.4
    q_equiv = _release_equiv(q_release)
    c_equiv = _release_equiv(c_release)
    if q_equiv and c_equiv and q_equiv == c_equiv:
        return 0.4
    return -0.4


def release_form_score(query: str, candidate: str) -> float:
    """Return -1.0 on release/form mismatch, 0.0 otherwise."""

    q_release = normalize_release(query)
    c_release = normalize_release(candidate, ignore_bracketed=True)
    q_form = normalize_dosage_form(query)
    c_form = normalize_dosage_form(candidate, ignore_bracketed=True)

    if q_release and c_release and q_release != c_release:
        return -1.0
    if q_form and c_form and q_form != c_form:
        return -1.0
    return 0.0


def form_route_score(query: str, candidate: str) -> float:
    """Return -1.0 on dosage form/route mismatch, +1.0 on match, 0.0 if unknown."""

    q_form = normalize_dosage_form(query)
    c_form = normalize_dosage_form(candidate, ignore_bracketed=True)
    if q_form == "powder_unspecified" or c_form == "powder_unspecified":
        return 0.5
    if not q_form and c_form:
        return -0.5
    if q_form and c_form:
        if q_form == c_form:
            return 1.0
        q_route = DOSAGE_FORMS_CANONICAL.get(q_form, {}).get("route")
        c_route = DOSAGE_FORMS_CANONICAL.get(c_form, {}).get("route")
        if q_route and c_route and q_route == c_route:
            return 0.5
        return -1.0
    return 0.0
BRAND_IN_BRACKETS = re.compile(r"\[([^\[\]]+)\]")
UNITLESS_CHAIN = re.compile(
    r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)(?:\s*(?:/|\+|&|and)\s*(\d+(?:\.\d+)?))+",
    re.IGNORECASE,
)


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


def _normalize_unit_text(unit: str) -> str:
    u = unit.strip().lower()
    u = u.replace("µ", "u").replace("μ", "u")
    u = re.sub(r"\s+", " ", u)
    u = u.replace("-", " ")
    u = u.replace(".", "")
    return u


def _singularize_unit_text(unit: str) -> str:
    parts = unit.split(" ")
    if not parts:
        return unit
    last = parts[-1]
    if last.endswith("ies") and len(last) > 3:
        last = last[:-3] + "y"
    elif last.endswith("es") and last[:-2].endswith(("s", "x", "z", "ch", "sh")):
        last = last[:-2]
    elif last.endswith("s") and not last.endswith("ss"):
        last = last[:-1]
    parts[-1] = last
    return " ".join(parts)


_MASS_UNITS = {
    "mg": (1.0, "mg"),
    "milligram": (1.0, "mg"),
    "g": (1000.0, "mg"),
    "gram": (1000.0, "mg"),
    "kg": (1_000_000.0, "mg"),
    "mcg": (0.001, "mg"),
    "ug": (0.001, "mg"),
    "microgram": (0.001, "mg"),
}
_VOLUME_UNITS = {
    "ml": (1.0, "ml"),
    "mcl": (0.001, "ml"),
    "l": (1000.0, "ml"),
    "liter": (1000.0, "ml"),
    "litre": (1000.0, "ml"),
    "gallon": (3785.411784, "ml"),
}
_IU_UNITS = {
    "iu": (1.0, "iu"),
    "kiu": (1000.0, "iu"),
    "mu": (1_000_000.0, "iu"),
}
_OTHER_UNITS = {
    "meq": (1.0, "meq"),
    "mmol": (1.0, "mmol"),
    "au": (1.0, "au"),
    "bau": (1.0, "bau"),
    "pfu": (1.0, "pfu"),
    "cfu": (1.0, "cfu"),
    "mbq": (1.0, "mbq"),
    "lsu": (1.0, "lsu"),
    "lf": (1.0, "lf"),
    "ld50": (1.0, "ld50"),
    "ccid50": (1.0, "ccid50"),
    "tcid50": (1.0, "tcid50"),
    "lgccid50": (1.0, "lgccid50"),
    "log10ccid50": (1.0, "log10ccid50"),
    "sq hdm": (1.0, "sq-hdm"),
    "w": (1.0, "w"),
    "anti xa unit": (1.0, "anti-xa unit"),
    "d antigen": (1.0, "d antigen"),
    "d antigen unit": (1.0, "d antigen"),
    "elisa u": (1.0, "elisa u"),
    "vaccinating dose": (1.0, "vaccinating dose"),
    "billion organism": (1.0, "billion organism"),
    "billion spore": (1.0, "billion spore"),
    "unit": (1.0, "unit"),
    "unt": (1.0, "unit"),
}
_DENOM_UNITS = {
    "actuation": (1.0, "actuation"),
    "actuat": (1.0, "actuation"),
    "puff": (1.0, "actuation"),
    "spray": (1.0, "actuation"),
    "drop": (1.0, "drop"),
    "dose": (1.0, "dose"),
    "tablet": (1.0, "tablet"),
    "capsule": (1.0, "capsule"),
    "lozenge": (1.0, "lozenge"),
    "patch": (1.0, "patch"),
    "implant": (1.0, "implant"),
    "bead": (1.0, "bead"),
    "wafer": (1.0, "wafer"),
    "suppository": (1.0, "suppository"),
    "ampoule": (1.0, "ampoule"),
    "applicator": (1.0, "applicator"),
    "bag": (1.0, "bag"),
    "bar": (1.0, "bar"),
    "bottle": (1.0, "bottle"),
    "box": (1.0, "box"),
    "bucket": (1.0, "bucket"),
    "can": (1.0, "can"),
    "canister": (1.0, "canister"),
    "cartridge": (1.0, "cartridge"),
    "dropper bottle": (1.0, "dropper bottle"),
    "gum": (1.0, "gum"),
    "inhalation": (1.0, "inhalation"),
    "inhaler": (1.0, "inhaler"),
    "jar": (1.0, "jar"),
    "measure": (1.0, "measure"),
    "pack": (1.0, "pack"),
    "pen": (1.0, "pen"),
    "piece": (1.0, "piece"),
    "prefilled pen": (1.0, "prefilled pen"),
    "prefilled syr": (1.0, "prefilled syringe"),
    "prefilled syringe": (1.0, "prefilled syringe"),
    "sachet": (1.0, "sachet"),
    "set": (1.0, "set"),
    "sheet": (1.0, "sheet"),
    "syringe": (1.0, "syringe"),
    "tube": (1.0, "tube"),
    "unit dose": (1.0, "unit dose"),
    "vial": (1.0, "vial"),
    "sqcm": (1.0, "sq.cm"),
    "hr": (1.0, "hr"),
    "hour": (1.0, "hr"),
    "day": (1.0, "day"),
}
_BASE_UNIT_MAP = {**_MASS_UNITS, **_VOLUME_UNITS, **_IU_UNITS, **_OTHER_UNITS}
_NUM_UNIT_MAP = dict(_BASE_UNIT_MAP)
_DEN_UNIT_MAP = {**_BASE_UNIT_MAP, **_DENOM_UNITS}
_BASE_CANONICAL_UNITS = {canonical for _, canonical in _BASE_UNIT_MAP.values()}
_COUNT_UNITS = {
    canonical for _, canonical in _DENOM_UNITS.values() if canonical not in _BASE_CANONICAL_UNITS
}


def _canon_unit(amount: float, unit: str, unit_map: Dict[str, tuple[float, str]]) -> tuple[float, str]:
    norm = _normalize_unit_text(unit)
    norm = _singularize_unit_text(norm)
    if norm in unit_map:
        factor, canonical = unit_map[norm]
        return amount * factor, canonical
    return amount, norm


def _canon_num_unit(amount: float, unit: str) -> tuple[float, str]:
    return _canon_unit(amount, unit, _NUM_UNIT_MAP)


def _canon_den_unit(amount: float, unit: str) -> tuple[float, str]:
    return _canon_unit(amount, unit, _DEN_UNIT_MAP)


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
    if denom_unit in {"mg", "mcg", "g"}:
        if denom_unit == "mg":
            denom_grams = denom / 1000.0
        elif denom_unit == "mcg":
            denom_grams = denom / 1_000_000.0
        else:
            denom_grams = denom
        if denom_grams <= 0:
            return
        pct = (grams / denom_grams) * 100.0
    elif denom_unit == "ml":
        pct = (grams / denom) * 100.0
    else:
        return
    components.append(StrengthComponent("percent", pct))


def _normalize_percent_qualifier(raw: str | None) -> str | None:
    if not raw:
        return None
    q = raw.lower().strip()
    q = re.sub(r"[\s\(\)]+", "", q)
    q = q.replace("wv", "w/v").replace("ww", "w/w").replace("vv", "v/v")
    if q in {"w/v", "w/w", "v/v"}:
        return q
    return None


def _percent_to_ratio_components(value: float, qualifier: str | None) -> List[StrengthComponent]:
    if qualifier == "v/v":
        return [
            StrengthComponent("ratio", value, unit="ml", denom_value=100.0, denom_unit="ml")
        ]
    if qualifier == "w/w":
        num = value * 1000.0
        denom = 100.0 * 1000.0
        return [StrengthComponent("ratio", num, unit="mg", denom_value=denom, denom_unit="mg")]
    # Default to w/v (mass per 100 mL)
    num = value * 1000.0
    return [StrengthComponent("ratio", num, unit="mg", denom_value=100.0, denom_unit="ml")]


def extract_strengths_with_spans(
    text: str,
    *,
    ignore_bracketed: bool = False,
) -> tuple[List[StrengthComponent], List[tuple[int, int]]]:
    spans: List[tuple[int, int]] = []
    components: List[StrengthComponent] = []
    bracket_spans = _bracket_spans(text) if ignore_bracketed else []

    def _parse_number(raw: str | None) -> float:
        if raw is None:
            return 0.0
        return float(re.sub(r"[\s,]+", "", raw))

    def _overlaps_bracket(start: int, end: int) -> bool:
        return any(not (end <= s or start >= e) for s, e in bracket_spans)

    def free(start: int, end: int) -> bool:
        if _overlaps_bracket(start, end):
            return False
        return all(end <= s or start >= e for s, e in spans)

    def handle_ratio(match: re.Match[str]) -> None:
        if not free(match.start(), match.end()):
            return
        num, num_unit = _canon_num_unit(_parse_number(match.group("a")), match.group("u"))
        raw_den = match.group("b")
        den_value = _parse_number(raw_den) if raw_den is not None else 1.0
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
        a, ua = _canon_num_unit(_parse_number(m.group("a")), m.group("u"))
        b, ub = _canon_num_unit(_parse_number(m.group("b")), m.group("u2"))
        components.extend(
            [StrengthComponent("single", a, unit=ua), StrengthComponent("single", b, unit=ub)]
        )
        spans.append((m.start(), m.end()))

    for m in HYPHEN_COMBO.finditer(text):
        if not free(m.start(), m.end()):
            continue
        a, unit = _canon_num_unit(_parse_number(m.group("a")), m.group("u"))
        b, _ = _canon_num_unit(_parse_number(m.group("b")), m.group("u"))
        components.extend(
            [StrengthComponent("single", a, unit=unit), StrengthComponent("single", b, unit=unit)]
        )
        spans.append((m.start(), m.end()))

    for m in PERCENT.finditer(text):
        if not free(m.start(), m.end()):
            continue
        value = _parse_number(m.group("a"))
        qualifier = _normalize_percent_qualifier(m.group("qual"))
        components.append(StrengthComponent("percent", value))
        components.extend(_percent_to_ratio_components(value, qualifier))
        spans.append((m.start(), m.end()))

    for m in SINGLE.finditer(text):
        if not free(m.start(), m.end()):
            continue
        value, unit = _canon_num_unit(_parse_number(m.group("a")), m.group("u"))
        components.append(StrengthComponent("single", value, unit=unit))
        spans.append((m.start(), m.end()))

    for m in COUNT.finditer(text):
        if not free(m.start(), m.end()):
            continue
        value = _parse_number(m.group("a"))
        _, unit = _canon_den_unit(value, m.group("u"))
        if unit in _BASE_CANONICAL_UNITS:
            continue
        components.append(StrengthComponent("single", value, unit=unit))
        spans.append((m.start(), m.end()))

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
        if comp.kind == "percent":
            continue
        bucket = comp.bucket()
        if comp.kind == "ratio" and comp.unit == "mg" and comp.denom_unit in {"mg", "ml"}:
            bucket = ("mass_ratio", None)
            val = comp.normalized() * (1000.0 if comp.denom_unit == "mg" else 1.0)
        else:
            val = comp.normalized()
        buckets.setdefault(bucket, []).append(val)
    return buckets


def _bucket_weight(bucket: tuple[str | None, str | None]) -> float:
    unit, denom = bucket
    if unit == "mass_ratio":
        return 1.0
    if denom is not None:
        return 1.0
    if unit == "mass_single":
        return 0.8
    if unit == "ml":
        return 0.4
    if unit in _COUNT_UNITS:
        return 0.3
    if unit == "percent":
        return 0.8
    return 0.6


def _min_cost_assignment(cost: List[List[float]], dummy_cost: float) -> tuple[float, int]:
    """Bitmask-DP linear assignment for small matrices; returns (min_cost, real_matches)."""

    if not cost:
        return 0.0, 0
    m = len(cost)
    k = max(len(row) for row in cost) if m else 0
    n = max(m, k)
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


def _dose_gate_stats(
    q_buckets: Dict[tuple[str | None, str | None], List[float]],
    d_buckets: Dict[tuple[str | None, str | None], List[float]],
    *,
    tau: float = 0.6,
    kappa_extra: float = 0.7,
) -> Dict[str, float]:
    """Compute dose similarity stats for strength matching."""

    eps = 1e-12
    total_real = 0
    sum_cost_real = 0.0
    extras_total = 0
    extras_unmatched = 0
    total_q = 0.0
    total_d = 0.0

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
        weight = _bucket_weight(bucket)
        total_q += weight * len(q_vals)
        total_d += weight * len(d_vals)
        if not q_vals and d_vals:
            extras_unmatched += weight * len(d_vals)
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
            sum_cost_real += weight * max(real_cost_sum, 0.0)
            total_real += weight * real
            extras_total += weight * max(len(d_vals) - real, 0)
    if total_q > 0:
        extras_total += extras_unmatched
    if total_real <= 0:
        return {
            "s_dose": 0.0,
            "p_extra": 1.0,
            "mean_cost": float("inf"),
            "total_real": 0.0,
            "coverage_q": 0.0,
            "total_q": float(total_q),
            "total_d": float(total_d),
        }
    mean_cost = sum_cost_real / total_real
    s_dose = math.exp(-mean_cost)
    p_extra = math.exp(-kappa_extra * extras_total)
    if s_dose < tau:
        s_dose = 0.0
    coverage_q = total_real / max(total_q, 1e-9)
    return {
        "s_dose": float(s_dose),
        "p_extra": float(p_extra),
        "mean_cost": float(mean_cost),
        "total_real": float(total_real),
        "coverage_q": float(coverage_q),
        "total_q": float(total_q),
        "total_d": float(total_d),
    }


def _dose_gate_and_extra(
    q_buckets: Dict[tuple[str | None, str | None], List[float]],
    d_buckets: Dict[tuple[str | None, str | None], List[float]],
    *,
    tau: float = 0.6,
    kappa_extra: float = 0.7,
) -> tuple[float, float]:
    """Compute dose gate metric S_dose and extra-active penalty P_extra."""

    stats = _dose_gate_stats(q_buckets, d_buckets, tau=tau, kappa_extra=kappa_extra)
    return float(stats["s_dose"]), float(stats["p_extra"])


def _sim(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return 0.0
    return math.exp(-abs(math.log(a / b)))


_MIN_STRENGTH_COVERAGE = 0.3
_MISMATCH_SDose = 0.2


def _has_volume_ratio(components: Sequence[StrengthComponent]) -> bool:
    return any(c.kind == "ratio" and c.denom_unit == "ml" for c in components)


def _has_mass_ratio(components: Sequence[StrengthComponent]) -> bool:
    return any(c.kind == "ratio" and c.unit == "mg" and c.denom_unit == "mg" for c in components)


def _has_mass_single(components: Sequence[StrengthComponent]) -> bool:
    return any(c.kind == "single" and c.unit == "mg" for c in components)


def _soft_mass_ratio_to_volume(components: Sequence[StrengthComponent]) -> List[StrengthComponent]:
    """Convert mass-based denominators to volume (1 g == 1 mL) for ratio matching."""

    out: List[StrengthComponent] = []
    for comp in components:
        if comp.kind == "ratio" and comp.unit == "mg" and comp.denom_unit == "mg":
            denom = comp.denom_value if comp.denom_value else 1.0
            denom_ml = denom / 1000.0
            out.append(
                StrengthComponent(
                    "ratio",
                    comp.value,
                    unit=comp.unit,
                    denom_value=denom_ml,
                    denom_unit="ml",
                )
            )
        else:
            out.append(comp)
    return out


def _soft_mass_ratio_to_amount(
    ratio_side: Sequence[StrengthComponent],
    single_side: Sequence[StrengthComponent],
) -> List[StrengthComponent]:
    """Add numerator-only mass components for ratio terms to allow soft amount matching."""

    if not _has_mass_ratio(ratio_side) or not _has_mass_single(single_side):
        return list(ratio_side)
    out: List[StrengthComponent] = list(ratio_side)
    for comp in ratio_side:
        if comp.kind == "ratio" and comp.unit == "mg" and comp.denom_unit == "mg":
            out.append(StrengthComponent("single", comp.value, unit="mg"))
    return out


def strength_sim(query: str, candidate: str) -> float:
    q_comp, _ = extract_strengths_with_spans(query)
    d_comp, _ = extract_strengths_with_spans(candidate, ignore_bracketed=True)
    if not q_comp or not d_comp:
        return 0.0
    if _has_volume_ratio(q_comp) or _has_volume_ratio(d_comp):
        q_comp = _soft_mass_ratio_to_volume(q_comp)
        d_comp = _soft_mass_ratio_to_volume(d_comp)
    if _has_mass_ratio(q_comp) and _has_mass_single(d_comp) and not _has_mass_ratio(d_comp):
        q_comp = _soft_mass_ratio_to_amount(q_comp, d_comp)
    if _has_mass_ratio(d_comp) and _has_mass_single(q_comp) and not _has_mass_ratio(q_comp):
        d_comp = _soft_mass_ratio_to_amount(d_comp, q_comp)
    q_buckets = _unit_bucket(q_comp)
    d_buckets = _unit_bucket(d_comp)
    overlap = set(q_buckets.keys()) & set(d_buckets.keys())
    if not overlap:
        return -1.0
    stats = _dose_gate_stats(q_buckets, d_buckets, tau=0.0, kappa_extra=0.7)
    if stats["total_real"] <= 0:
        return -1.0
    coverage = stats["coverage_q"]
    if coverage < _MIN_STRENGTH_COVERAGE:
        return 0.0
    s_dose = stats["s_dose"]
    if s_dose < _MISMATCH_SDose:
        return -1.0
    score = s_dose * stats["p_extra"] * coverage
    return float(score)


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    return normalized.lower()


def _simple_tokenize(text: str) -> List[str]:
    normalized = _normalize_text(text)
    normalized = re.sub(r"\[[^\]]+\]", " ", normalized)
    normalized = re.sub(r"[\(\)\[\],;:]", " ", normalized)
    normalized = re.sub(r"[-_/]", " ", normalized)
    tokens = re.split(r"[^a-z0-9]+", normalized)
    out: List[str] = []
    for tok in tokens:
        if not tok or tok in STOP_WORDS or tok.isdigit():
            continue
        out.append(FORM_TOKEN_MAP.get(tok, tok))
    return out


def _release_form_tokens(text: str, *, ignore_bracketed: bool = False) -> List[str]:
    info = eu_to_us_style(text, ignore_bracketed=ignore_bracketed)
    tokens: List[str] = []
    abbr = info.get("us_release_abbr")
    if abbr:
        tokens.append(str(abbr).lower())
    combined = info.get("us_combined_label") or ""
    tokens.extend(_simple_tokenize(combined))
    return tokens


def _tokenize(text: str, *, ignore_bracketed: bool = False) -> List[str]:
    out = _simple_tokenize(text)
    out.extend(_release_form_tokens(text, ignore_bracketed=ignore_bracketed))
    return out


def _bigrams(tokens: Sequence[str]) -> set[str]:
    return {f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)} if len(tokens) > 1 else set()


def jaccard_remainder(query: str, candidate: str) -> float:
    _, q_spans = extract_strengths_with_spans(query)
    _, d_spans = extract_strengths_with_spans(candidate, ignore_bracketed=True)
    q_rem = strip_spans(query, q_spans)
    d_rem = strip_spans(candidate, d_spans)
    q_tokens = _tokenize(q_rem)
    d_tokens = _tokenize(d_rem, ignore_bracketed=True)
    set_q = set(q_tokens)
    set_d = set(d_tokens)
    union = set_q | set_d
    j1 = (len(set_q & set_d) / len(union)) if union else 0.0
    b_q = _bigrams(q_tokens)
    b_d = _bigrams(d_tokens)
    b_union = b_q | b_d
    j2 = (len(b_q & b_d) / len(b_union)) if b_union else 0.0
    return 0.7 * j1 + 0.3 * j2


def _split_alnum_tokens(text: str) -> List[str]:
    tokens: List[str] = []
    current: List[str] = []
    normalized = _normalize_text(text)
    for ch in normalized:
        if ch.isalnum():
            current.append(ch)
        else:
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))
    return tokens


def _compact_alnum(text: str) -> str:
    return "".join(_split_alnum_tokens(text))


def _extract_document_brands(candidate: str) -> List[str]:
    return [raw for raw in BRAND_IN_BRACKETS.findall(candidate) if raw.strip()]


def _brand_pattern(brand: str) -> str:
    raw = brand.strip()
    if not raw:
        return ""
    parts = [re.escape(p) for p in re.split(r"[\s-]+", raw) if p]
    if not parts:
        return ""
    body = r"(?:\s+|-)+".join(parts)
    return rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])"


def _compact_sequence_in_tokens(tokens: Sequence[str], target: str) -> bool:
    if not tokens or not target:
        return False
    target_len = len(target)
    for i in range(len(tokens)):
        concat = ""
        for j in range(i, len(tokens)):
            concat += tokens[j]
            if len(concat) > target_len:
                break
            if concat == target:
                return True
    return False


def brand_score(query: str, candidate: str) -> float:
    """Return +1.0 on brand match, -1.0 if candidate has brand but query does not."""

    candidate_brands = _extract_document_brands(candidate)
    if not candidate_brands:
        return 0.0

    for brand in candidate_brands:
        pattern = _brand_pattern(brand)
        if not pattern:
            continue
        if re.search(pattern, query, flags=re.IGNORECASE):
            return 1.0
    return -1.0


def simple_strength_plus_jaccard(
    query: str,
    candidate: str,
    *,
    w_strength: float = 0.6,
    w_jaccard: float = 0.4,
    w_brand_penalty: float = 0.5,
    w_form_penalty: float = 0.1,
) -> Dict[str, float]:
    q_comp, _ = extract_strengths_with_spans(query)
    d_comp, _ = extract_strengths_with_spans(candidate, ignore_bracketed=True)
    q_buckets = _unit_bucket(q_comp)
    d_buckets = _unit_bucket(d_comp)
    s_dose, p_extra = _dose_gate_and_extra(q_buckets, d_buckets, tau=0.6, kappa_extra=0.7)
    strength = strength_sim(query, candidate)
    jaccard = jaccard_remainder(query, candidate)
    brand = brand_score(query, candidate)
    release = release_score(query, candidate)
    form_rel = release_form_score(query, candidate)
    form_route = form_route_score(query, candidate)
    if s_dose <= 0.0:
        return {
            "strength_sim": float(strength),
            "jaccard_text": float(jaccard),
            "brand_score": float(brand),
            "release_score": float(release),
            "form_release_score": float(form_rel),
            "form_route_score": float(form_route),
            "post_score": 0.0,
            "simple_score": 0.0,
        }
    pos_weight = max(w_strength + w_jaccard, 1e-9)
    base = (max(s_dose * p_extra, 1e-12) ** w_strength) * (max(jaccard, 1e-12) ** w_jaccard)
    base = base ** (1.0 / pos_weight)
    brand_factor = math.exp(w_brand_penalty * brand)
    form_factor = math.exp(w_form_penalty * form_rel)
    release_factor = math.exp(w_form_penalty * release)
    post_score = base * brand_factor * form_factor * release_factor
    return {
        "strength_sim": float(strength),
        "jaccard_text": float(jaccard),
        "brand_score": float(brand),
        "release_score": float(release),
        "form_release_score": float(form_rel),
        "form_route_score": float(form_route),
        "post_score": float(post_score),
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
    w_brand_penalty: float = 0.5,
    w_form_penalty: float = 0.1,
    minmax_within_query: bool = False,
) -> Dict[str, List[float]]:
    strength_scores: List[float] = []
    jaccard_scores: List[float] = []
    brand_scores: List[float] = []
    release_scores: List[float] = []
    form_scores: List[float] = []
    form_route_scores: List[float] = []
    for candidate in candidates:
        features = simple_strength_plus_jaccard(
            query,
            candidate,
            w_strength=w_strength,
            w_jaccard=w_jaccard,
            w_brand_penalty=w_brand_penalty,
            w_form_penalty=w_form_penalty,
        )
        strength_scores.append(features["strength_sim"])
        jaccard_scores.append(features["jaccard_text"])
        brand_scores.append(features["brand_score"])
        release_scores.append(features["release_score"])
        form_scores.append(features["form_release_score"])
        form_route_scores.append(features["form_route_score"])

    if minmax_within_query:
        strength_norm = _minmax(strength_scores)
        jaccard_norm = _minmax(jaccard_scores)
    else:
        strength_norm = strength_scores
        jaccard_norm = jaccard_scores
    pos_weight = max(w_strength + w_jaccard, 1e-9)
    post = []
    for s, j, b, r, f in zip(strength_norm, jaccard_norm, brand_scores, release_scores, form_scores):
        gm = (max(s, 1e-12) ** w_strength) * (max(j, 1e-12) ** w_jaccard)
        gm = gm ** (1.0 / pos_weight)
        brand_factor = math.exp(w_brand_penalty * b)
        form_factor = math.exp(w_form_penalty * f)
        release_factor = math.exp(w_form_penalty * r)
        post.append(gm * brand_factor * form_factor * release_factor)
    return {
        "strength_sim": strength_scores,
        "jaccard_text": jaccard_scores,
        "brand_score": brand_scores,
        "release_score": release_scores,
        "form_release_score": form_scores,
        "form_route_score": form_route_scores,
        "post_score": post,
        "simple_score": post,
    }


__all__ = [
    "extract_strengths_with_spans",
    "strip_spans",
    "strength_sim",
    "jaccard_remainder",
    "brand_score",
    "normalize_release",
    "normalize_dosage_form",
    "eu_to_us_style",
    "release_score",
    "release_form_score",
    "form_route_score",
    "simple_strength_plus_jaccard",
    "batch_features",
]
