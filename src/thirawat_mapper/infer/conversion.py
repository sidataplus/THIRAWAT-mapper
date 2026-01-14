"""INN/BAN → USAN conversion helpers for inference."""

from __future__ import annotations

import re
from typing import Mapping

DEFAULT_INN_TO_USAN: dict[str, str] = {
    "paracetamol": "acetaminophen",
    "salbutamol": "albuterol",
    "salbutamol sulfate": "albuterol sulfate",
    "adrenaline": "epinephrine",
    "noradrenaline": "norepinephrine",
    "aciclovir": "acyclovir",
    "cefalexin": "cephalexin",
    "cefradine": "cephradine",
    "ciclosporin": "cyclosporine",
    "glyceryl trinitrate": "nitroglycerin",
    "lignocaine": "lidocaine",
    "lignocaine hydrochloride": "lidocaine hydrochloride",
    "rifampicin": "rifampin",
    "glibenclamide": "glyburide",
    "isoprenaline": "isoproterenol",
    "isoprenaline hydrochloride": "isoproterenol hydrochloride",
    "orciprenaline": "metaproterenol",
    "orciprenaline sulfate": "metaproterenol sulfate",
    "pethidine": "meperidine",
    "pethidine hydrochloride": "meperidine hydrochloride",
    "thiamazole": "methimazole",
    "hydroxycarbamide": "hydroxyurea",
    "chlortalidone": "chlorthalidone",
    "colestyramine": "cholestyramine",
    "chlorphenamine": "chlorpheniramine",
    "ethinylestradiol": "ethinyl estradiol",
    "metamizole": "dipyrone",
    "amfetamine": "amphetamine",
    "glycopyrronium": "glycopyrrolate",
    "clomifene": "clomiphene",
    "frusemide": "furosemide",
    "thiopentone": "thiopental",
    "oestradiol": "estradiol",
    "oestrone": "estrone",
    "oestriol": "estriol",
    "suxamethonium": "succinylcholine",
    "suxamethonium chloride": "succinylcholine chloride",
    "phenobarbitone": "phenobarbital",
    "phenobarbitone sodium": "phenobarbital sodium",
    "meclozine": "meclizine",
    "dicycloverine": "dicyclomine",
    "thiomersal": "thimerosal",
    "sodium cromoglicate": "cromolyn sodium",
    "aluminium": "aluminum",
    "sulphate": "sulfate",
    "guaiphenesin": "guaifenesin",
    "amoxycillin": "amoxicillin",
    "mesalazine": "mesalamine",
    "phytomenadione": "phytonadione",
    "benzylpenicillin": "penicillin G",
    "dimeticone": "dimethicone",
    "oxetacaine": "oxethazaine",
    "mepyramine": "pyrilamine",
    "methylthioninium chloride": "methylene blue",
    "clopidogrel bisulphate": "clopidogrel bisulfate",
    "imatinib mesilate": "imatinib mesylate",
    "ketorolac trometamol": "ketorolac tromethamine",
    "sodium picosulphate": "sodium picosulfate",
    "beclometasone": "beclomethasone",
    "beclometasone dipropionate": "beclomethasone dipropionate",
}

MAPPER_EXTRA_INN_TO_USAN: dict[str, str] = {
    # Mapper-only alias; keep off by default for trainer parity.
    "glucose": "dextrose",
}


_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in DEFAULT_INN_TO_USAN) + r")\b",
    flags=re.IGNORECASE,
)


def convert_inn_ban_to_usan(text: str, mapping: Mapping[str, str] | None = None) -> str:
    """Return a copy of *text* with INN/BAN tokens replaced by their USAN equivalents."""

    if not text:
        return text
    if mapping is None:
        mapping = DEFAULT_INN_TO_USAN
    if text and not mapping:
        return text

    pattern = _PATTERN if mapping is DEFAULT_INN_TO_USAN else re.compile(
        r"\b(" + "|".join(re.escape(k) for k in mapping) + r")\b",
        flags=re.IGNORECASE,
    )

    def _replacer(match: re.Match[str]) -> str:
        key = match.group(0)
        return mapping.get(key.lower(), key)

    return pattern.sub(_replacer, text)


__all__ = ["convert_inn_ban_to_usan", "DEFAULT_INN_TO_USAN", "MAPPER_EXTRA_INN_TO_USAN"]
