from __future__ import annotations


def normalize_text_value(text: str) -> str:
    """Lowercase and collapse all whitespace to single spaces.

    Safe for any input; non-string values are cast to string first.
    """
    s = str(text).lower()
    # split() collapses all whitespace; join with a single space
    return " ".join(s.split())

