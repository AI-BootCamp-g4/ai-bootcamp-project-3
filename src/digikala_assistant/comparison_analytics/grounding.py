"""Check that numbers and citations in an answer are backed by the data.

Used both inside the pipelines (to flag bad claims before the user sees
them) and by the evaluation (to report Grounding as a number).
"""

from __future__ import annotations

import re

from .data_prep import normalize_fa

_THOUSANDS = r"\d{1,3}(?:[,٬]\d{3})+(?:[.٫]\d+)?"
_NUMBER = re.compile(rf"{_THOUSANDS}|\d+(?:[.,/٫]\d+)?")


def numbers_in(text: str) -> list[float]:
    """Parse Persian and Latin digits with decimal and thousands separators."""
    text = normalize_fa(text)
    out = []
    for raw in _NUMBER.findall(text):
        if re.fullmatch(_THOUSANDS, raw):
            raw = raw.replace(",", "").replace("٬", "")
        raw = raw.replace("٫", ".").replace("/", ".").replace(",", ".")
        try:
            out.append(float(raw))
        except ValueError:
            pass
    return out


def collect_numbers(obj) -> set[float]:
    """Every number inside a facts structure (dicts, lists, DataFrames)."""
    found: set[float] = set()
    if obj is None or isinstance(obj, bool):
        return found
    if isinstance(obj, (int, float)):
        found.add(float(obj))
    elif isinstance(obj, str):
        found.update(numbers_in(obj))
    elif isinstance(obj, dict):
        for value in obj.values():
            found |= collect_numbers(value)
    elif isinstance(obj, (list, tuple, set)):
        for value in obj:
            found |= collect_numbers(value)
    elif hasattr(obj, "to_dict"):
        found |= collect_numbers(obj.to_dict(orient="records"))
    return found


def _supported(x: float, facts: set[float]) -> bool:
    for f in facts:
        tol = max(0.06, abs(f) * 0.006)  # rounding: 81.9 -> 82, 4.18 -> 4.2
        if abs(x - f) <= tol or abs(x - round(f)) < 1e-9:
            return True
        if f >= 1000 and abs(x * 1000 - f) <= f * 0.01:  # "1,810 هزار تومان"
            return True
    return False


def number_grounding(answer: str, facts, ignore_small: bool = True) -> dict:
    """Share of numbers in the answer that appear in the facts.

    Small integers (1..10) are ignored because they are usually counts in
    the sentence itself ("دو محصول", "۳ دلیل"), not data claims.
    """
    fact_numbers = collect_numbers(facts)
    claimed = [x for x in numbers_in(answer) if not (ignore_small and x.is_integer() and x <= 10)]
    unsupported = [x for x in claimed if not _supported(x, fact_numbers)]
    return {"numbers": len(claimed), "unsupported": unsupported,
            "number_precision": None if not claimed else round(1 - len(unsupported) / len(claimed), 3)}


def citation_check(cited: list, allowed: set) -> dict:
    """Citations must point to comments that were actually shown to the model."""
    bad = [c for c in cited if c not in allowed]
    return {"citations": len(cited), "invalid": bad,
            "citation_precision": None if not cited else round(1 - len(bad) / len(cited), 3)}
