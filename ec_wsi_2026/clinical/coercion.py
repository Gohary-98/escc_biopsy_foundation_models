"""Explicit, logged coercion of the all-string workbook values into real types.

Every cell is read as text upstream, because letting pandas infer types silently
destroys the substage letters (``ypT '1b'``), the node-positive marker
(``cN '+'``), the multi-site code (``Localisation '1+3'``) and the in-situ grade
(``'Tis'``). Coercion therefore happens here, once, and every value that fails is
recorded in a :class:`ParseLog` rather than quietly becoming NaN.

Dates are the hard case: they are free text, mostly ISO but with a German
day-first minority, and a number carry a trailing annotation such as
``21.06.22 (zusammen mit Koloskopie)`` or ``05.04.2024 (PET)``. Both are parsed
by matching a date *prefix* and keeping the remainder as a note -- the annotation
records imaging provenance and is worth preserving.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime as dt
import re

import pandas as pd

from ec_wsi_2026.clinical.schema import NULL_SENTINELS

# Anything outside this window is a typo, not a date.
MIN_DATE = pd.Timestamp("1900-01-01")
MIN_BIRTH_DATE = pd.Timestamp("1910-01-01")
MAX_BIRTH_DATE = pd.Timestamp("2010-12-31")

_ISO_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})")
_GERMAN_RE = re.compile(r"^(\d{1,2})[./](\d{1,2})[./](\d{2,4})")
# Excel serial dates, in case a future workbook is written by a different tool.
_SERIAL_RE = re.compile(r"^(\d{5})(?:\.0)?$")
_EXCEL_EPOCH = pd.Timestamp("1899-12-30")

_BRACHY_RE = re.compile(r"^(\d+)\s*[x*]\s*(\d+(?:[.,]\d+)?)$", re.IGNORECASE)
_STAGE_RE = re.compile(r"^(\d+)\s*([a-z]*)$", re.IGNORECASE)
_BECKER_RE = re.compile(r"^([123])\s*([ab]?)$", re.IGNORECASE)


@dataclass
class ParseLog:
    """Collects every value that could not be coerced, for the audit report."""

    failures: list[dict] = field(default_factory=list)

    def record(self, *, cohort: str, row: int, column: str, value: object, reason: str) -> None:
        self.failures.append(
            {
                "cohort": cohort,
                "source_row": row,
                "column": column,
                "value": "" if value is None else str(value),
                "reason": reason,
            }
        )

    def to_frame(self) -> pd.DataFrame:
        cols = ["cohort", "source_row", "column", "value", "reason"]
        if not self.failures:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(self.failures, columns=cols)

    def __len__(self) -> int:
        return len(self.failures)


def is_null(value: object) -> bool:
    """True for blanks and every sentinel spelling used across the workbooks."""
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if value is pd.NaT:
        return True
    return str(value).strip().casefold() in NULL_SENTINELS


def _clean(value: object) -> str:
    """Strip whitespace and any non-breaking spaces pasted in from Word/Excel."""
    return str(value).replace("\xa0", " ").strip()


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------
def coerce_date(value: object) -> tuple[pd.Timestamp | None, str | None, str | None]:
    """Parse a free-text date.

    Returns ``(timestamp, note, error)``. *note* is whatever followed the date --
    ``'(PET)'``, ``'(extern)'`` -- and is kept because it records how the imaging
    was acquired. *error* is a short reason string when parsing failed, so the
    caller can log it; a null input yields all-None with no error.
    """
    if is_null(value):
        return None, None, None

    text = _clean(value)

    # A datetime may already have survived as an object if a future workbook
    # stores real Excel dates rather than text.
    if isinstance(value, (dt.datetime, dt.date, pd.Timestamp)):
        ts = pd.Timestamp(value)
        return (ts, None, None) if _in_range(ts) else (None, None, "out of range")

    match = _ISO_RE.match(text)
    if match:
        year, month, day = (int(g) for g in match.groups())
        rest = text[match.end() :].strip()
        return _build(year, month, day, rest)

    match = _GERMAN_RE.match(text)
    if match:
        day, month, year = (int(g) for g in match.groups())
        if year < 100:
            # Two-digit years in this cohort are all 2000s: accrual runs 2010-2025.
            year += 2000
        rest = text[match.end() :].strip()
        return _build(year, month, day, rest)

    match = _SERIAL_RE.match(text)
    if match:
        ts = _EXCEL_EPOCH + pd.Timedelta(days=int(match.group(1)))
        return (ts, None, None) if _in_range(ts) else (None, None, "out of range")

    return None, None, "unrecognised date format"


def _build(
    year: int, month: int, day: int, rest: str
) -> tuple[pd.Timestamp | None, str | None, str | None]:
    try:
        ts = pd.Timestamp(year=year, month=month, day=day)
    except ValueError as exc:
        return None, None, f"invalid date: {exc}"
    if not _in_range(ts):
        return None, None, "out of range"
    note = rest.strip(" ()[]") or None
    return ts, note, None


def _in_range(ts: pd.Timestamp) -> bool:
    return MIN_DATE <= ts <= pd.Timestamp.today().normalize()


def coerce_birth_date(value: object) -> tuple[pd.Timestamp | None, str | None, str | None]:
    """As :func:`coerce_date`, with a birth-plausible range."""
    ts, note, err = coerce_date(value)
    if ts is not None and not (MIN_BIRTH_DATE <= ts <= MAX_BIRTH_DATE):
        return None, note, "birth date out of plausible range"
    return ts, note, err


# --------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------
def coerce_float(value: object) -> tuple[float | None, str | None]:
    """Parse a decimal, accepting the German comma decimal separator."""
    if is_null(value):
        return None, None
    text = _clean(value).replace(",", ".")
    # Tolerate a stray unit suffix such as '50.4 Gy'.
    match = re.match(r"^(-?\d+(?:\.\d+)?)", text)
    if not match:
        return None, "not numeric"
    return float(match.group(1)), None


def coerce_int(value: object) -> tuple[int | None, str | None]:
    """Parse an integer, tolerating a '.0' float rendering."""
    number, err = coerce_float(value)
    if number is None:
        return None, err
    if number != int(number):
        return None, "not an integer"
    return int(number), None


def coerce_code(value: object) -> tuple[int | None, str | None]:
    """Parse a small categorical code. Meanings live in code_legends.yaml."""
    return coerce_int(value)


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------
def coerce_stage(value: object) -> tuple[int | None, str | None, bool, str | None]:
    """Split a stage cell into ``(number, suffix, positive_marker, error)``.

    Handles ``'3'``, ``'1b'`` and ``'4b'`` (substage letters, which carry real
    prognostic meaning and must not be discarded), ``'+'`` (node-positive with an
    unspecified count, seen in ``cN``), and ``'Tis'`` (carcinoma in situ, which has
    no numeric stage and is flagged via the suffix).
    """
    if is_null(value):
        return None, None, False, None

    text = _clean(value)
    if text == "+":
        return None, None, True, None

    lowered = text.casefold()
    if lowered in {"tis", "is"}:
        return None, "is", False, None
    if lowered == "x":
        return None, "x", False, None

    match = _STAGE_RE.match(text)
    if match:
        number = int(match.group(1))
        suffix = (match.group(2) or "").casefold() or None
        return number, suffix, False, None

    return None, None, False, "unrecognised stage"


def coerce_becker(value: object) -> tuple[str | None, float | None, str | None]:
    """Parse a Becker tumour-regression grade.

    Returns ``(label, residual_tumour_upper_bound_pct, error)``. Becker grades run
    1a (0% residual), 1b (<10%), 2 (10-50%), 3 (>50%); the numeric column is the
    upper bound so the grade can be used ordinally without inventing a scale.
    """
    if is_null(value):
        return None, None, None
    match = _BECKER_RE.match(_clean(value))
    if not match:
        return None, None, "unrecognised Becker grade"
    label = f"{match.group(1)}{match.group(2).casefold()}"
    bounds = {"1a": 0.0, "1b": 10.0, "1": 10.0, "2": 50.0, "3": 100.0}
    return label, bounds.get(label), None


def coerce_brachyboost(value: object) -> tuple[int | None, float | None, float | None, str | None]:
    """Parse a brachytherapy boost written as ``'2 x 6'`` (fractions x dose).

    Returns ``(fractions, dose_per_fraction, total_dose, error)``. A bare number
    is treated as a total dose with unknown fractionation.
    """
    if is_null(value):
        return None, None, None, None
    text = _clean(value)
    match = _BRACHY_RE.match(text)
    if match:
        fractions = int(match.group(1))
        per_fraction = float(match.group(2).replace(",", "."))
        return fractions, per_fraction, fractions * per_fraction, None
    total, err = coerce_float(text)
    if total is None:
        return None, None, None, err or "unrecognised brachyboost"
    return None, None, total, None


def years_between(start: pd.Timestamp | None, end: pd.Timestamp | None) -> float | None:
    """Age in years, or None if either endpoint is missing."""
    if start is None or end is None or pd.isna(start) or pd.isna(end):
        return None
    return (end - start).days / 365.25
