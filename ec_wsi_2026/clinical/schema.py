"""Canonical column names, source-header patterns, and cohort constants.

The workbooks are hand-maintained, so headers drift: a column may gain a unit, a
parenthetical legend, or a translation. Rather than indexing by position (which
is *known* to differ between the two files) or by exact string (which breaks on
the next edit), every canonical column declares a list of regex patterns matched
against normalised headers. Resolution is logged so a silent mismatch is visible
in the audit report.
"""

from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------------------
# Cohorts
# --------------------------------------------------------------------------
COHORTS = ("definitive", "neoadjuvant")
SHEET_BY_COHORT = {"definitive": "Definitiv", "neoadjuvant": "Neoadjuvant"}
COHORT_PREFIX = {"definitive": "def", "neoadjuvant": "neo"}
EXPECTED_ROWS = {"definitive": 118, "neoadjuvant": 91}

# --------------------------------------------------------------------------
# Institutions
# --------------------------------------------------------------------------
INSTITUTIONS = ("TUM", "LMU", "Wuerzburg", "Erlangen")

# "Würzburg" decodes correctly today, but it is the one site name that can
# arrive mojibaked ("Wurzburg", "WÃ¼rzburg", "W?rzburg") if a future export
# mishandles the encoding. Match on an ASCII-folded key so every spelling lands
# on the same canonical value.
_INSTITUTION_ALIASES = {
    "tum": "TUM",
    "lmu": "LMU",
    "erlangen": "Erlangen",
    "erl": "Erlangen",
    "wurzburg": "Wuerzburg",
    "wuerzburg": "Wuerzburg",
    "wrzburg": "Wuerzburg",
    "wzburg": "Wuerzburg",
    "wue": "Wuerzburg",
}

# Slide folders use a short institution code: `tum_def_1`, `erl_neo_78`.
INSTITUTION_TO_SLIDE_CODE = {
    "TUM": "tum",
    "LMU": "lmu",
    "Erlangen": "erl",
    "Wuerzburg": "wue",
}
SLIDE_CODE_TO_INSTITUTION = {v: k for k, v in INSTITUTION_TO_SLIDE_CODE.items()}

# `Patient No` is assigned in contiguous per-institution blocks. This is a
# redundant encoding of `Institution`, which makes it a free cross-check on the
# encoding recovery above -- and a fallback if that column is ever unreadable.
# Ranges are inclusive.
PATIENT_NO_BLOCKS: dict[str, list[tuple[int, int, str]]] = {
    "definitive": [(1, 63, "TUM"), (64, 87, "LMU"), (88, 92, "Wuerzburg"), (93, 118, "Erlangen")],
    "neoadjuvant": [(1, 57, "TUM"), (58, 67, "LMU"), (68, 76, "Wuerzburg"), (77, 91, "Erlangen")],
}


def normalise_header(text: object) -> str:
    """Casefold, strip accents, and collapse non-alphanumerics to single spaces."""
    s = unicodedata.normalize("NFKD", str(text))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.casefold()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()


def canonical_institution(value: object) -> str | None:
    """Map a raw institution cell to a canonical name, tolerating mojibake."""
    if value is None:
        return None
    key = normalise_header(value).replace(" ", "")
    # Drop replacement chars and any residue from a failed cp1252/utf-8 decode.
    key = re.sub(r"[^a-z]", "", key)
    return _INSTITUTION_ALIASES.get(key)


def institution_from_patient_no(cohort: str, patient_no: int | None) -> str | None:
    """Recover the institution from the patient number block ranges."""
    if patient_no is None:
        return None
    for lo, hi, inst in PATIENT_NO_BLOCKS[cohort]:
        if lo <= patient_no <= hi:
            return inst
    return None


def patient_key(cohort: str, patient_no: int) -> str:
    """Stable de-identified key: ``def_096``, ``neo_058``."""
    return f"{COHORT_PREFIX[cohort]}_{int(patient_no):03d}"


# --------------------------------------------------------------------------
# Sentinels
# --------------------------------------------------------------------------
# Values that mean "no data" across every column type. Matched after stripping
# and casefolding. Note '0' is NOT here -- it is a real code in several columns.
NULL_SENTINELS = frozenset(
    {"", "-", "--", "/", "?", "na", "n a", "n.a.", "nan", "none", "keine", "unbekannt", "unknown"}
)

# --------------------------------------------------------------------------
# Column resolution
# --------------------------------------------------------------------------
# canonical name -> (patterns, kind). `kind` drives coercion in coercion.py.
#   id        pass-through string
#   date      free-text date, see coercion.coerce_date
#   float     decimal number
#   int       plain integer
#   code      small-integer categorical, meanings in references/code_legends.yaml
#   stage     integer with an optional substage letter ('1b') or '+' modifier
#   text      free text, preserved verbatim
COLUMN_KIND = {}
COLUMN_PATTERNS: dict[str, list[str]] = {}


def _col(name: str, kind: str, *patterns: str) -> None:
    COLUMN_PATTERNS[name] = list(patterns)
    COLUMN_KIND[name] = kind


# --- identity -------------------------------------------------------------
_col("patient_no", "int", r"^patient no$", r"^patient nr$", r"^pat no$")
_col("institution", "text", r"^institution$", r"^zentrum$", r"^site$")
_col("patient_id", "id", r"^patientid$", r"^patient id$")  # PHI: hospital MRN

# --- demographics ---------------------------------------------------------
_col("date_of_birth", "date", r"^date of birth$", r"geburtsdatum")  # PHI
_col("sex", "code", r"^sex\b", r"^geschlecht")
_col("age_at_crt_recorded", "float", r"^age at crt")

# --- baseline tumour ------------------------------------------------------
_col("localisation", "code", r"^localisation", r"^lokalisation")
_col("tumor_length_cm", "float", r"^tumor length", r"^tumorlange")
_col("ct_stage", "stage", r"^ct stage$")
_col("cn_stage", "stage", r"^cn stage$")
_col("uicc_stage", "stage", r"^uicc stage$")
_col("histologic_grade", "stage", r"^histologic grade$", r"^grading$")

# --- endoscopy ------------------------------------------------------------
_col("date_endoscopy", "date", r"^date endoscopy$")
_col("endo_availability", "code", r"^endo 0 none")

# --- treatment ------------------------------------------------------------
_col("date_start_crt", "date", r"^date start crt$")
_col("date_end_crt", "date", r"^date end crt$")
_col("date_op", "date", r"^date op$")
_col("cht_regimen", "code", r"^cht regimen")
_col("cht_other", "text", r"^cht other$")
_col("radiation_technique", "code", r"^radiation technique")
_col("total_dose_gy", "float", r"^total dosage")
_col("ed_gy", "float", r"^ed gy$")
_col("boost_gd_gy", "float", r"^boost gd")
_col("boost_ed_gy", "float", r"^boost ed")
_col("brachyboost_gy", "text", r"^brachyboost")  # free text: '2 x 6'
_col("additional_neoadj_cht", "code", r"^additional neoadjuvant cht")
_col("adj_systemic_therapy", "code", r"^adj systemic therapy")

# --- imaging availability -------------------------------------------------
_col("date_ct_post_crt", "date", r"^date ct post crt$", r"^date ct post crt pre surgery$")
_col("ct_post_crt_available", "code", r"^ct post crt available")
_col("date_ct_pre", "date", r"^ct pre crt$", r"^date diagnostic ct pre therapy$")

# --- pathology response (neoadjuvant only) --------------------------------
_col("ypt_stage", "stage", r"^ypt stage$")
_col("ypn_stage", "stage", r"^ypn stage$")
_col("pcr", "code", r"^pcr")
_col("mandard", "code", r"^tumorregression mandard")
_col("becker", "text", r"^tumorregression becker")  # '1a', '1b', '2', '3'

# --- outcomes -------------------------------------------------------------
_col("local_failure", "code", r"^local failure")
_col("date_local_failure", "date", r"^date local failure$")
_col("progress", "code", r"^progress")
_col("date_progress", "date", r"^date progress$")
_col("last_fu_date", "date", r"^last fu date$")
_col("last_date_alive", "date", r"^last date alive$")
_col("death", "code", r"^death")
_col("date_death", "date", r"^date death$")
_col("reason_of_death", "code", r"^reason of death")

# Columns present in the workbooks that are deliberately NOT resolved:
#   'Time staging to Start CRT (days)', 'Time End CRT to OP (days)',
#   'LFFS (days)', 'PFS (days)', 'OS (days)'  -- all 0% populated, derived instead
#   'FU TIME (days)'                          -- live DATEDIF formulas whose cached
#                                                values are stale zeros. Junk; derived.
IGNORED_SOURCE_PATTERNS = [
    r"^time staging to start crt",
    r"^time end crt to op",
    r"^lffs days$",
    r"^pfs days$",
    r"^os days$",
    r"^fu time days$",
]

# Columns that must resolve in both cohorts or the load is aborted.
REQUIRED_COLUMNS = (
    "patient_no",
    "institution",
    "date_start_crt",
    "progress",
    "death",
)

# Cohort-specific columns. Absence here is expected, not an error.
DEFINITIVE_ONLY = ("local_failure", "date_local_failure")
NEOADJUVANT_ONLY = ("date_op", "ypt_stage", "ypn_stage", "pcr", "mandard", "becker")

# Direct identifiers, dropped before anything is written to data/processed.
PHI_COLUMNS = ("patient_id", "date_of_birth")

# Columns whose values are small integer codes with externally-defined meanings.
CODE_COLUMNS = tuple(name for name, kind in COLUMN_KIND.items() if kind == "code")


def resolve_columns(headers: list[object]) -> tuple[dict[str, str], list[str]]:
    """Match workbook headers to canonical names.

    Returns ``(mapping, unmatched)`` where *mapping* is canonical -> source header
    and *unmatched* lists source headers that matched nothing and are not in the
    deliberately-ignored set. Both are reported by the audit so that a renamed or
    newly added column cannot pass unnoticed.
    """
    normalised = {h: normalise_header(h) for h in headers}
    mapping: dict[str, str] = {}
    claimed: set[object] = set()

    for canonical, patterns in COLUMN_PATTERNS.items():
        for pattern in patterns:
            hit = next(
                (h for h, n in normalised.items() if h not in claimed and re.search(pattern, n)),
                None,
            )
            if hit is not None:
                mapping[canonical] = hit
                claimed.add(hit)
                break

    ignored = [
        h
        for h, n in normalised.items()
        if any(re.search(p, n) for p in IGNORED_SOURCE_PATTERNS)
    ]
    unmatched = [h for h in headers if h not in claimed and h not in ignored]
    return mapping, unmatched
