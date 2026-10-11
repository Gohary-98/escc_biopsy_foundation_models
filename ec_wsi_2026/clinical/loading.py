"""Load and harmonise the two cohort workbooks into one typed long table.

Two structural quirks drive the design:

1. **The header is on the second row.** Row 1 is an institution colour legend, so
   every read uses ``header=1``.
2. **The two workbooks were written by different tools** -- ``definitive`` uses a
   shared string table, ``neoadjuvant`` uses inline strings. Both currently
   decode correctly ("Würzburg" arrives as U+00FC, verified), but non-ASCII
   German text in a hand-maintained multi-site workbook is exactly the thing that
   breaks on the next export. :func:`read_sheet` therefore falls back to
   re-encoding the XML parts in memory if a decode ever fails, and institution is
   independently cross-checked against the patient-number blocks in
   :func:`_resolve_institution`. That cross-check, not the fallback, is what
   guarantees the analysis cannot be corrupted by a mangled site name.

The two sheets have different column *orders* and different cohort-specific
extras, so they are unioned by canonical name, never by position.
"""

from __future__ import annotations

import io
from pathlib import Path
import re
import zipfile

from loguru import logger
import pandas as pd

from ec_wsi_2026.clinical import coercion as co
from ec_wsi_2026.clinical import schema as sch
from ec_wsi_2026.clinical.coercion import ParseLog
from ec_wsi_2026.config import CLINICAL_DEF_XLSX, CLINICAL_NEO_XLSX

WORKBOOK_BY_COHORT = {
    "definitive": CLINICAL_DEF_XLSX,
    "neoadjuvant": CLINICAL_NEO_XLSX,
}


# --------------------------------------------------------------------------
# Reading, with encoding repair
# --------------------------------------------------------------------------
def read_sheet(path: Path, sheet: str) -> pd.DataFrame:
    """Read one sheet as all-strings, repairing a mis-declared encoding if needed."""
    try:
        return pd.read_excel(path, sheet_name=sheet, header=1, dtype=str, engine="openpyxl")
    except (UnicodeDecodeError, zipfile.BadZipFile, ValueError) as exc:
        logger.warning(f"{path.name}: standard read failed ({type(exc).__name__}: {exc}); repairing")
        return pd.read_excel(
            _repair_workbook_encoding(path), sheet_name=sheet, header=1, dtype=str, engine="openpyxl"
        )


def _repair_workbook_encoding(path: Path) -> io.BytesIO:
    """Rewrite the xlsx in memory, re-encoding any non-UTF-8 XML part as UTF-8.

    Insurance against a future export that writes cp1252 bytes into a part whose
    declaration claims UTF-8 -- the failure mode that turns "Würzburg" into
    mojibake or makes the file unreadable outright. Parts that already decode as
    UTF-8, which is all of them today, are copied byte-for-byte.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename.endswith(".xml") or item.filename.endswith(".rels"):
                try:
                    data.decode("utf-8")
                except UnicodeDecodeError:
                    data = data.decode("cp1252", errors="replace").encode("utf-8")
            dst.writestr(item, data)
    buffer.seek(0)
    return buffer


# --------------------------------------------------------------------------
# Per-cohort load
# --------------------------------------------------------------------------
def load_cohort(cohort: str, path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Read one cohort workbook and project it onto canonical column names.

    Returns the raw all-string frame plus a resolution report naming which source
    header fed each canonical column and which headers matched nothing.
    """
    path = Path(path or WORKBOOK_BY_COHORT[cohort])
    sheet = sch.SHEET_BY_COHORT[cohort]
    raw = read_sheet(path, sheet)

    # Trailing all-blank rows are common in hand-edited sheets.
    raw = raw.dropna(how="all").reset_index(drop=True)

    mapping, unmatched = sch.resolve_columns(list(raw.columns))
    missing_required = [c for c in sch.REQUIRED_COLUMNS if c not in mapping]
    if missing_required:
        raise ValueError(
            f"{path.name} [{sheet}]: required column(s) did not resolve: {missing_required}. "
            f"Headers present: {list(raw.columns)}"
        )

    expected = sch.EXPECTED_ROWS.get(cohort)
    if expected is not None and len(raw) != expected:
        logger.warning(f"{cohort}: expected {expected} rows, read {len(raw)}")

    out = pd.DataFrame(index=raw.index)
    out["cohort"] = cohort
    out["source_file"] = path.name
    # +3: one for the legend row, one for the header row, one for 1-based Excel.
    out["source_row"] = raw.index + 3
    for canonical, source in mapping.items():
        out[canonical] = raw[source]

    report = {
        "cohort": cohort,
        "path": str(path),
        "sheet": sheet,
        "n_rows": len(raw),
        "resolved": mapping,
        "unmatched_headers": unmatched,
        "unresolved_canonical": sorted(set(sch.COLUMN_PATTERNS) - set(mapping)),
    }
    if unmatched:
        logger.warning(f"{cohort}: {len(unmatched)} source header(s) matched nothing: {unmatched}")
    logger.info(f"{cohort}: {len(raw)} rows, {len(mapping)} columns resolved")
    return out, report


def load_cohorts(paths: dict[str, Path] | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Load both workbooks and stack them, unioning columns by name."""
    paths = paths or {}
    frames, reports = [], []
    for cohort in sch.COHORTS:
        frame, report = load_cohort(cohort, paths.get(cohort))
        frames.append(frame)
        reports.append(report)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    logger.info(f"combined: {len(combined)} rows across {len(frames)} cohorts")
    return combined, reports


# --------------------------------------------------------------------------
# Coercion
# --------------------------------------------------------------------------
def coerce_cohorts(raw: pd.DataFrame) -> tuple[pd.DataFrame, ParseLog]:
    """Type the all-string long table, logging every value that fails.

    Date columns gain a companion ``{col}__note`` holding any trailing annotation.
    Stage columns are split into ``{col}_num`` / ``{col}_suffix``, plus
    ``cn_positive`` for the bare ``'+'`` marker.
    """
    log = ParseLog()
    out = pd.DataFrame(index=raw.index)
    for passthrough in ("cohort", "source_file", "source_row"):
        out[passthrough] = raw[passthrough]

    rows = raw.to_dict("records")

    for column, kind in sch.COLUMN_KIND.items():
        if column not in raw.columns:
            continue
        values = raw[column]

        if kind == "date":
            coercer = co.coerce_birth_date if column == "date_of_birth" else co.coerce_date
            parsed, notes = [], []
            for row, value in zip(rows, values):
                ts, note, err = coercer(value)
                if err:
                    log.record(
                        cohort=row["cohort"],
                        row=row["source_row"],
                        column=column,
                        value=value,
                        reason=err,
                    )
                parsed.append(ts)
                notes.append(note)
            out[column] = pd.to_datetime(pd.Series(parsed, index=raw.index))
            if any(n is not None for n in notes):
                out[f"{column}__note"] = notes

        elif kind == "stage":
            numbers, suffixes, positives = [], [], []
            for row, value in zip(rows, values):
                number, suffix, positive, err = co.coerce_stage(value)
                if err:
                    log.record(
                        cohort=row["cohort"],
                        row=row["source_row"],
                        column=column,
                        value=value,
                        reason=err,
                    )
                numbers.append(number)
                suffixes.append(suffix)
                positives.append(positive)
            out[f"{column}_num"] = pd.array(numbers, dtype="Int64")
            out[f"{column}_suffix"] = suffixes
            if any(positives):
                out[f"{column}_positive"] = positives

        elif kind in {"code", "int"}:
            parsed = []
            for row, value in zip(rows, values):
                number, err = co.coerce_int(value)
                if err:
                    log.record(
                        cohort=row["cohort"],
                        row=row["source_row"],
                        column=column,
                        value=value,
                        reason=err,
                    )
                parsed.append(number)
            out[column] = pd.array(parsed, dtype="Int64")

        elif kind == "float":
            parsed = []
            for row, value in zip(rows, values):
                number, err = co.coerce_float(value)
                if err:
                    log.record(
                        cohort=row["cohort"],
                        row=row["source_row"],
                        column=column,
                        value=value,
                        reason=err,
                    )
                parsed.append(number)
            out[column] = pd.array(parsed, dtype="Float64")

        else:  # id, text
            out[column] = values.map(lambda v: None if co.is_null(v) else str(v).strip())

    _add_derived_columns(out, raw, log)
    _resolve_institution(out, raw, log)

    if len(log):
        logger.warning(f"{len(log)} value(s) failed coercion; see the parse-failure log")
    return out, log


def _add_derived_columns(out: pd.DataFrame, raw: pd.DataFrame, log: ParseLog) -> None:
    """Structured parses that produce more than one column."""
    rows = raw.to_dict("records")

    if "becker" in raw.columns:
        labels, bounds = [], []
        for row, value in zip(rows, raw["becker"]):
            label, bound, err = co.coerce_becker(value)
            if err:
                log.record(
                    cohort=row["cohort"],
                    row=row["source_row"],
                    column="becker",
                    value=value,
                    reason=err,
                )
            labels.append(label)
            bounds.append(bound)
        out["becker_grade"] = labels
        out["becker_residual_max_pct"] = pd.array(bounds, dtype="Float64")

    if "brachyboost_gy" in raw.columns:
        fractions, per_fraction, totals = [], [], []
        for row, value in zip(rows, raw["brachyboost_gy"]):
            n, dose, total, err = co.coerce_brachyboost(value)
            if err:
                log.record(
                    cohort=row["cohort"],
                    row=row["source_row"],
                    column="brachyboost_gy",
                    value=value,
                    reason=err,
                )
            fractions.append(n)
            per_fraction.append(dose)
            totals.append(total)
        out["brachy_fractions"] = pd.array(fractions, dtype="Int64")
        out["brachy_dose_per_fraction_gy"] = pd.array(per_fraction, dtype="Float64")
        out["brachy_total_gy"] = pd.array(totals, dtype="Float64")

    # 'Localisation' carries '1+3' for a tumour spanning two segments. Keep the
    # primary segment for ordinal use and expose per-segment indicators.
    if "localisation" in raw.columns:
        segments = raw["localisation"].map(_parse_localisation)
        out["localisation_primary"] = pd.array(
            [s[0] if s else None for s in segments], dtype="Int64"
        )
        out["localisation_multisite"] = [bool(s and len(s) > 1) for s in segments]
        for segment in (1, 2, 3):
            out[f"localisation_seg{segment}"] = [bool(s and segment in s) for s in segments]

    # Nodal involvement, the modelling covariate. `cn_stage_num` is the parsed
    # count-bearing stage -- missing for exactly the nine patients recorded as
    # a bare '+', which `cn_stage_positive` flags. A bare '+' is read as
    # node-positive with the count unspecified (the plausible reading), so involvement reaches
    # full coverage where `cn_stage_num` alone cannot.
    if "cn_stage_num" in out.columns:
        plus = (
            out["cn_stage_positive"]
            if "cn_stage_positive" in out.columns
            else pd.Series(False, index=out.index)
        )
        involved = []
        for number, is_plus in zip(out["cn_stage_num"], plus):
            if bool(is_plus) or (not pd.isna(number) and int(number) > 0):
                involved.append(True)
            elif pd.isna(number):
                involved.append(None)
            else:
                involved.append(False)
        out["cn_involved"] = pd.array(involved, dtype="boolean")

    # Age: derived from the birth date where available, otherwise the recorded
    # value. The two sources are disjoint, not redundant -- the 14 patients with a
    # recorded age are exactly the 14 with no birth date (all of them Würzburg,
    # which evidently ships a pre-computed age instead of a DOB). So this is a
    # coalesce that reaches full coverage, not a cross-check.
    if {"date_of_birth", "date_start_crt"} <= set(out.columns):
        derived = [
            co.years_between(b, s) for b, s in zip(out["date_of_birth"], out["date_start_crt"])
        ]
        recorded = (
            out["age_at_crt_recorded"].tolist()
            if "age_at_crt_recorded" in out.columns
            else [None] * len(out)
        )
        ages, sources = [], []
        for d, r in zip(derived, recorded):
            if d is not None:
                ages.append(d)
                sources.append("derived_from_dob")
            elif r is not None and not pd.isna(r):
                ages.append(float(r))
                sources.append("recorded")
            else:
                ages.append(None)
                sources.append(None)
        out["age_at_crt"] = pd.array(ages, dtype="Float64")
        out["age_source"] = sources


def _parse_localisation(value: object) -> list[int] | None:
    if co.is_null(value):
        return None
    parts = [p for p in re.split(r"[+,/;&]", str(value)) if p.strip()]
    codes = []
    for part in parts:
        number, err = co.coerce_int(part)
        if number is not None and not err:
            codes.append(number)
    return codes or None


def _resolve_institution(out: pd.DataFrame, raw: pd.DataFrame, log: ParseLog) -> None:
    """Canonicalise institution and cross-check it against the patient-number blocks.

    The two sources are redundant by construction, which is exactly why the check
    is worth running: a mismatch means either the encoding repair mangled a name
    or a patient was entered under the wrong number.
    """
    if "institution" not in raw.columns:
        return
    resolved = raw["institution"].map(sch.canonical_institution)
    expected = [
        sch.institution_from_patient_no(cohort, None if pd.isna(no) else int(no))
        for cohort, no in zip(out["cohort"], out.get("patient_no", pd.Series(dtype="Int64")))
    ]
    out["institution"] = [
        r if r is not None else e for r, e in zip(resolved, expected)
    ]
    out["institution_expected"] = expected
    out["institution_matches_block"] = [
        (r is None) or (e is None) or (r == e) for r, e in zip(resolved, expected)
    ]

    for row, raw_value, r, e in zip(
        raw.to_dict("records"), raw["institution"], resolved, expected
    ):
        if r is None and e is None:
            log.record(
                cohort=row["cohort"],
                row=row["source_row"],
                column="institution",
                value=raw_value,
                reason="unrecognised institution and patient number outside all blocks",
            )
        elif r is not None and e is not None and r != e:
            log.record(
                cohort=row["cohort"],
                row=row["source_row"],
                column="institution",
                value=raw_value,
                reason=f"institution '{r}' contradicts patient-number block '{e}'",
            )

    n_mismatch = int((~pd.Series(out["institution_matches_block"])).sum())
    if n_mismatch:
        logger.warning(f"{n_mismatch} row(s) disagree with the patient-number institution blocks")
    else:
        logger.info("institution cross-check passed for all rows")


def add_patient_key(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the stable de-identified key and assert it is unique."""
    df = df.copy()
    df["patient_key"] = [
        None if pd.isna(no) else sch.patient_key(cohort, int(no))
        for cohort, no in zip(df["cohort"], df["patient_no"])
    ]
    missing = int(df["patient_key"].isna().sum())
    if missing:
        raise ValueError(f"{missing} row(s) have no patient number and cannot be keyed")
    duplicated = df["patient_key"][df["patient_key"].duplicated()].tolist()
    if duplicated:
        raise ValueError(f"duplicate patient keys: {sorted(set(duplicated))}")
    return df


def write_parse_failures(log: ParseLog, path: Path) -> Path:
    """Write the coercion failure log, always -- an empty file is a real result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    log.to_frame().to_csv(path, index=False, encoding="utf-8")
    return path
