"""CLI for curating the clinical data.

    python -m ec_wsi_2026.dataset all        # the usual entry point
    python -m ec_wsi_2026.dataset clinical   # workbooks -> typed interim table
    python -m ec_wsi_2026.dataset labels     # interim -> processed analysis tables
    python -m ec_wsi_2026.dataset audit      # -> reports/clinical_audit.md

The stages are separable so the expensive-to-eyeball parts can be re-run alone,
but ``all`` is what you want: they are fast and the intermediate artifacts exist
for traceability rather than for caching.

**PHI boundary.** ``data/interim`` holds the hospital MRNs and birth dates and
must never be committed or copied off the machine. ``data/processed`` is
de-identified and is what modelling reads.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from ec_wsi_2026.clinical import audit as audit_mod
from ec_wsi_2026.clinical import dictionary as dict_mod
from ec_wsi_2026.clinical import endpoints as ep
from ec_wsi_2026.clinical import schema as sch
from ec_wsi_2026.clinical.loading import (
    add_patient_key,
    coerce_cohorts,
    load_cohorts,
    write_parse_failures,
)
from ec_wsi_2026.config import (
    CLINICAL_AUDIT_MD,
    CLINICAL_COERCED_PARQUET,
    CLINICAL_LONG_RAW_PARQUET,
    CLINICAL_PARQUET,
    CLINICAL_PARSE_FAILURES_CSV,
    DATA_DICTIONARY_MD,
    INTERIM_DATA_DIR,
    LABELS_PARQUET,
    PATIENT_ID_MAP_PARQUET,
)

app = typer.Typer(add_completion=False, help=__doc__)

RESOLUTION_JSON = INTERIM_DATA_DIR / "clinical_column_resolution.json"


@app.command()
def clinical(
    strict: bool = typer.Option(True, help="Abort if an endpoint validation fails."),
) -> None:
    """Read both workbooks, coerce types, derive endpoints, write the interim table."""
    raw, reports = load_cohorts()
    _write(raw, CLINICAL_LONG_RAW_PARQUET)

    coerced, parse_log = coerce_cohorts(raw)
    coerced = add_patient_key(coerced)
    coerced = ep.derive(coerced)
    ep.validate(coerced, strict=strict)

    write_parse_failures(parse_log, CLINICAL_PARSE_FAILURES_CSV)
    _write(coerced, CLINICAL_COERCED_PARQUET)

    # Persist the column resolution so `audit` and `dictionary` can run standalone.
    RESOLUTION_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESOLUTION_JSON.write_text(json.dumps(reports, indent=2, default=str), encoding="utf-8")
    logger.info(f"wrote {RESOLUTION_JSON.name}")


def deidentify(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the coerced table into the id map and the de-identified table.

    The id map is the only artifact that can re-identify a patient; it never
    leaves ``data/interim``. The de-identified frame keeps every other column.
    """
    id_columns = ["patient_key", "cohort", "source_row"] + [
        c for c in sch.PHI_COLUMNS if c in df.columns
    ]
    id_map = df[id_columns].copy()

    dropped = [c for c in sch.PHI_COLUMNS if c in df.columns]
    deidentified = df.drop(columns=dropped)
    logger.info(f"dropped direct identifiers from the processed table: {dropped}")
    return id_map, deidentified


@app.command()
def labels() -> None:
    """Split the interim table into the de-identified analysis tables."""
    df = pd.read_parquet(CLINICAL_COERCED_PARQUET)

    # The MRN map stays in interim, which is gitignored, and is written
    # separately so that nothing downstream has to touch it.
    id_map, deidentified = deidentify(df)
    _write(id_map, PATIENT_ID_MAP_PARQUET)
    _write(deidentified, CLINICAL_PARQUET)
    _write(ep.labels_table(deidentified), LABELS_PARQUET)


@app.command()
def audit(
    strict: bool = typer.Option(True, help="Abort on an undocumented code value."),
) -> None:
    """Write the markdown audit report."""
    df = pd.read_parquet(CLINICAL_COERCED_PARQUET)
    failures = pd.read_csv(CLINICAL_PARSE_FAILURES_CSV) if CLINICAL_PARSE_FAILURES_CSV.exists() else pd.DataFrame()

    legends = audit_mod.load_code_legends()
    audit_mod.check_code_legends(df, legends, strict=strict)

    text = audit_mod.build_report(
        df, resolution_reports=_load_resolution(), parse_failures=failures, legends=legends
    )
    audit_mod.write_report(text, CLINICAL_AUDIT_MD)


@app.command()
def dictionary() -> None:
    """Regenerate references/data_dictionary.md from the schema and the curated table."""
    df = pd.read_parquet(CLINICAL_COERCED_PARQUET)
    resolution = {r["cohort"]: r["resolved"] for r in _load_resolution()}
    dict_mod.write(dict_mod.build(df, resolution), DATA_DICTIONARY_MD)


@app.command()
def all(
    strict: bool = typer.Option(True, help="Abort on any validation failure."),
) -> None:
    """Run clinical -> labels -> audit -> dictionary."""
    clinical(strict=strict)
    labels()
    audit(strict=strict)
    dictionary()
    logger.success("clinical curation complete")


def _load_resolution() -> list[dict]:
    if not RESOLUTION_JSON.exists():
        logger.warning(f"{RESOLUTION_JSON.name} not found; run `dataset clinical` first")
        return []
    return json.loads(RESOLUTION_JSON.read_text(encoding="utf-8"))


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Object columns holding pd.NA alongside strings break the parquet writer's
    # type inference; normalise them first.
    out = df.copy()
    for column in out.columns:
        if out[column].dtype == object:
            out[column] = out[column].astype("object").where(out[column].notna(), None)
    out.to_parquet(path, index=False)
    logger.info(f"wrote {path.relative_to(path.parents[2])}  ({len(out)} rows x {len(out.columns)} cols)")


if __name__ == "__main__":
    app()
