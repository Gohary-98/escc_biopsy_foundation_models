"""Load and enforce the pre-specified covariate sets in references/covariates.yaml.

The YAML file is the specification; this module refuses to hand a model a
covariate set that references a missing column, uses a column with an
unconfirmed code legend, or exceeds the events-per-variable budget. Encoding
happens here too (:func:`design_matrix`), so every model sees the same numeric
matrix. Imputation deliberately does **not** happen here -- it belongs inside
the training fold, fitted on training rows only.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
import pandas as pd
import yaml

from ec_wsi_2026.clinical.audit import unconfirmed_legends
from ec_wsi_2026.clinical.endpoints import ENDPOINTS_BY_NAME
from ec_wsi_2026.config import COVARIATES_YAML

# Events per variable: the conventional floor for regression on censored or
# binary outcomes. For binary endpoints the binding count is the minority class.
EPV = 10


def load_spec(path: Path = COVARIATES_YAML) -> dict:
    """Read the covariate specification. The file is required, not optional."""
    if not path.exists():
        raise FileNotFoundError(
            f"covariate specification not found at {path}. Training refuses to run without a "
            "pre-specified covariate set; see references/covariates.yaml."
        )
    with path.open(encoding="utf-8") as handle:
        spec = yaml.safe_load(handle) or {}
    if "sets" not in spec:
        raise ValueError(f"{path.name} has no `sets` mapping")
    return spec


def effective_events(labels: pd.DataFrame, endpoint: str) -> int:
    """The event count that governs the EPV budget for *endpoint*."""
    kind = ENDPOINTS_BY_NAME[endpoint].kind
    events = labels[f"{endpoint}_event"].dropna().astype(int)
    if kind == "time_to_event":
        return int((events == 1).sum())
    # Binary: the minority class binds.
    return int(min((events == 1).sum(), (events == 0).sum()))


def validate_spec(spec: dict, clinical: pd.DataFrame, labels: pd.DataFrame) -> None:
    """Raise unless every covariate set is usable as specified."""
    blocklist = set(unconfirmed_legends())
    problems: list[str] = []

    for endpoint, entry in spec["sets"].items():
        if endpoint not in ENDPOINTS_BY_NAME:
            problems.append(f"{endpoint}: not a registered endpoint")
            continue
        covariates = entry.get("covariates", [])
        missing = [c for c in covariates if c not in clinical.columns]
        if missing:
            problems.append(f"{endpoint}: column(s) absent from clinical.parquet: {missing}")
        blocked = [c for c in covariates if c in blocklist]
        if blocked:
            problems.append(
                f"{endpoint}: column(s) with unconfirmed code legends: {blocked} "
                "(see references/code_legends.yaml)"
            )
        budget = effective_events(labels, endpoint) // EPV
        if len(covariates) > budget:
            problems.append(
                f"{endpoint}: {len(covariates)} covariates exceed the EPV budget of {budget} "
                f"({effective_events(labels, endpoint)} effective events / {EPV})"
            )

    if problems:
        raise ValueError(
            f"{len(problems)} covariate specification problem(s):\n  " + "\n  ".join(problems)
        )
    logger.success("covariate specification validated against the current tables")


def design_matrix(clinical: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    """Encode the requested covariates as a numeric matrix, indexed like *clinical*.

    Missing values pass through as NaN -- imputation is the training fold's job.
    """
    out = pd.DataFrame(index=clinical.index)
    for column in covariates:
        series = clinical[column]
        if column == "cohort":
            unexpected = set(series.dropna().unique()) - {"definitive", "neoadjuvant"}
            if unexpected:
                raise ValueError(f"unexpected cohort value(s): {sorted(unexpected)}")
            out[column] = (series == "neoadjuvant").astype("Int64")
        elif pd.api.types.is_bool_dtype(series):
            # Covers plain bool and the nullable boolean extension dtype
            # (`cn_involved`), which `series.dtype == bool` would miss.
            out[column] = series.astype("Int64")
        else:
            out[column] = pd.to_numeric(series, errors="raise")
    return out.astype("Float64")
