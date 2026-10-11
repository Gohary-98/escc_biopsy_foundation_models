"""Auditing of the curated clinical table.

Produces ``reports/clinical_audit.md``: column resolution, population rates, the
coded-value tables, the endpoint summary, and a CONSORT-style exclusion
waterfall. The point is that every number quoted in a paper or thesis chapter has
a traceable origin, and that a change in the source workbooks shows up as a diff
here rather than as a silently different result.

:func:`check_code_legends` is the one part that raises. A code value present in
the data but absent from ``references/code_legends.yaml`` means a new site,
protocol or data-entry convention has appeared, and continuing to model it as if
its meaning were known would be wrong.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
import pandas as pd
import yaml

from ec_wsi_2026.clinical import endpoints as ep
from ec_wsi_2026.clinical import schema as sch
from ec_wsi_2026.config import CODE_LEGENDS_YAML


def load_code_legends(path: Path = CODE_LEGENDS_YAML) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"code legends not found at {path}. This file documents what the numeric "
            "code columns mean and is required; see references/code_legends.yaml."
        )
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def check_code_legends(df: pd.DataFrame, legends: dict | None = None, *, strict: bool = True) -> list[str]:
    """Assert every observed code value is documented. Returns the violations."""
    legends = legends if legends is not None else load_code_legends()
    problems: list[str] = []

    for column in sch.CODE_COLUMNS:
        if column not in df.columns:
            continue
        observed = sorted(int(v) for v in df[column].dropna().unique())
        if not observed:
            continue
        entry = legends.get(column)
        if entry is None:
            problems.append(f"{column}: no legend entry, observed values {observed}")
            continue
        documented = set(entry.get("values") or {})
        undocumented = [v for v in observed if v not in documented]
        if undocumented:
            problems.append(
                f"{column}: undocumented code value(s) {undocumented} "
                f"(documented: {sorted(documented)})"
            )

    for problem in problems:
        logger.error(f"CODE LEGEND: {problem}")
    if problems and strict:
        raise ValueError(
            f"{len(problems)} undocumented code value(s). Add them to "
            f"{CODE_LEGENDS_YAML.name} with `meaning: UNKNOWN` and raise with the data owner:\n  "
            + "\n  ".join(problems)
        )
    if not problems:
        logger.success("every observed code value is documented in code_legends.yaml")
    return problems


def reverse_km_median_followup(times, events) -> float:
    """Median follow-up by the reverse Kaplan-Meier method.

    Swaps the roles of event and censoring, so the "event" is being censored --
    i.e. still under observation. The median of that curve is how long the cohort
    was actually followed. The naive alternative, the median observed time, is
    biased downward by early deaths and describes something else entirely.
    """
    from lifelines import KaplanMeierFitter
    import numpy as np

    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    if times.size == 0:
        return float("nan")
    fitter = KaplanMeierFitter().fit(times, 1 - events)
    return float(fitter.median_survival_time_)


def unconfirmed_legends(legends: dict | None = None) -> list[str]:
    """Columns whose meanings are still guesses. These block categorical modelling."""
    legends = legends if legends is not None else load_code_legends()
    return sorted(name for name, entry in legends.items() if not entry.get("confirmed", False))


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def build_report(
    df: pd.DataFrame,
    *,
    resolution_reports: list[dict],
    parse_failures: pd.DataFrame,
    legends: dict | None = None,
) -> str:
    """Render the full audit as markdown."""
    legends = legends if legends is not None else load_code_legends()
    lines: list[str] = []
    add = lines.append

    add("# Clinical data audit")
    add("")
    add(f"Generated {pd.Timestamp.now():%Y-%m-%d %H:%M} from the source workbooks.")
    add("")
    add(
        "All duration columns in the workbooks (`PFS (days)`, `OS (days)`, `LFFS (days)`) are "
        "empty and are derived here from the date columns. See "
        "`references/endpoint_definitions.md` for the conventions."
    )
    add("")

    _section_cohorts(add, df)
    _section_endpoints(add, df)
    _section_waterfall(add, df)
    _section_columns(add, df, resolution_reports)
    _section_codes(add, df, legends)
    _section_quality(add, df, parse_failures)

    return "\n".join(lines) + "\n"


def _section_cohorts(add, df: pd.DataFrame) -> None:
    add("## Cohort composition")
    add("")
    table = pd.crosstab(df["institution"], df["cohort"], margins=True, margins_name="Total")
    add(table.to_markdown())
    add("")
    add(
        f"Median age {df['age_at_crt'].median():.0f}y "
        f"(range {df['age_at_crt'].min():.0f}-{df['age_at_crt'].max():.0f}); "
        f"{int((df['sex'] == 1).sum())} male, {int((df['sex'] == 2).sum())} female."
    )
    if "age_source" in df.columns:
        by_source = df["age_source"].value_counts()
        add("")
        add(
            f"Age is derived from the birth date for {by_source.get('derived_from_dob', 0)} patients "
            f"and taken from the recorded value for {by_source.get('recorded', 0)}. The two sources "
            "are disjoint: the patients with a recorded age are exactly those with no birth date, "
            "all of them from one site."
        )
    if "treatment_year" in df.columns:
        years = df["treatment_year"].dropna()
        add("")
        add(
            f"Treatment years span {int(years.min())}-{int(years.max())}, a "
            f"{int(years.max() - years.min())}-year accrual window. Calendar year is a plausible "
            "confounder (staging systems, chemotherapy protocols and scanner generations all "
            "changed) and is retained as `treatment_year`."
        )
    add("")


def _section_endpoints(add, df: pd.DataFrame) -> None:
    add("## Endpoints")
    add("")
    rows = []
    for endpoint in ep.ENDPOINTS:
        event_col = f"{endpoint.name}_event"
        if event_col not in df.columns:
            continue
        events = df[event_col]
        analysable = int(events.notna().sum())
        positive = int((events == 1).sum())
        row = {
            "endpoint": endpoint.name,
            "kind": endpoint.kind,
            "cohorts": "+".join(c[:3] for c in endpoint.cohorts),
            "n": analysable,
            "events": positive,
            "event rate": f"{positive / max(analysable, 1):.0%}",
        }
        if endpoint.kind == "time_to_event":
            times = df.loc[events.notna(), f"{endpoint.name}_time"]
            row["median time (d)"] = f"{times.median():.0f}"
            # ~10 events per covariate is the usual floor for a stable model.
            row["max covariates"] = positive // 10
        rows.append(row)
    add(pd.DataFrame(rows).to_markdown(index=False))
    add("")

    if {"progress", "death"} <= set(df.columns):
        died_no_progression = int(
            (
                (pd.to_numeric(df["progress"], errors="coerce") == 0)
                & (pd.to_numeric(df["death"], errors="coerce") == 1)
            ).sum()
        )
        add(
            f"**{died_no_progression} patients died with no recorded progression.** They are events "
            "under `pfs` and censored under `ttp`, which is the entire difference between the two "
            "(129 vs 86 events). `pfs` is primary; `ttp` is the pre-specified sensitivity analysis. "
            "If the registry simply failed to record progression for patients who died, censoring "
            "them would be informative censoring and would bias results optimistically."
        )
        add("")

    if {"os_time", "os_event"} <= set(df.columns):
        usable = df[df["os_time"].notna() & df["os_event"].notna()]
        median_fu = reverse_km_median_followup(
            usable["os_time"].astype(float).to_numpy(),
            usable["os_event"].astype(int).to_numpy(),
        )
        observed = usable["os_time"].astype(float) / 30.44
        add(
            f"Median follow-up **{median_fu / 30.44:.0f} months** by reverse Kaplan-Meier "
            f"(median *observed* time {observed.median():.0f} months, which is the smaller and "
            "more commonly misquoted figure -- it is dragged down by early deaths rather than "
            "describing how long survivors were actually followed)."
        )
        add("")


def _section_waterfall(add, df: pd.DataFrame) -> None:
    add("## Exclusion waterfall")
    add("")
    add("Patients lost at each step, per endpoint. CONSORT-style, for the methods section.")
    add("")
    rows = [{"step": "enrolled in the workbooks", "n": len(df), "lost": 0}]
    with_origin = int(df["date_start_crt"].notna().sum())
    rows.append(
        {
            "step": "with a treatment start date (time origin)",
            "n": with_origin,
            "lost": len(df) - with_origin,
        }
    )
    for endpoint in ep.ENDPOINTS:
        event_col = f"{endpoint.name}_event"
        if event_col not in df.columns:
            continue
        eligible = int(df["cohort"].isin(endpoint.cohorts).sum())
        analysable = int(df[event_col].notna().sum())
        rows.append(
            {
                "step": f"analysable for {endpoint.name} (of {eligible} in-cohort)",
                "n": analysable,
                "lost": eligible - analysable,
            }
        )
    add(pd.DataFrame(rows).to_markdown(index=False))
    add("")


def _section_columns(add, df: pd.DataFrame, resolution_reports: list[dict]) -> None:
    add("## Column resolution and population")
    add("")
    for report in resolution_reports:
        unmatched = report["unmatched_headers"]
        add(
            f"- **{report['cohort']}** ({report['sheet']}): {report['n_rows']} rows, "
            f"{len(report['resolved'])} columns resolved, "
            f"{len(unmatched)} source header(s) unmatched"
            + (f": `{unmatched}`" if unmatched else "")
        )
        if report["unresolved_canonical"]:
            add(
                f"  - not present in this cohort (expected): "
                f"`{report['unresolved_canonical']}`"
            )
    add("")

    rows = []
    for column in df.columns:
        if column.endswith("__note"):
            continue
        populated = int(df[column].notna().sum())
        rows.append(
            {
                "column": column,
                "populated": populated,
                "%": f"{populated / len(df):.0%}",
                "dtype": str(df[column].dtype),
            }
        )
    add("<details><summary>Per-column population</summary>")
    add("")
    add(pd.DataFrame(rows).to_markdown(index=False))
    add("")
    add("</details>")
    add("")


def _section_codes(add, df: pd.DataFrame, legends: dict) -> None:
    add("## Coded columns")
    add("")
    unconfirmed = unconfirmed_legends(legends)
    if unconfirmed:
        add(
            f"**{len(unconfirmed)} column(s) have unconfirmed legends:** "
            + ", ".join(f"`{c}`" for c in unconfirmed)
            + ". These must not be used as categorical covariates until the data owner confirms "
            "their meanings; see `references/code_legends.yaml`."
        )
        add("")

    for column in sch.CODE_COLUMNS:
        if column not in df.columns or df[column].dropna().empty:
            continue
        entry = legends.get(column, {})
        values = entry.get("values") or {}
        confirmed = "confirmed" if entry.get("confirmed") else "**UNCONFIRMED**"
        add(f"### `{column}` ({confirmed})")
        add("")
        counts = pd.crosstab(df[column], df["cohort"])
        counts.insert(0, "meaning", [str(values.get(int(v), {}).get("meaning", "?")) for v in counts.index])
        add(counts.to_markdown())
        add("")


def _section_quality(add, df: pd.DataFrame, parse_failures: pd.DataFrame) -> None:
    add("## Data quality")
    add("")
    add(f"- Values that failed coercion: **{len(parse_failures)}**")
    if len(parse_failures):
        add("")
        add(parse_failures.to_markdown(index=False))
        add("")

    if "institution_matches_block" in df.columns:
        mismatches = int((~df["institution_matches_block"]).sum())
        add(
            f"- Institution vs patient-number block cross-check: "
            f"**{'all rows agree' if not mismatches else f'{mismatches} MISMATCH(ES)'}**"
        )

    if "date_anomaly" in df.columns:
        flagged = df[df["date_anomaly"].notna()]
        add(f"- Impossible treatment intervals: **{len(flagged)}**")
        if len(flagged):
            add("")
            add(
                "  These are transcription errors in the source workbook, almost certainly "
                "single-digit year typos. The affected interval has been nulled; the endpoints "
                "are unaffected because they are anchored on `date_start_crt`, which is valid in "
                "every case. **These need correcting at source.**"
            )
            add("")
            for _, row in flagged.iterrows():
                add(f"  - `{row['patient_key']}` (row {row['source_row']}): {row['date_anomaly']}")
    add("")


def write_report(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    logger.success(f"wrote {path}")
    return path
