"""Derive the time-to-event and binary endpoints from the date columns.

Every duration column in the workbooks (``PFS (days)``, ``OS (days)``,
``LFFS (days)``) is entirely empty, so all of them are computed here. Two
conventions do the real work and are documented in
``references/endpoint_definitions.md``:

**Time origin is ``Date Start CRT`` for both cohorts.** It is the only index date
populated for all 209 patients, and it is the only choice that permits pooling.
Anchoring neoadjuvant patients at ``Date OP`` while anchoring definitive patients
at CRT start would guarantee that neoadjuvant patients survived long enough to
reach surgery -- immortal time bias, favouring the neoadjuvant arm by
construction.

Two secondary origins are emitted as explicit **landmark analyses** and must
never be pooled with the CRT-anchored endpoints: ``dfs`` (neoadjuvant only,
anchored at ``Date OP`` for comparability with the surgical literature, which
reports disease-free survival from resection) and ``pfs_endcrt``/``os_endcrt``
(anchored at ``Date End CRT``, a sensitivity analysis for the origin choice).
A patient enters a landmark endpoint only if the origin date exists, is not
implicated in a suspected transcription error (``anomalous_date_columns``), and
no event or censoring falls on or before it -- a patient who progressed before
surgery was never disease-free at the landmark, and excluding them is the whole
point: counting them would smuggle the immortal-time bias back in. Ineligible
rows are missing, not clamped.

**The censoring date is the maximum over all follow-up evidence**, not a
preference-ordered coalesce. ``Last FU date`` and ``Last date alive`` are each
populated for different subsets and neither dominates -- ``Last date alive`` is
sometimes the later of the two. Taking the max uses every contact; a coalesce in
either direction discards information and shortens follow-up.

Death is counted as a PFS event. 43 patients died with no recorded progression;
treating them as censored would be informative censoring, since dying is not
independent of progressing. ``ttp`` (progression only, death censors) is emitted
alongside as the pre-specified sensitivity analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
import numpy as np
import pandas as pd

# Columns searched for the censoring date, in no particular order -- the max wins.
CENSOR_DATE_COLUMNS = (
    "last_date_alive",
    "last_fu_date",
    "date_progress",
    "date_local_failure",
    "date_death",
    "date_ct_post_crt",
    "date_op",
)


@dataclass(frozen=True)
class Endpoint:
    """One derived endpoint.

    ``event_columns`` are the indicator columns OR-ed together to form the event;
    ``date_columns`` are the corresponding event dates, of which the *earliest*
    observed is the event time. ``cohorts`` restricts the endpoint to the arms
    where it is meaningful. ``origin`` is the column supplying t0; any origin
    other than the default turns the endpoint into a landmark analysis with the
    stricter eligibility rules documented in the module docstring.
    """

    name: str
    kind: str  # "time_to_event" | "binary"
    cohorts: tuple[str, ...]
    event_columns: tuple[str, ...] = ()
    date_columns: tuple[str, ...] = ()
    origin: str = "date_start_crt"
    description: str = ""

    @property
    def is_landmark(self) -> bool:
        return self.origin != "date_start_crt"


ENDPOINTS: tuple[Endpoint, ...] = (
    Endpoint(
        name="os",
        kind="time_to_event",
        cohorts=("definitive", "neoadjuvant"),
        event_columns=("death",),
        date_columns=("date_death",),
        description="Overall survival: death from any cause.",
    ),
    Endpoint(
        name="pfs",
        kind="time_to_event",
        cohorts=("definitive", "neoadjuvant"),
        event_columns=("progress", "death"),
        date_columns=("date_progress", "date_death"),
        description="Progression-free survival: progression or death, whichever first.",
    ),
    Endpoint(
        name="ttp",
        kind="time_to_event",
        cohorts=("definitive", "neoadjuvant"),
        event_columns=("progress",),
        date_columns=("date_progress",),
        description="Time to progression; death censors. Sensitivity analysis for PFS.",
    ),
    Endpoint(
        name="lffs",
        kind="time_to_event",
        cohorts=("definitive",),
        event_columns=("local_failure",),
        date_columns=("date_local_failure",),
        description="Local failure-free survival. Definitive cohort only.",
    ),
    Endpoint(
        name="dfs",
        kind="time_to_event",
        cohorts=("neoadjuvant",),
        event_columns=("progress", "death"),
        date_columns=("date_progress", "date_death"),
        origin="date_op",
        description=(
            "Disease-free survival from surgery: recurrence or death after resection. "
            "Landmark at Date OP, conditional on having reached surgery; comparable with "
            "the surgical literature but never pooled with the CRT-anchored endpoints."
        ),
    ),
    Endpoint(
        name="pfs_endcrt",
        kind="time_to_event",
        cohorts=("definitive", "neoadjuvant"),
        event_columns=("progress", "death"),
        date_columns=("date_progress", "date_death"),
        origin="date_end_crt",
        description=(
            "PFS anchored at Date End CRT. Sensitivity analysis: tests whether conclusions "
            "depend on the time-origin choice."
        ),
    ),
    Endpoint(
        name="os_endcrt",
        kind="time_to_event",
        cohorts=("definitive", "neoadjuvant"),
        event_columns=("death",),
        date_columns=("date_death",),
        origin="date_end_crt",
        description=(
            "OS anchored at Date End CRT. Sensitivity analysis: tests whether conclusions "
            "depend on the time-origin choice."
        ),
    ),
    Endpoint(
        name="pcr",
        kind="binary",
        cohorts=("neoadjuvant",),
        description="Pathological complete response (ypT0 ypN0). Neoadjuvant only.",
    ),
    Endpoint(
        name="becker_major",
        kind="binary",
        cohorts=("neoadjuvant",),
        description="Major pathological response: Becker 1a or 1b (<10% residual tumour).",
    ),
)

ENDPOINTS_BY_NAME = {e.name: e for e in ENDPOINTS}


def derive(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``{endpoint}_time`` / ``{endpoint}_event`` columns plus supporting fields.

    Returns a copy of *df*. Rows outside an endpoint's cohorts get ``NaN`` for that
    endpoint rather than being dropped, so one table serves every analysis.
    """
    out = df.copy()
    out["censor_date"], out["censor_source"] = _censor_date(out)
    _add_treatment_intervals(out)

    for endpoint in ENDPOINTS:
        in_cohort = out["cohort"].isin(endpoint.cohorts)
        if endpoint.kind == "binary":
            out[f"{endpoint.name}_event"] = _binary_event(out, endpoint.name).where(in_cohort)
            continue

        event, event_date = _event_and_date(out, endpoint)
        end_date = event_date.where(event == 1, out["censor_date"])
        origin = out.get(endpoint.origin, pd.Series(pd.NaT, index=out.index))
        time = (end_date - origin).dt.days

        # An event with no usable date cannot be timed, and a duration that cannot
        # be computed is not a zero -- both must be missing, not guessed.
        usable = in_cohort & event.notna() & time.notna()

        if endpoint.is_landmark:
            # Landmark conditioning: the origin must exist and be trustworthy, and
            # the patient must still be event-free *after* it. `time > 0` covers
            # both an event on/before the origin (never at risk from t0) and a
            # censoring contact on/before it (no at-risk time to contribute).
            suspect = _origin_is_anomalous(out, endpoint.origin)
            eligible = in_cohort & origin.notna() & ~suspect & (time > 0)
            n_in_cohort = int(in_cohort.sum())
            excluded = int((in_cohort & event.notna() & ~eligible).sum())
            if excluded:
                logger.info(
                    f"{endpoint.name}: landmark at {endpoint.origin} excludes {excluded} of "
                    f"{n_in_cohort} in-cohort patient(s) (missing/suspect origin, or an event "
                    "or censoring on or before it)"
                )
            usable &= eligible

        out[f"{endpoint.name}_event"] = event.where(usable).astype("Int64")
        out[f"{endpoint.name}_time"] = time.where(usable).astype("Float64")
        out[f"{endpoint.name}_end_date"] = end_date.where(usable)

    _log_summary(out)
    return out


def _censor_date(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Latest follow-up contact of any kind, and which column supplied it."""
    present = [c for c in CENSOR_DATE_COLUMNS if c in df.columns]
    if not present:
        raise ValueError("no follow-up date columns available to derive a censoring date")
    dates = df[present]
    latest = dates.max(axis=1)
    # idxmax over a boolean frame gives the first column attaining the max.
    source = (dates.eq(latest, axis=0) & dates.notna()).idxmax(axis=1)
    return latest, source.where(latest.notna())


def _event_and_date(df: pd.DataFrame, endpoint: Endpoint) -> tuple[pd.Series, pd.Series]:
    """OR the indicator columns; take the earliest corresponding event date."""
    indicators, dates = [], []
    for event_col, date_col in zip(endpoint.event_columns, endpoint.date_columns):
        if event_col not in df.columns:
            continue
        flag = pd.to_numeric(df[event_col], errors="coerce")
        indicators.append(flag)
        if date_col in df.columns:
            dates.append(df[date_col].where(flag == 1))

    if not indicators:
        nan = pd.Series(np.nan, index=df.index)
        return nan, pd.Series(pd.NaT, index=df.index)

    stacked = pd.concat(indicators, axis=1)
    # Any 1 makes an event; otherwise 0 if at least one indicator was observed.
    event = pd.Series(np.nan, index=df.index)
    event[stacked.notna().any(axis=1)] = 0.0
    event[(stacked == 1).any(axis=1)] = 1.0

    if dates:
        event_date = pd.concat(dates, axis=1).min(axis=1)
    else:
        event_date = pd.Series(pd.NaT, index=df.index)
    return event, event_date


def _origin_is_anomalous(df: pd.DataFrame, origin: str) -> pd.Series:
    """Rows where *origin* participates in a suspected transcription error."""
    if "anomalous_date_columns" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["anomalous_date_columns"].apply(
        lambda s: origin in s.split(";") if isinstance(s, str) else False
    )


def _binary_event(df: pd.DataFrame, name: str) -> pd.Series:
    if name == "pcr":
        if "pcr" not in df.columns:
            return pd.Series(pd.NA, index=df.index, dtype="Int64")
        return pd.to_numeric(df["pcr"], errors="coerce").astype("Int64")
    if name == "becker_major":
        if "becker_grade" not in df.columns:
            return pd.Series(pd.NA, index=df.index, dtype="Int64")
        grade = df["becker_grade"]
        return pd.Series(
            [pd.NA if g is None else int(g in {"1a", "1b", "1"}) for g in grade],
            index=df.index,
            dtype="Int64",
        )
    raise ValueError(f"unknown binary endpoint: {name}")


def _add_treatment_intervals(out: pd.DataFrame) -> None:
    """Durations describing the treatment course itself, all derived from dates.

    An interval that comes out negative is a transcription error in the source
    workbook, not a real duration. Those are nulled and flagged in
    ``date_anomaly`` rather than corrected here: the fix belongs to the data
    owner, and a silently "repaired" date is worse than a missing one. None of
    them touch the endpoints, which are anchored on ``date_start_crt``.
    """
    anomalies: list[list[str]] = [[] for _ in range(len(out))]
    suspect_columns: list[set[str]] = [set() for _ in range(len(out))]

    for name, (a, b) in {
        "crt_duration_days": ("date_start_crt", "date_end_crt"),
        "crt_end_to_op_days": ("date_end_crt", "date_op"),
        "staging_to_crt_days": ("date_endoscopy", "date_start_crt"),
    }.items():
        if a not in out.columns or b not in out.columns:
            continue
        series = (out[b] - out[a]).dt.days.astype("Float64")
        negative = series.notna() & (series < 0)
        for position in np.flatnonzero(negative.to_numpy()):
            anomalies[position].append(
                f"{name}={int(series.iloc[position])}d ({b}={out[b].iloc[position]:%Y-%m-%d} "
                f"precedes {a}={out[a].iloc[position]:%Y-%m-%d})"
            )
            # The code cannot know which of the pair holds the typo, so both are
            # suspect. Landmark endpoints refuse to anchor on a suspect date; the
            # default-origin endpoints are unaffected, per the documented finding
            # that all three known anomalies leave `date_start_crt` valid.
            suspect_columns[position].update((a, b))
        out[name] = series.where(~negative)

    out["date_anomaly"] = ["; ".join(a) if a else None for a in anomalies]
    out["anomalous_date_columns"] = [";".join(sorted(s)) if s else None for s in suspect_columns]
    n_flagged = int(out["date_anomaly"].notna().sum())
    if n_flagged:
        logger.warning(
            f"{n_flagged} patient(s) have an impossible treatment interval; the interval was "
            "nulled and flagged in `date_anomaly`. Endpoints are unaffected."
        )

    if "date_start_crt" in out.columns:
        out["treatment_year"] = out["date_start_crt"].dt.year.astype("Int64")
    if "censor_date" in out.columns and "date_start_crt" in out.columns:
        out["followup_days"] = (out["censor_date"] - out["date_start_crt"]).dt.days.astype("Float64")


def _log_summary(out: pd.DataFrame) -> None:
    for endpoint in ENDPOINTS:
        event_col = f"{endpoint.name}_event"
        if event_col not in out.columns:
            continue
        events = out[event_col]
        n = int(events.notna().sum())
        positive = int((events == 1).sum())
        if endpoint.kind == "time_to_event":
            median = out.loc[events.notna(), f"{endpoint.name}_time"].median()
            logger.info(
                f"{endpoint.name}: n={n} events={positive} censored={n - positive} "
                f"median_time={median:.0f}d"
            )
        else:
            logger.info(f"{endpoint.name}: n={n} positive={positive} ({positive / max(n, 1):.0%})")


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate(df: pd.DataFrame, *, strict: bool = True) -> list[str]:
    """Check the derived endpoints against known invariants of this cohort.

    Returns the list of violations. With ``strict`` the caller is expected to
    abort: these are not warnings about data quality, they are signals that the
    derivation logic or the source data changed in a way that invalidates the
    labels.
    """
    problems: list[str] = []

    def fail(message: str) -> None:
        problems.append(message)
        logger.error(f"VALIDATION: {message}")

    # 1. Durations must be positive. A negative one means an event date precedes
    #    the treatment start, i.e. a data-entry error.
    for endpoint in ENDPOINTS:
        col = f"{endpoint.name}_time"
        if col not in df.columns:
            continue
        negative = df[df[col].notna() & (df[col] < 0)]
        if len(negative):
            keys = negative["patient_key"].tolist()
            fail(f"{endpoint.name}: {len(negative)} negative duration(s): {keys}")

    # 2. An observed event must have a date, or it cannot be timed.
    for event_col, date_col in (
        ("death", "date_death"),
        ("progress", "date_progress"),
        ("local_failure", "date_local_failure"),
    ):
        if event_col not in df.columns or date_col not in df.columns:
            continue
        undated = df[(pd.to_numeric(df[event_col], errors="coerce") == 1) & df[date_col].isna()]
        if len(undated):
            fail(
                f"{event_col}=1 with no {date_col} for {len(undated)} patient(s): "
                f"{undated['patient_key'].tolist()}"
            )

    # 3. Local failure is a form of progression, so it must imply it.
    if {"local_failure", "progress"} <= set(df.columns):
        contradiction = df[
            (pd.to_numeric(df["local_failure"], errors="coerce") == 1)
            & (pd.to_numeric(df["progress"], errors="coerce") == 0)
        ]
        if len(contradiction):
            fail(
                f"local failure without progression for {len(contradiction)} patient(s): "
                f"{contradiction['patient_key'].tolist()}"
            )

    # 4. pCR is defined as ypT0 ypN0; if the columns ever disagree, one of them
    #    is wrong and the pCR label cannot be trusted.
    if {"pcr", "ypt_stage_num", "ypn_stage_num"} <= set(df.columns):
        subset = df[df["pcr"].notna() & df["ypt_stage_num"].notna() & df["ypn_stage_num"].notna()]
        complete = (subset["ypt_stage_num"] == 0) & (subset["ypn_stage_num"] == 0)
        disagree = subset[(pd.to_numeric(subset["pcr"], errors="coerce") == 1) != complete]
        if len(disagree):
            fail(
                f"pCR disagrees with ypT0/ypN0 for {len(disagree)} patient(s): "
                f"{disagree['patient_key'].tolist()}"
            )

    # 5. Follow-up must not precede the time origin. Unlike the intra-treatment
    #    intervals (quarantined in `_add_treatment_intervals`), this one *would*
    #    corrupt every endpoint, so it blocks.
    if {"date_start_crt", "censor_date"} <= set(df.columns):
        both = df[df["date_start_crt"].notna() & df["censor_date"].notna()]
        inverted = both[both["censor_date"] < both["date_start_crt"]]
        if len(inverted):
            fail(
                f"censor_date precedes date_start_crt for {len(inverted)} patient(s): "
                f"{inverted['patient_key'].tolist()}"
            )

    # Transcription errors in the treatment timeline are reported, not raised:
    # they are the data owner's to fix and they do not affect the endpoints.
    if "date_anomaly" in df.columns:
        flagged = df[df["date_anomaly"].notna()]
        for _, row in flagged.iterrows():
            logger.warning(f"SUSPECTED TRANSCRIPTION ERROR {row['patient_key']}: {row['date_anomaly']}")

    # 6. Age must be resolved for every patient, and must be plausible. There is
    #    no derived-vs-recorded cross-check to be had: the two sources are
    #    disjoint by construction (only the Würzburg patients have a recorded age,
    #    and they are exactly the ones with no birth date).
    if "age_at_crt" in df.columns:
        missing_age = df[df["age_at_crt"].isna()]
        if len(missing_age):
            fail(
                f"age could not be resolved for {len(missing_age)} patient(s): "
                f"{missing_age['patient_key'].tolist()}"
            )
        implausible = df[df["age_at_crt"].notna() & ((df["age_at_crt"] < 18) | (df["age_at_crt"] > 100))]
        if len(implausible):
            fail(
                f"implausible age for {len(implausible)} patient(s): "
                f"{implausible['patient_key'].tolist()}"
            )

    # 7. A landmark endpoint must never carry a value for a patient without its
    #    origin date -- that would be exactly the immortal-time construction the
    #    landmark design exists to prevent.
    for endpoint in ENDPOINTS:
        if endpoint.kind != "time_to_event" or not endpoint.is_landmark:
            continue
        time_col = f"{endpoint.name}_time"
        if time_col not in df.columns or endpoint.origin not in df.columns:
            continue
        orphaned = df[df[time_col].notna() & df[endpoint.origin].isna()]
        if len(orphaned):
            fail(
                f"{endpoint.name}: {len(orphaned)} patient(s) have a time but no "
                f"{endpoint.origin}: {orphaned['patient_key'].tolist()}"
            )

    if problems and strict:
        raise ValueError(
            f"{len(problems)} endpoint validation failure(s):\n  " + "\n  ".join(problems)
        )
    if not problems:
        logger.success("all endpoint validations passed")
    return problems


def labels_table(df: pd.DataFrame) -> pd.DataFrame:
    """Project to the minimal modelling table: keys, strata, and endpoints."""
    keep = ["patient_key", "cohort", "institution", "treatment_year", "followup_days"]
    for endpoint in ENDPOINTS:
        keep.append(f"{endpoint.name}_event")
        if endpoint.kind == "time_to_event":
            keep.append(f"{endpoint.name}_time")
    return df[[c for c in keep if c in df.columns]].copy()
