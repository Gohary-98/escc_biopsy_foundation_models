"""Endpoint derivation tests on synthetic cohorts.

These exercise the two conventions that carry the most statistical weight -- that
death counts as a PFS event, and that the censoring date is the maximum over all
follow-up evidence -- plus the validation rules that are supposed to catch a
future change in the source data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ec_wsi_2026.clinical import endpoints as ep


def make_cohort(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal frame with every column `derive` expects."""
    defaults = {
        "cohort": "definitive",
        "patient_key": "def_001",
        "source_row": 3,
        "date_start_crt": pd.NaT,
        "date_end_crt": pd.NaT,
        "date_op": pd.NaT,
        "date_endoscopy": pd.NaT,
        "date_progress": pd.NaT,
        "date_local_failure": pd.NaT,
        "date_death": pd.NaT,
        "date_ct_post_crt": pd.NaT,
        "last_fu_date": pd.NaT,
        "last_date_alive": pd.NaT,
        "progress": pd.NA,
        "local_failure": pd.NA,
        "death": pd.NA,
        "pcr": pd.NA,
        "becker_grade": None,
        "age_at_crt": 65.0,
    }
    frame = pd.DataFrame([{**defaults, **row} for row in rows])
    for column in frame.columns:
        if column.startswith("date_") or column in {"last_fu_date", "last_date_alive"}:
            frame[column] = pd.to_datetime(frame[column])
    return frame


class TestCensoringDate:
    def test_takes_the_maximum_not_a_coalesce(self):
        # `last_date_alive` is sometimes later than `last_fu_date`, and each is
        # populated for a different subset. A coalesce in either direction would
        # discard the later contact and shorten follow-up.
        df = make_cohort(
            [
                {
                    "date_start_crt": "2022-01-01",
                    "last_fu_date": "2023-02-11",
                    "last_date_alive": "2023-05-16",
                    "progress": 0,
                    "death": 0,
                }
            ]
        )
        out = ep.derive(df)
        assert out.loc[0, "censor_date"] == pd.Timestamp("2023-05-16")
        assert out.loc[0, "censor_source"] == "last_date_alive"

    def test_uses_any_contact_when_the_usual_columns_are_empty(self):
        df = make_cohort(
            [{"date_start_crt": "2022-01-01", "date_ct_post_crt": "2022-09-30", "progress": 0, "death": 0}]
        )
        out = ep.derive(df)
        assert out.loc[0, "censor_date"] == pd.Timestamp("2022-09-30")


class TestPfsVersusTtp:
    def test_death_without_progression_is_a_pfs_event_but_not_a_ttp_event(self):
        # This single distinction is the whole 129-vs-86 event gap in the cohort.
        df = make_cohort(
            [
                {
                    "date_start_crt": "2022-01-01",
                    "date_death": "2022-07-01",
                    "last_fu_date": "2022-07-01",
                    "progress": 0,
                    "death": 1,
                }
            ]
        )
        out = ep.derive(df)
        assert out.loc[0, "pfs_event"] == 1
        assert out.loc[0, "pfs_time"] == 181
        assert out.loc[0, "ttp_event"] == 0
        # Censored for TTP, so its time runs to last contact rather than to death.
        assert out.loc[0, "ttp_time"] == 181

    def test_progression_before_death_times_pfs_at_progression(self):
        df = make_cohort(
            [
                {
                    "date_start_crt": "2022-01-01",
                    "date_progress": "2022-04-01",
                    "date_death": "2022-10-01",
                    "progress": 1,
                    "death": 1,
                }
            ]
        )
        out = ep.derive(df)
        assert out.loc[0, "pfs_event"] == 1
        assert out.loc[0, "pfs_time"] == 90  # progression, not death
        assert out.loc[0, "os_time"] == 273

    def test_neither_event_is_censored_at_last_contact(self):
        df = make_cohort(
            [{"date_start_crt": "2022-01-01", "last_fu_date": "2023-01-01", "progress": 0, "death": 0}]
        )
        out = ep.derive(df)
        assert out.loc[0, "pfs_event"] == 0
        assert out.loc[0, "pfs_time"] == 365


class TestCohortRestriction:
    def test_lffs_is_definitive_only(self):
        df = make_cohort(
            [
                {"cohort": "definitive", "patient_key": "def_001", "date_start_crt": "2022-01-01",
                 "last_fu_date": "2023-01-01", "progress": 0, "death": 0, "local_failure": 0},
                {"cohort": "neoadjuvant", "patient_key": "neo_001", "date_start_crt": "2022-01-01",
                 "last_fu_date": "2023-01-01", "progress": 0, "death": 0},
            ]
        )
        out = ep.derive(df)
        assert out.loc[0, "lffs_event"] == 0
        assert pd.isna(out.loc[1, "lffs_event"])

    def test_pcr_is_neoadjuvant_only(self):
        df = make_cohort(
            [
                {"cohort": "neoadjuvant", "patient_key": "neo_001", "date_start_crt": "2022-01-01",
                 "last_fu_date": "2023-01-01", "progress": 0, "death": 0, "pcr": 1},
                {"cohort": "definitive", "patient_key": "def_001", "date_start_crt": "2022-01-01",
                 "last_fu_date": "2023-01-01", "progress": 0, "death": 0, "pcr": 1},
            ]
        )
        out = ep.derive(df)
        assert out.loc[0, "pcr_event"] == 1
        assert pd.isna(out.loc[1, "pcr_event"])

    def test_becker_major_is_1a_or_1b(self):
        df = make_cohort(
            [
                {"cohort": "neoadjuvant", "patient_key": f"neo_{i:03d}", "date_start_crt": "2022-01-01",
                 "last_fu_date": "2023-01-01", "progress": 0, "death": 0, "becker_grade": grade}
                for i, grade in enumerate(["1a", "1b", "2", "3", None])
            ]
        )
        out = ep.derive(df)
        assert out["becker_major_event"].tolist()[:4] == [1, 1, 0, 0]
        assert pd.isna(out.loc[4, "becker_major_event"])


class TestTreatmentIntervals:
    def test_impossible_interval_is_nulled_and_flagged_not_corrected(self):
        # A year typo: CRT "ends" ten years before it starts. Endpoints anchor on
        # date_start_crt so they stay valid; only the interval is unusable.
        df = make_cohort(
            [
                {
                    "date_start_crt": "2024-11-07",
                    "date_end_crt": "2014-12-10",
                    "last_fu_date": "2025-01-01",
                    "progress": 0,
                    "death": 0,
                }
            ]
        )
        out = ep.derive(df)
        assert pd.isna(out.loc[0, "crt_duration_days"])
        assert "crt_duration_days" in out.loc[0, "date_anomaly"]
        assert out.loc[0, "pfs_time"] == 55  # unaffected

    def test_normal_interval_is_kept(self):
        df = make_cohort(
            [{"date_start_crt": "2022-01-01", "date_end_crt": "2022-02-10",
              "last_fu_date": "2023-01-01", "progress": 0, "death": 0}]
        )
        out = ep.derive(df)
        assert out.loc[0, "crt_duration_days"] == 40
        assert out.loc[0, "date_anomaly"] is None


class TestValidation:
    def _base(self, **overrides):
        row = {
            "date_start_crt": "2022-01-01",
            "last_fu_date": "2023-01-01",
            "progress": 0,
            "death": 0,
            "local_failure": 0,
        }
        row.update(overrides)
        return ep.derive(make_cohort([row]))

    def test_clean_cohort_passes(self):
        assert ep.validate(self._base(), strict=False) == []

    def test_event_without_a_date_is_caught(self):
        out = self._base(death=1)
        problems = ep.validate(out, strict=False)
        assert any("no date_death" in p for p in problems)

    def test_local_failure_without_progression_is_caught(self):
        out = self._base(local_failure=1, date_local_failure="2022-06-01", progress=0)
        assert any("local failure without progression" in p for p in ep.validate(out, strict=False))

    def test_pcr_disagreeing_with_ypt_ypn_is_caught(self):
        out = self._base()
        out["pcr"] = 1
        out["ypt_stage_num"] = 2
        out["ypn_stage_num"] = 0
        assert any("pCR disagrees" in p for p in ep.validate(out, strict=False))

    def test_missing_age_is_caught(self):
        out = self._base()
        out["age_at_crt"] = pd.NA
        assert any("age could not be resolved" in p for p in ep.validate(out, strict=False))

    def test_strict_mode_raises(self):
        with pytest.raises(ValueError, match="validation failure"):
            ep.validate(self._base(death=1), strict=True)


class TestLandmarkEndpoints:
    """dfs (anchored at surgery) and the Date-End-CRT sensitivity endpoints."""

    def test_dfs_is_measured_from_surgery_not_crt_start(self):
        df = make_cohort(
            [
                {
                    "cohort": "neoadjuvant",
                    "patient_key": "neo_001",
                    "date_start_crt": "2022-01-01",
                    "date_end_crt": "2022-02-15",
                    "date_op": "2022-04-01",
                    "date_death": "2022-10-01",
                    "progress": 0,
                    "death": 1,
                }
            ]
        )
        out = ep.derive(df)
        assert out.loc[0, "dfs_event"] == 1
        assert out.loc[0, "dfs_time"] == (
            pd.Timestamp("2022-10-01") - pd.Timestamp("2022-04-01")
        ).days
        # The CRT-anchored twin keeps its own origin.
        assert out.loc[0, "pfs_time"] == (
            pd.Timestamp("2022-10-01") - pd.Timestamp("2022-01-01")
        ).days

    def test_patient_without_surgery_has_no_dfs(self):
        # The immortal-time guard: no origin, no landmark endpoint.
        df = make_cohort(
            [
                {
                    "cohort": "neoadjuvant",
                    "patient_key": "neo_002",
                    "date_start_crt": "2022-01-01",
                    "date_death": "2022-10-01",
                    "last_fu_date": "2022-10-01",
                    "progress": 0,
                    "death": 1,
                }
            ]
        )
        out = ep.derive(df)
        assert pd.isna(out.loc[0, "dfs_event"])
        assert pd.isna(out.loc[0, "dfs_time"])
        assert out.loc[0, "pfs_event"] == 1  # the default-origin endpoint is unaffected

    def test_progression_before_surgery_is_excluded_not_clamped(self):
        # A patient who progressed before resection was never disease-free at
        # the landmark; counting them would re-introduce immortal-time bias.
        df = make_cohort(
            [
                {
                    "cohort": "neoadjuvant",
                    "patient_key": "neo_003",
                    "date_start_crt": "2022-01-01",
                    "date_op": "2022-04-01",
                    "date_progress": "2022-03-01",
                    "progress": 1,
                    "death": 0,
                }
            ]
        )
        out = ep.derive(df)
        assert pd.isna(out.loc[0, "dfs_event"])
        assert pd.isna(out.loc[0, "dfs_time"])

    def test_anomalous_origin_date_is_refused(self):
        # The neo_022 regression: date_op is a decade before date_end_crt, so
        # the pair is a suspected transcription error and dfs must not anchor
        # on it -- even though the resulting duration would pass the
        # negative-duration check.
        df = make_cohort(
            [
                {
                    "cohort": "neoadjuvant",
                    "patient_key": "neo_022",
                    "date_start_crt": "2024-09-01",
                    "date_end_crt": "2024-12-18",
                    "date_op": "2015-01-29",
                    "date_death": "2025-06-01",
                    "last_fu_date": "2025-06-01",
                    "progress": 0,
                    "death": 1,
                }
            ]
        )
        out = ep.derive(df)
        assert "date_op" in out.loc[0, "anomalous_date_columns"]
        assert pd.isna(out.loc[0, "dfs_event"])
        assert out.loc[0, "pfs_event"] == 1  # CRT-anchored endpoints unaffected

    def test_endcrt_origin_shifts_time_but_preserves_the_event(self):
        df = make_cohort(
            [
                {
                    "cohort": "definitive",
                    "patient_key": "def_010",
                    "date_start_crt": "2022-01-01",
                    "date_end_crt": "2022-02-10",
                    "date_progress": "2022-08-01",
                    "progress": 1,
                    "death": 0,
                }
            ]
        )
        out = ep.derive(df)
        assert out.loc[0, "pfs_endcrt_event"] == out.loc[0, "pfs_event"] == 1
        shift = (pd.Timestamp("2022-02-10") - pd.Timestamp("2022-01-01")).days
        assert out.loc[0, "pfs_time"] - out.loc[0, "pfs_endcrt_time"] == shift

    def test_death_during_crt_is_excluded_from_the_endcrt_endpoints(self):
        df = make_cohort(
            [
                {
                    "cohort": "definitive",
                    "patient_key": "def_011",
                    "date_start_crt": "2022-01-01",
                    "date_end_crt": "2022-02-10",
                    "date_death": "2022-02-01",
                    "progress": 0,
                    "death": 1,
                }
            ]
        )
        out = ep.derive(df)
        assert out.loc[0, "os_event"] == 1
        assert pd.isna(out.loc[0, "os_endcrt_event"])

    def test_labels_table_carries_the_new_endpoints(self):
        df = make_cohort(
            [
                {
                    "cohort": "neoadjuvant",
                    "patient_key": "neo_004",
                    "date_start_crt": "2022-01-01",
                    "date_op": "2022-04-01",
                    "last_fu_date": "2023-04-01",
                    "progress": 0,
                    "death": 0,
                }
            ]
        )
        table = ep.labels_table(ep.derive(df))
        for column in (
            "dfs_event",
            "dfs_time",
            "pfs_endcrt_event",
            "pfs_endcrt_time",
            "os_endcrt_event",
            "os_endcrt_time",
        ):
            assert column in table.columns

    def test_validate_rejects_a_dfs_value_without_a_surgery_date(self):
        df = make_cohort(
            [
                {
                    "cohort": "neoadjuvant",
                    "patient_key": "neo_005",
                    "date_start_crt": "2022-01-01",
                    "date_op": "2022-04-01",
                    "last_fu_date": "2023-04-01",
                    "progress": 0,
                    "death": 0,
                }
            ]
        )
        out = ep.derive(df)
        out.loc[0, "date_op"] = pd.NaT  # corrupt after derivation
        with pytest.raises(ValueError, match="dfs"):
            ep.validate(out, strict=True)
