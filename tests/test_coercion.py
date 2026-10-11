"""Coercion tests, built from the exact dirty values present in the workbooks.

Each case here is a real value that appeared in `definitive_clean.xlsx` or
`neoadjuvant_clean.xlsx`. They are pinned as tests because the failure mode they
guard against is silent: pandas' type inference turns every one of them into NaN
without complaint, and a dropped substage or a mis-parsed date does not announce
itself downstream -- it just shifts a survival curve.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ec_wsi_2026.clinical import coercion as co


class TestNullSentinels:
    @pytest.mark.parametrize(
        "value", ["", " ", "-", "--", "/", "?", "NA", "na", "n.a.", "none", None, float("nan")]
    )
    def test_recognised_as_null(self, value):
        assert co.is_null(value)

    @pytest.mark.parametrize("value", ["0", 0, "1", "1a", "+", "Tis"])
    def test_real_values_are_not_null(self, value):
        # '0' is a meaningful code in death/progress/pCR and must survive.
        assert not co.is_null(value)


class TestDates:
    def test_iso(self):
        ts, note, err = co.coerce_date("2022-06-24")
        assert ts == pd.Timestamp("2022-06-24") and note is None and err is None

    def test_german_two_digit_year(self):
        ts, _, err = co.coerce_date("21.06.22")
        assert ts == pd.Timestamp("2022-06-21") and err is None

    def test_german_four_digit_year(self):
        ts, _, err = co.coerce_date("05.04.2024")
        assert ts == pd.Timestamp("2024-04-05") and err is None

    @pytest.mark.parametrize(
        ("value", "expected", "expected_note"),
        [
            ("21.06.22 (zusammen mit Koloskopie)", "2022-06-21", "zusammen mit Koloskopie"),
            ("05.04.2024 (PET)", "2024-04-05", "PET"),
            ("28.07.23 (extern)", "2023-07-28", "extern"),
            ("24.10.22 (PET)", "2022-10-24", "PET"),
        ],
    )
    def test_trailing_annotation_is_parsed_and_preserved(self, value, expected, expected_note):
        # The annotation records imaging provenance and is worth keeping, so the
        # parser matches a date *prefix* rather than requiring an exact match.
        ts, note, err = co.coerce_date(value)
        assert ts == pd.Timestamp(expected)
        assert note == expected_note
        assert err is None

    def test_day_first_not_month_first(self):
        # 13 cannot be a month, so this also proves the ordering is day-first.
        ts, _, _ = co.coerce_date("13.02.2024")
        assert ts == pd.Timestamp("2024-02-13")

    def test_null_yields_no_error(self):
        ts, note, err = co.coerce_date("NA")
        assert ts is None and note is None and err is None

    def test_garbage_is_reported_not_swallowed(self):
        ts, _, err = co.coerce_date("sometime in spring")
        assert ts is None and err is not None

    def test_impossible_date_is_rejected(self):
        ts, _, err = co.coerce_date("2024-02-31")
        assert ts is None and "invalid date" in err

    def test_future_date_is_rejected(self):
        future = pd.Timestamp.today() + pd.Timedelta(days=400)
        ts, _, err = co.coerce_date(future.strftime("%Y-%m-%d"))
        assert ts is None and err == "out of range"

    def test_birth_date_range_is_tighter(self):
        # A plausible treatment date is not a plausible birth date.
        assert co.coerce_birth_date("2024-01-01")[2] is not None
        assert co.coerce_birth_date("1941-12-14")[0] == pd.Timestamp("1941-12-14")

    def test_excel_serial(self):
        ts, _, err = co.coerce_date("44736")
        assert ts == pd.Timestamp("2022-06-24") and err is None


class TestStages:
    @pytest.mark.parametrize(
        ("value", "number", "suffix"),
        [("3", 3, None), ("1a", 1, "a"), ("1b", 1, "b"), ("4b", 4, "b"), ("0", 0, None)],
    )
    def test_substage_letter_is_kept(self, value, number, suffix):
        # ypT '1b' is not ypT 1: the substage carries prognostic meaning.
        num, sfx, positive, err = co.coerce_stage(value)
        assert (num, sfx, positive, err) == (number, suffix, False, None)

    def test_bare_plus_means_node_positive(self):
        num, _, positive, err = co.coerce_stage("+")
        assert num is None and positive is True and err is None

    def test_tis_has_no_numeric_stage(self):
        num, sfx, _, err = co.coerce_stage("Tis")
        assert num is None and sfx == "is" and err is None

    def test_null(self):
        assert co.coerce_stage("NA") == (None, None, False, None)

    def test_garbage_is_reported(self):
        assert co.coerce_stage("weird")[3] is not None


class TestNumbers:
    def test_plain(self):
        assert co.coerce_float("50.4") == (50.4, None)

    def test_german_decimal_comma(self):
        assert co.coerce_float("1,8") == (1.8, None)

    def test_unit_suffix_tolerated(self):
        assert co.coerce_float("50.4 Gy") == (50.4, None)

    def test_sentinel_is_null_not_zero(self):
        # 'Boost GD (Gy)' holds '-' for patients who received no boost. Reading
        # that as 0.0 would silently invent a dose of zero.
        assert co.coerce_float("-") == (None, None)

    def test_float_rendering_of_an_int(self):
        assert co.coerce_int("1.0") == (1, None)

    def test_non_integer_is_reported(self):
        assert co.coerce_int("1.5")[1] == "not an integer"


class TestBecker:
    @pytest.mark.parametrize(
        ("value", "label", "bound"), [("1a", "1a", 0.0), ("1b", "1b", 10.0), ("2", "2", 50.0), ("3", "3", 100.0)]
    )
    def test_grades(self, value, label, bound):
        assert co.coerce_becker(value) == (label, bound, None)

    def test_unknown_grade_is_reported(self):
        assert co.coerce_becker("4")[2] is not None


class TestBrachyboost:
    def test_fractionated(self):
        # 'Brachyboost (Gy)' is free text: '2 x 6' means 2 fractions of 6 Gy.
        assert co.coerce_brachyboost("2 x 6") == (2, 6.0, 12.0, None)

    def test_alternate_separator(self):
        assert co.coerce_brachyboost("2 X 7")[2] == 14.0

    def test_bare_total(self):
        fractions, per_fraction, total, err = co.coerce_brachyboost("10")
        assert fractions is None and per_fraction is None and total == 10.0 and err is None


class TestYearsBetween:
    def test_age(self):
        age = co.years_between(pd.Timestamp("1941-12-14"), pd.Timestamp("2022-07-18"))
        assert age == pytest.approx(80.6, abs=0.1)

    def test_missing_endpoint(self):
        assert co.years_between(None, pd.Timestamp("2022-07-18")) is None


class TestParseLog:
    def test_records_and_renders(self):
        log = co.ParseLog()
        assert len(log) == 0
        assert list(log.to_frame().columns) == ["cohort", "source_row", "column", "value", "reason"]
        log.record(cohort="definitive", row=7, column="date_death", value="??", reason="bad")
        assert len(log) == 1
        assert log.to_frame().iloc[0]["source_row"] == 7
