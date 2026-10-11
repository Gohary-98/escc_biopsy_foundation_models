"""Manifest tests.

The manifest is the only link between a slide and a patient -- there is no slide
identifier in the clinical workbooks -- so a parsing or reconciliation bug here
mislabels the training data rather than crashing. These tests use fake files so
they run without the WSI sample present.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ec_wsi_2026 import manifest as mf


class TestFolderNameParsing:
    @pytest.mark.parametrize(
        ("name", "institution", "cohort", "patient_no"),
        [
            ("tum_def_1", "TUM", "definitive", 1),
            ("lmu_neo_58", "LMU", "neoadjuvant", 58),
            ("erl_def_96", "Erlangen", "definitive", 96),
            ("wue_neo_70", "Wuerzburg", "neoadjuvant", 70),
            ("TUM_DEF_7", "TUM", "definitive", 7),
        ],
    )
    def test_valid(self, name, institution, cohort, patient_no):
        parsed = mf.parse_folder_name(name)
        assert parsed["institution_from_folder"] == institution
        assert parsed["cohort"] == cohort
        assert parsed["patient_no"] == patient_no

    @pytest.mark.parametrize("name", ["random_folder", "tum_1", "tum_xyz_1", "", "tum_def_"])
    def test_invalid_returns_none(self, name):
        assert mf.parse_folder_name(name) is None

    def test_leading_zeros_are_normalised(self):
        # def_096 must key the same patient whether the folder says 96 or 096.
        assert mf.parse_folder_name("erl_def_096")["patient_no"] == 96


class TestDiscovery:
    def test_empty_directory_becomes_a_row(self, tmp_path):
        # A folder with no slide is a failed transfer. Skipping it silently is
        # how a missing patient goes unnoticed.
        (tmp_path / "erl_neo_77").mkdir()
        frame = mf.discover_slides(tmp_path, read_properties=False)
        assert len(frame) == 1
        assert frame.iloc[0]["status"] == "missing_file"
        assert frame.iloc[0]["file_name"] is None
        assert frame.iloc[0]["patient_no"] == 77

    def test_slide_id_includes_the_folder(self, tmp_path):
        # tum_def_1/1.svs and tum_neo_1/1.svs have identical file names.
        for folder in ("tum_def_1", "tum_neo_1"):
            directory = tmp_path / folder
            directory.mkdir()
            (directory / "1.svs").write_bytes(b"not a real slide")
        frame = mf.discover_slides(tmp_path, read_properties=False)
        assert sorted(frame["slide_id"]) == ["tum_def_1__1", "tum_neo_1__1"]

    def test_one_specimen_in_two_formats_shares_a_slide_id_but_not_a_file_id(self, tmp_path):
        # The shared slide_id is what makes the duplicate detectable; file_id
        # stays unique so it can key a row.
        directory = tmp_path / "erl_def_96"
        directory.mkdir()
        (directory / "96.ndpi").write_bytes(b"x")
        (directory / "96.tif").write_bytes(b"x")
        frame = mf.discover_slides(tmp_path, read_properties=False)
        assert frame["slide_id"].nunique() == 1
        assert frame["file_id"].is_unique

    def test_non_slide_files_are_ignored(self, tmp_path):
        directory = tmp_path / "tum_def_1"
        directory.mkdir()
        (directory / "1.svs").write_bytes(b"x")
        (directory / "notes.txt").write_text("ignore me")
        (directory / "thumbnail.jpg").write_bytes(b"x")
        frame = mf.discover_slides(tmp_path, read_properties=False)
        assert len(frame) == 1


class TestStatusAssignment:
    def _frame(self, rows):
        base = {
            "folder": "erl_def_96",
            "readable": True,
            "status": None,
            "primary": False,
        }
        return pd.DataFrame([{**base, **r} for r in rows])

    def test_vendor_native_wins_over_a_re_export(self):
        # 96.ndpi and 96.tif are one specimen in two formats. The vendor file
        # keeps the full pyramid and the acquisition metadata.
        frame = self._frame(
            [
                {"file_name": "96.tif", "file_format": ".tif", "readable": False},
                {"file_name": "96.ndpi", "file_format": ".ndpi", "readable": True},
            ]
        )
        out = mf.assign_status(frame)
        native = out[out["file_format"] == ".ndpi"].iloc[0]
        export = out[out["file_format"] == ".tif"].iloc[0]
        assert native["status"] == "ok" and native["primary"]
        assert export["status"] == "duplicate_conversion" and not export["primary"]

    def test_preference_applies_even_when_both_are_readable(self):
        frame = self._frame(
            [
                {"file_name": "96.tif", "file_format": ".tif", "readable": True},
                {"file_name": "96.ndpi", "file_format": ".ndpi", "readable": True},
            ]
        )
        out = mf.assign_status(frame)
        assert out[out["primary"]].iloc[0]["file_format"] == ".ndpi"

    def test_a_lone_unreadable_file_is_unreadable_not_duplicate(self):
        frame = self._frame([{"file_name": "1.svs", "file_format": ".svs", "readable": False}])
        out = mf.assign_status(frame)
        assert out.iloc[0]["status"] == "unreadable"
        assert not out.iloc[0]["primary"]

    def test_distinct_stems_are_distinct_specimens(self):
        # Two blocks from one patient are two slides, not a duplicate pair.
        frame = self._frame(
            [
                {"file_name": "96_A.svs", "file_format": ".svs", "readable": True},
                {"file_name": "96_B.svs", "file_format": ".svs", "readable": True},
            ]
        )
        out = mf.assign_status(frame)
        assert int(out["primary"].sum()) == 2


class TestReconciliation:
    def _clinical(self):
        return pd.DataFrame(
            [
                {"patient_key": "def_096", "institution": "Erlangen"},
                {"patient_key": "neo_058", "institution": "LMU"},
            ]
        )

    def _manifest(self, rows):
        base = {"status": "ok", "primary": True, "file_name": "x.svs"}
        return pd.DataFrame([{**base, **r} for r in rows])

    def test_patient_key_and_institution_agree(self):
        frame = self._manifest(
            [{"cohort": "definitive", "patient_no": 96, "institution_from_folder": "Erlangen"}]
        )
        out = mf.reconcile_with_clinical(frame, self._clinical())
        assert out.iloc[0]["patient_key"] == "def_096"
        assert out.iloc[0]["institution_match"] == True
        assert out.iloc[0]["matched_clinical"]

    def test_institution_disagreement_is_flagged(self):
        # A slide filed under the wrong patient would be silently mislabelled.
        frame = self._manifest(
            [{"cohort": "definitive", "patient_no": 96, "institution_from_folder": "LMU"}]
        )
        out = mf.reconcile_with_clinical(frame, self._clinical())
        assert out.iloc[0]["institution_match"] == False

    def test_unmatched_slide_has_no_institution_verdict(self):
        # Not comparable is not the same as disagreeing; NA must not read as False.
        frame = self._manifest(
            [{"cohort": "definitive", "patient_no": 999, "institution_from_folder": "TUM"}]
        )
        out = mf.reconcile_with_clinical(frame, self._clinical())
        assert pd.isna(out.iloc[0]["institution_match"])

    def test_slide_with_no_clinical_record_is_marked_not_dropped(self):
        frame = self._manifest(
            [{"cohort": "definitive", "patient_no": 999, "institution_from_folder": "TUM"}]
        )
        out = mf.reconcile_with_clinical(frame, self._clinical())
        assert len(out) == 1  # kept, so it cannot go unnoticed
        assert out.iloc[0]["status"] == "unmatched"
        assert not out.iloc[0]["primary"]

    def test_multiple_primaries_get_distinct_slide_indices(self):
        # The multi-slide contract: the moment a patient has two primary slides,
        # bag construction must concatenate them, never average — so each slide
        # needs a stable index, not a collapsed identity.
        frame = self._manifest(
            [
                {
                    "cohort": "definitive",
                    "patient_no": 96,
                    "institution_from_folder": "Erlangen",
                    "file_name": "96_A.svs",
                },
                {
                    "cohort": "definitive",
                    "patient_no": 96,
                    "institution_from_folder": "Erlangen",
                    "file_name": "96_B.svs",
                },
            ]
        )
        out = mf.reconcile_with_clinical(frame, self._clinical())
        indices = sorted(out.loc[out["primary"], "slide_index"].tolist())
        assert indices == [0, 1]

    def test_single_primary_gets_index_zero(self):
        frame = self._manifest(
            [{"cohort": "definitive", "patient_no": 96, "institution_from_folder": "Erlangen"}]
        )
        out = mf.reconcile_with_clinical(frame, self._clinical())
        assert out.loc[out["primary"], "slide_index"].tolist() == [0]


class TestStandardisedMeanDifference:
    def test_identical_groups(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0])
        assert mf.standardised_mean_difference(s, s) == pytest.approx(0.0)

    def test_shifted_groups(self):
        a = pd.Series([1.0, 2.0, 3.0, 4.0])
        assert mf.standardised_mean_difference(a, a + 10) < -1

    def test_too_few_observations(self):
        import math

        assert math.isnan(mf.standardised_mean_difference(pd.Series([1.0]), pd.Series([2.0, 3.0])))

    def test_zero_variance(self):
        a = pd.Series([2.0, 2.0, 2.0])
        assert mf.standardised_mean_difference(a, a) == 0.0
