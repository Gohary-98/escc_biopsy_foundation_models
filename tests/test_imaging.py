"""Tests for consolidating TRIDENT's raw per-patient slide-embedding output.

Synthetic .h5 files stand in for TRIDENT's real ``slide_features_<encoder>/``
output (one ``features`` dataset per patient, keyed by filename stem).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from ec_wsi_2026.modeling import imaging as im


def _write_embedding(path: Path, values: list[float]) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=np.array(values, dtype="float32"))
        handle.create_dataset("coords", data=np.zeros((3, 2), dtype="int64"))


class TestConsolidateSlideEmbeddings:
    def test_reads_one_row_per_patient_with_named_dimension_columns(self, tmp_path):
        _write_embedding(tmp_path / "def_001.h5", [1.0, 2.0, 3.0])
        _write_embedding(tmp_path / "neo_002.h5", [4.0, 5.0, 6.0])

        table = im.consolidate_slide_embeddings(tmp_path, prefix="titan")

        assert sorted(table["patient_key"]) == ["def_001", "neo_002"]
        assert list(table.columns) == ["patient_key", "titan_000", "titan_001", "titan_002"]
        row = table.set_index("patient_key").loc["def_001"]
        assert row.tolist() == [1.0, 2.0, 3.0]

    def test_inconsistent_dimension_raises(self, tmp_path):
        # A patient with a different embedding length means the run itself was
        # inconsistent (e.g. two different runs' outputs got mixed together) --
        # this must fail loudly, not silently pad or truncate.
        _write_embedding(tmp_path / "def_001.h5", [1.0, 2.0, 3.0])
        _write_embedding(tmp_path / "def_002.h5", [1.0, 2.0])
        with pytest.raises(ValueError, match="embedding dim"):
            im.consolidate_slide_embeddings(tmp_path, prefix="titan")

    def test_no_h5_files_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            im.consolidate_slide_embeddings(tmp_path, prefix="titan")

    def test_missing_and_extra_patients_are_tolerated_not_raised(self, tmp_path):
        # A handful of failed extractions is expected occasionally; the point
        # is to name the discrepancy (logged), not to block the whole run.
        _write_embedding(tmp_path / "def_001.h5", [1.0])
        table = im.consolidate_slide_embeddings(
            tmp_path, prefix="titan", expected_patient_keys={"def_001", "def_002"}
        )
        assert list(table["patient_key"]) == ["def_001"]


class TestFindSlideFeaturesDir:
    def test_finds_nested_directory_by_encoder_name(self, tmp_path):
        target = tmp_path / "20x_512px_0px_overlap" / "slide_features_titan"
        target.mkdir(parents=True)
        (target / "def_001.h5").touch()

        found = im.find_slide_features_dir(tmp_path, encoder="titan")
        assert found == target

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            im.find_slide_features_dir(tmp_path, encoder="titan")
