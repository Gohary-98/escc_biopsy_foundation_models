"""Fold-construction tests: leakage, coverage, pinning, and staleness.

The folds are the artifact every model — clinical and imaging — trains on, so
this is where a silent bug would corrupt every downstream comparison at once.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ec_wsi_2026.modeling import folds as fm


class TestStratified:
    def test_every_rotating_patient_validates_exactly_once_per_seed(self, make_labels):
        labels = make_labels()
        folds = fm.build_stratified(labels, "pfs")
        rotating = folds[folds["role"] == "rotating"]
        per_seed = rotating.groupby("seed")["patient_key"].nunique()
        assert (per_seed == 100).all()  # 110 patients minus 10 pinned Wuerzburg
        # one row per (patient, seed) means exactly one validation fold each
        assert not rotating.duplicated(["seed", "patient_key"]).any()

    def test_wuerzburg_never_appears_in_a_validation_fold(self, make_labels):
        labels = make_labels()
        folds = fm.build_stratified(labels, "pfs")
        wue_keys = set(labels.loc[labels["institution"] == "Wuerzburg", "patient_key"])
        val_keys = set(folds.loc[folds["fold"].notna(), "patient_key"])
        assert not (wue_keys & val_keys)
        assert (folds.loc[folds["patient_key"].isin(wue_keys), "role"] == "train_only").all()

    def test_wuerzburg_joins_every_training_split(self, make_labels):
        labels = make_labels()
        folds = fm.build_stratified(labels, "pfs")
        wue_keys = set(labels.loc[labels["institution"] == "Wuerzburg", "patient_key"])
        seed = folds["seed"].iloc[0]
        for fold in folds.loc[folds["fold"].notna(), "fold"].unique():
            train, val = fm.train_val_split(folds, seed, fold)
            assert wue_keys <= set(train)
            assert not (wue_keys & set(val))

    def test_identical_patient_sets_across_seeds(self, make_labels):
        labels = make_labels()
        folds = fm.build_stratified(labels, "pfs")
        sets = folds.groupby("seed")["patient_key"].apply(frozenset).unique()
        assert len(sets) == 1

    def test_no_train_val_overlap(self, make_labels):
        labels = make_labels()
        folds = fm.build_stratified(labels, "pfs")
        for seed in folds["seed"].unique():
            for fold in folds.loc[folds["fold"].notna(), "fold"].unique():
                train, val = fm.train_val_split(folds, seed, fold)
                assert not (set(train) & set(val))

    def test_assert_folds_catches_manufactured_leakage(self, make_labels):
        labels = make_labels()
        folds = fm.build_stratified(labels, "pfs")
        corrupted = pd.concat([folds, folds.iloc[[0]]], ignore_index=True)
        with pytest.raises(AssertionError):
            fm.assert_folds(corrupted, labels)

    def test_stratum_coarsening_falls_back_when_a_stratum_is_tiny(self, make_labels):
        # Two patients at one site cannot fill five folds under the full
        # institution stratification; the chain must coarsen, not crash.
        labels = make_labels({"TUM": 40, "LMU": 30, "Erlangen": 2, "Wuerzburg": 6})
        folds = fm.build_stratified(labels, "pfs")
        fm.assert_folds(folds, labels)


class TestPersistence:
    def test_round_trip_and_staleness_guard(self, make_labels, tmp_path):
        labels = make_labels()
        labels_path = tmp_path / "labels.parquet"
        labels.to_parquet(labels_path, index=False)
        folds_path = tmp_path / "folds.csv"
        manifest_path = tmp_path / "folds_manifest.json"

        fm.write_folds(
            labels, endpoints=("pfs",), folds_path=folds_path, manifest_path=manifest_path
        )
        loaded, sha = fm.load_folds(
            "pfs",
            folds_path=folds_path,
            manifest_path=manifest_path,
            labels_path=labels_path,
        )
        assert len(loaded) and len(sha) == 64

        # Drop one patient: the cohort hash changes and the folds must refuse.
        labels.iloc[1:].to_parquet(labels_path, index=False)
        with pytest.raises(ValueError, match="stale"):
            fm.load_folds(
                "pfs",
                folds_path=folds_path,
                manifest_path=manifest_path,
                labels_path=labels_path,
            )
