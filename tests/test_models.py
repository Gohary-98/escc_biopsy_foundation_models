"""Tests for the model-training machinery (``modeling/models.py``).

The two load-bearing pieces are fold restriction (the paired-comparison
guarantee: assignments preserved, nothing silently emptied) and the
fold-internal PCA Cox fit (dimensionality chosen without ever seeing a
validation row).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ec_wsi_2026.modeling import folds as fm
from ec_wsi_2026.modeling import models


def test_every_model_embedding_table_is_registered():
    # An ARM_SPECS entry naming a table that EMBEDDING_TABLES lacks would only
    # fail at training time; catch the drift here instead.
    for spec in models.ARM_SPECS.values():
        assert spec.get("table", "titan") in models.EMBEDDING_TABLES


def test_the_seven_paper_models():
    assert models.MODELS == (
        "clinical", "titan", "uni-mean", "conch-mean", "fusion", "fusion-uni", "fusion-conch"
    )
    for name, spec in models.ARM_SPECS.items():
        assert spec["clinical"] == (name == "clinical" or name.startswith("fusion"))


class TestRestrictFolds:
    def test_assignments_are_preserved_for_kept_patients(self, make_labels):
        labels = make_labels()
        folds = fm.build_stratified(labels, "pfs")
        kept = set(labels["patient_key"].iloc[:80])

        restricted = models.restrict_folds(folds, kept)

        assert set(restricted["patient_key"]) <= kept
        # A kept patient's (seed, fold) rows are byte-identical to before.
        merged = restricted.merge(
            folds, on=["patient_key", "seed"], suffixes=("_r", "_orig")
        )
        assert (merged["fold_r"].fillna("NA") == merged["fold_orig"].fillna("NA")).all()

    def test_no_leakage_after_restriction(self, make_labels):
        labels = make_labels()
        folds = fm.build_stratified(labels, "pfs")
        kept = set(labels["patient_key"].iloc[:80])
        restricted = models.restrict_folds(folds, kept)

        seed = restricted["seed"].iloc[0]
        for fold in restricted.loc[restricted["fold"].notna(), "fold"].unique():
            train, val = fm.train_val_split(restricted, seed, fold)
            assert not (set(train) & set(val))

    def test_emptied_validation_cell_raises(self, make_labels):
        labels = make_labels()
        folds = fm.build_stratified(labels, "pfs")
        # Keep only one rotating patient: most validation cells lose everyone.
        rotating = folds.loc[folds["role"] == "rotating", "patient_key"]
        with pytest.raises(ValueError, match="validation cell"):
            models.restrict_folds(folds, {rotating.iloc[0]})


def _synthetic_survival(n: int = 120, dims: int = 16, seed: int = 0):
    """Embeddings whose first direction drives the hazard; the rest is noise."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, dims)),
        index=[f"pt_{i:03d}" for i in range(n)],
        columns=[f"titan_{i:03d}" for i in range(dims)],
    )
    risk = X.iloc[:, 0].to_numpy()
    time = np.exp(3.0 - risk + rng.normal(0, 0.4, n)) * 30
    event = (rng.random(n) < 0.7).astype(int)
    y = pd.DataFrame({"time": time, "event": event}, index=X.index)
    return X, y


class TestFitEmbeddingCoxFold:
    def test_recovers_signal_and_reports_choices(self):
        pytest.importorskip("lifelines")
        X, y = _synthetic_survival()
        train, val = X.index[:90], X.index[90:]

        fitted = models.fit_embedding_cox_fold(
            X.loc[train],
            y.loc[train],
            clinical_columns=[],
            embedding_columns=list(X.columns),
            k_grid=(4,),
            penalizers=(0.1, 1.0),
            seed=0,
        )
        assert fitted["k"] == 4
        assert fitted["penalizer"] in (0.1, 1.0)

        val_matrix = models._to_frame(fitted["preprocessor"].transform(X.loc[val]), val)
        risk = fitted["model"].predict_partial_hazard(val_matrix).to_numpy()
        from ec_wsi_2026.modeling.evaluation import harrell_c

        c, pairs = harrell_c(
            y.loc[val, "time"].to_numpy(), y.loc[val, "event"].to_numpy(), risk
        )
        assert pairs > 0
        assert c > 0.6  # the signal sits in component 1; PCA(4) must find it

    def test_fusion_matrix_width_is_pca_plus_clinical(self):
        pytest.importorskip("lifelines")
        X, y = _synthetic_survival()
        X = X.copy()
        X["age"] = np.linspace(40, 80, len(X))

        fitted = models.fit_embedding_cox_fold(
            X,
            y,
            clinical_columns=["age"],
            embedding_columns=[c for c in X.columns if c.startswith("titan_")],
            k_grid=(4,),
            penalizers=(1.0,),
            seed=0,
        )
        transformed = fitted["preprocessor"].transform(X)
        assert transformed.shape[1] == 4 + 1  # PCA components + clinical column

    def test_pca_is_fit_on_the_given_rows_only(self):
        pytest.importorskip("lifelines")
        X, y = _synthetic_survival()
        train = X.index[:90]
        fitted = models.fit_embedding_cox_fold(
            X.loc[train],
            y.loc[train],
            clinical_columns=[],
            embedding_columns=list(X.columns),
            k_grid=(4,),
            penalizers=(1.0,),
            seed=0,
        )
        scaler = fitted["preprocessor"].named_transformers_["emb"].named_steps["scale"]
        np.testing.assert_allclose(
            scaler.mean_, X.loc[train].mean().to_numpy(), rtol=1e-10
        )

    def test_infeasible_k_falls_back_instead_of_crashing(self):
        # A k larger than the inner-training row count must lose the grid,
        # not kill the fold.
        pytest.importorskip("lifelines")
        X, y = _synthetic_survival(n=60)
        fitted = models.fit_embedding_cox_fold(
            X,
            y,
            clinical_columns=[],
            embedding_columns=list(X.columns),
            k_grid=(4, 500),
            penalizers=(1.0,),
            seed=0,
        )
        assert fitted["k"] == 4
