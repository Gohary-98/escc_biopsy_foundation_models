"""Metric correctness and the anti-leakage guards of the clinical Cox fit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ec_wsi_2026.modeling import evaluation as ev
from ec_wsi_2026.modeling.survival import fit_cox_fold


class TestSurvivalMetrics:
    def test_harrell_c_agrees_with_lifelines(self):
        lifelines_utils = pytest.importorskip("lifelines.utils")
        rng = np.random.default_rng(3)
        n = 200
        # Continuous times: tied observed times are handled by convention-specific
        # rules that differ between implementations, and this cohort's derived
        # day-resolution times rarely tie.
        time = rng.uniform(10, 2000, n)
        event = rng.integers(0, 2, n)
        risk = rng.normal(size=n)
        ours, pairs = ev.harrell_c(time, event, risk)
        # lifelines scores concordance with survival times, so higher score =
        # longer survival; negate the risk to match its convention.
        theirs = lifelines_utils.concordance_index(time, -risk, event)
        assert ours == pytest.approx(theirs, abs=1e-12)
        assert pairs > 0

    def test_harrell_c_is_chance_for_a_random_score(self):
        rng = np.random.default_rng(2)
        n = 400
        time = rng.uniform(10, 2000, n)
        event = (rng.random(n) < 0.7).astype(int)
        c, _ = ev.harrell_c(time, event, rng.normal(size=n))
        assert 0.45 <= c <= 0.55

    def test_rank_normal_preserves_order_and_is_centred(self):
        x = np.array([5.0, -1.0, 3.0, 100.0])
        z = ev.rank_normal(x)
        assert (np.argsort(z) == np.argsort(x)).all()
        assert z.sum() == pytest.approx(0.0, abs=1e-12)

    def test_ipcw_auc_separates_an_informative_score(self):
        pytest.importorskip("sksurv")
        rng = np.random.default_rng(4)
        n = 300
        risk = rng.normal(size=n)
        time = np.exp(3 - risk + rng.normal(0, 0.4, n)) * 30
        event = (rng.random(n) < 0.7).astype(int)
        frame = pd.DataFrame({"time": time, "event": event})
        horizon = float(np.median(time))
        assert ev.ipcw_auc_pooled(frame, risk, horizon) > 0.75
        assert ev.ipcw_auc_pooled(frame, rng.normal(size=n), horizon) == pytest.approx(0.5, abs=0.08)


class TestCoxFoldDiscipline:
    def test_constant_column_is_dropped_not_fatal(self):
        # A covariate constant within the training rows would NaN lifelines'
        # internal normalisation; the pipeline must drop it, not crash. This is
        # the imaged-subset regression: a near-constant flag became exactly
        # constant after cohort restriction.
        pytest.importorskip("lifelines")
        rng = np.random.default_rng(8)
        n = 120
        X = pd.DataFrame(
            {"constant": np.ones(n), "signal": rng.normal(size=n)},
            index=[f"pt_{i}" for i in range(n)],
        )
        y = pd.DataFrame(
            {
                "time": np.exp(3 - X["signal"].to_numpy() + rng.normal(0, 0.3, n)) * 30,
                "event": (rng.random(n) < 0.7).astype(int),
            },
            index=X.index,
        )
        fitted = fit_cox_fold(X, y, seed=0)
        assert len(fitted["model"].params_) == 1  # the constant was dropped

    def test_imputer_learns_train_medians_only(self):
        pytest.importorskip("lifelines")
        rng = np.random.default_rng(7)
        n = 120
        X = pd.DataFrame(
            {"a": rng.normal(10, 2, n), "b": rng.normal(0, 1, n)},
            index=[f"pt_{i}" for i in range(n)],
        )
        X.iloc[:10, 0] = np.nan
        risk = X["b"].to_numpy()
        y = pd.DataFrame(
            {
                "time": np.exp(3 - risk + rng.normal(0, 0.3, n)) * 30,
                "event": (rng.random(n) < 0.7).astype(int),
            },
            index=X.index,
        )
        fitted = fit_cox_fold(X, y, seed=0)
        assert fitted["penalizer"] in (0.01, 0.1, 1.0, 10.0)
        imputer = fitted["preprocessor"].named_steps["impute"]
        assert imputer.statistics_[0] == pytest.approx(X["a"].median())
