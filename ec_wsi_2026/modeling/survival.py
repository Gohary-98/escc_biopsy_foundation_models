"""Penalised Cox fit of the clinical model (lifelines).

Ridge Cox (``CoxPHFitter(penalizer=lambda, l1_ratio=0)``): at these event
budgets the covariates are pre-specified (references/covariates.yaml), so there
is no variable-selection ambition to motivate lasso.

Leakage discipline:

- imputation (median) and scaling are fit on training rows only — including
  inside every inner fold, so hyperparameter selection never sees a validation
  statistic;
- the penalizer is chosen by an inner stratified CV *within* each outer
  training fold, scored by inner-validation Harrell C, then the model is refit
  on the full outer-training fold at the winner.
"""

from __future__ import annotations

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ec_wsi_2026.modeling.evaluation import harrell_c

PENALIZER_GRID = (0.01, 0.1, 1.0, 10.0)
# Splits of the inner stratified CV that selects the hyperparameters.
INNER_SPLITS = 5


def _preprocessor() -> Pipeline:
    # VarianceThreshold drops columns that are constant within the training rows:
    # lifelines re-normalises internally, and a zero-variance column would turn
    # into NaN. A constant carries no information for that fold's model anyway.
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("drop_constant", VarianceThreshold()),
        ]
    )


def _fit_cox(train_frame: pd.DataFrame, penalizer: float):
    """Fit lifelines CoxPHFitter on an already-transformed training frame."""
    from lifelines import CoxPHFitter

    fitter = CoxPHFitter(penalizer=penalizer, l1_ratio=0.0)
    fitter.fit(train_frame, duration_col="time", event_col="event")
    return fitter


def _transform(preprocessor: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    # Generic column names: VarianceThreshold may have dropped columns, so the
    # output width need not match X. Train and validation go through the same
    # fitted pipeline, so the names stay consistent where it matters.
    matrix = preprocessor.transform(X.astype(float))
    return pd.DataFrame(
        matrix, columns=[f"x{i:03d}" for i in range(matrix.shape[1])], index=X.index
    )


def fit_cox_fold(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    *,
    penalizers: tuple[float, ...] = PENALIZER_GRID,
    seed: int = 0,
) -> dict:
    """Inner-CV penalizer selection, then refit on the full outer-training fold.

    ``y_train`` needs ``time`` and ``event`` columns aligned with ``X_train``.
    Returns ``{"preprocessor", "model", "penalizer"}``.
    """
    events = y_train["event"].to_numpy(dtype=int)
    splitter = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed)

    scores = {p: [] for p in penalizers}
    for inner_train_idx, inner_val_idx in splitter.split(X_train, events):
        inner_prep = _preprocessor().fit(X_train.iloc[inner_train_idx].astype(float))
        inner_train = _transform(inner_prep, X_train.iloc[inner_train_idx])
        inner_train[["time", "event"]] = y_train.iloc[inner_train_idx][["time", "event"]].values
        inner_val_X = _transform(inner_prep, X_train.iloc[inner_val_idx])
        inner_val_y = y_train.iloc[inner_val_idx]
        for penalizer in penalizers:
            try:
                fitter = _fit_cox(inner_train, penalizer)
                risk = fitter.predict_partial_hazard(inner_val_X).to_numpy()
                c, pairs = harrell_c(
                    inner_val_y["time"].to_numpy(), inner_val_y["event"].to_numpy(), risk
                )
                scores[penalizer].append(c if pairs else np.nan)
            # lifelines signals convergence failure with several exception
            # types; any of them just means this penalizer is unusable here.
            except Exception as error:  # noqa: BLE001
                logger.debug(f"inner Cox fit failed at penalizer={penalizer}: {error}")
                scores[penalizer].append(np.nan)

    mean_scores = {p: np.nanmean(vals) if len(vals) else np.nan for p, vals in scores.items()}
    # Tie-break (and the all-failed fallback) toward the LARGEST penalizer:
    # when the inner CV cannot distinguish candidates, the most shrinkage is
    # the safest default, not the least.
    best = max(
        penalizers,
        key=lambda p: ((-np.inf if np.isnan(mean_scores[p]) else mean_scores[p]), p),
    )

    preprocessor = _preprocessor().fit(X_train.astype(float))
    train_frame = _transform(preprocessor, X_train)
    train_frame[["time", "event"]] = y_train[["time", "event"]].values
    model = _fit_cox(train_frame, best)
    return {"preprocessor": preprocessor, "model": model, "penalizer": best}
