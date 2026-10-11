"""The seven Cox models of the paper, trained out of fold on identical folds.

Models (``ARM_SPECS``), each a ridge (L2) Cox proportional hazards model:

- ``clinical``: the seven clinical variables (``references/covariates.yaml``),
  median imputation and standardisation fitted on the training fold.
- ``titan``, ``uni-mean``, ``conch-mean`` (imaging only): one slide embedding
  per patient (TITAN; mean-pooled UNI2-h or CONCH v1.5 tile features),
  standardised and reduced by PCA to k components.
- ``fusion`` (TITAN), ``fusion-uni``, ``fusion-conch`` (early fusion): the
  clinical block and the embedding's principal components in one Cox model
  with one shared penalty. PCA is unsupervised, so a fusion model's components
  equal those of its imaging-only model; only the coefficients are joint.

Every model reads the same persisted folds (``folds.py``) restricted to the
patients with all three embeddings, so all comparisons are paired. Inside each
outer training fold a five-fold inner stratified CV
selects k in ``PCA_GRID`` and the penalty in ``PENALIZER_GRID`` by Harrell's C;
all preprocessing (imputation, scaling, PCA) is refit inside every inner and
outer training split.

Output per model and endpoint: ``models/runs/paper/<endpoint>_<model>_stratified/``
with ``oof.csv`` (one row per patient x repeat: ``time, event, risk``, and
``high_risk`` = risk above the training fold's median risk) and
``model_meta.json`` (the chosen k and penalty per fold).
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ec_wsi_2026.config import (
    CLINICAL_PARQUET,
    CONCH_MEAN_EMBEDDINGS_PARQUET,
    LABELS_PARQUET,
    RUNS_DIR,
    TITAN_EMBEDDINGS_PARQUET,
    UNI_MEAN_EMBEDDINGS_PARQUET,
)
from ec_wsi_2026.modeling import covariates as cov
from ec_wsi_2026.modeling import evaluation as ev
from ec_wsi_2026.modeling import folds as folds_mod
from ec_wsi_2026.modeling.runs import run_tag, write_run
from ec_wsi_2026.modeling.survival import INNER_SPLITS, PENALIZER_GRID, _fit_cox, fit_cox_fold
from ec_wsi_2026.modeling.survival import _transform as _clinical_transform

PCA_GRID = (8, 16, 32)
ENDPOINTS = ("pfs", "os")

# model name -> (uses clinical, uses embeddings, PCA grid, embedding table)
ARM_SPECS: dict[str, dict] = {
    "clinical": {"clinical": True, "embeddings": False, "k_grid": None},
    "titan": {"clinical": False, "embeddings": True, "k_grid": PCA_GRID, "table": "titan"},
    "uni-mean": {"clinical": False, "embeddings": True, "k_grid": PCA_GRID, "table": "uni-mean"},
    "conch-mean": {"clinical": False, "embeddings": True, "k_grid": PCA_GRID, "table": "conch-mean"},
    "fusion": {"clinical": True, "embeddings": True, "k_grid": PCA_GRID, "table": "titan"},
    "fusion-uni": {"clinical": True, "embeddings": True, "k_grid": PCA_GRID, "table": "uni-mean"},
    "fusion-conch": {"clinical": True, "embeddings": True, "k_grid": PCA_GRID, "table": "conch-mean"},
}
MODELS = tuple(ARM_SPECS)

EMBEDDING_TABLES: dict[str, tuple[Path, str]] = {
    "titan": (TITAN_EMBEDDINGS_PARQUET, "consolidate-titan"),
    "uni-mean": (UNI_MEAN_EMBEDDINGS_PARQUET, "consolidate-mean"),
    "conch-mean": (CONCH_MEAN_EMBEDDINGS_PARQUET, "consolidate-mean"),
}

RUNS_ROOT = RUNS_DIR / "paper"
SCHEME = folds_mod.SCHEME


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
def load_embeddings(
    path: Path = TITAN_EMBEDDINGS_PARQUET, *, hint: str = "consolidate-titan"
) -> pd.DataFrame:
    """A consolidated embedding table, indexed by patient_key."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `python -m ec_wsi_2026.modeling {hint}` on the machine "
            "holding the TRIDENT output first"
        )
    table = pd.read_parquet(path)
    return table.set_index("patient_key")


def restrict_folds(folds: pd.DataFrame, keys: set[str]) -> pd.DataFrame:
    """Filter the persisted folds down to *keys*, preserving assignments.

    Fold membership is untouched — a patient keeps the exact (seed, fold)
    assignment the persisted folds gave them — rows for patients outside
    *keys* are simply dropped from both train and validation sides. Raises if
    any (seed, fold) validation cell ends up empty, which would silently
    shrink the evaluation.
    """
    restricted = folds.loc[folds["patient_key"].isin(keys)].copy()
    dropped = folds["patient_key"].nunique() - restricted["patient_key"].nunique()
    if dropped:
        logger.info(f"fold restriction drops {dropped} patient(s) without an embedding")

    rotating = folds.loc[folds["fold"].notna()]
    for (seed, fold), cell in rotating.groupby(["seed", "fold"]):
        if not cell["patient_key"].isin(keys).any():
            raise ValueError(
                f"restriction empties validation cell seed={seed} fold={fold!r} - "
                "the imaged subset is too sparse for these folds"
            )
    return restricted


# --------------------------------------------------------------------------
# Embedding / fusion fits (fold-internal PCA)
# --------------------------------------------------------------------------
def _preprocessor(
    clinical_columns: list[str], embedding_columns: list[str], k: int, seed: int
) -> ColumnTransformer:
    """Fold-internal preprocessing of the clinical block and the embedding block."""
    blocks = []
    if clinical_columns:
        blocks.append(
            (
                "clin",
                # VarianceThreshold mirrors survival._preprocessor: a column
                # constant within the training rows would NaN lifelines'
                # internal normalisation and crash the fit.
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                        ("drop_constant", VarianceThreshold()),
                    ]
                ),
                clinical_columns,
            )
        )
    if embedding_columns:
        steps = [("scale", StandardScaler()), ("pca", PCA(n_components=k, random_state=seed))]
        blocks.append(("emb", Pipeline(steps), embedding_columns))
    return ColumnTransformer(blocks)


def _to_frame(matrix: np.ndarray, index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        matrix, index=index, columns=[f"x{i:03d}" for i in range(matrix.shape[1])]
    )


def fit_embedding_cox_fold(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    *,
    clinical_columns: list[str],
    embedding_columns: list[str],
    k_grid: tuple[int, ...] = PCA_GRID,
    penalizers: tuple[float, ...] = PENALIZER_GRID,
    seed: int = 0,
) -> dict:
    """Inner-CV selection of (PCA components, penalizer), then refit.

    The PCA is refit inside every inner training split, so the dimensionality
    never sees a validation row and sweeping k cannot inflate the out-of-fold
    estimate.
    """
    events = y_train["event"].to_numpy(dtype=int)
    splitter = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed)

    scores: dict[tuple[int, float], list[float]] = {(k, p): [] for k in k_grid for p in penalizers}
    for inner_train_idx, inner_val_idx in splitter.split(X_train, events):
        X_tr = X_train.iloc[inner_train_idx]
        X_va = X_train.iloc[inner_val_idx]
        y_tr = y_train.iloc[inner_train_idx]
        y_va = y_train.iloc[inner_val_idx]
        for k in k_grid:
            if embedding_columns and k >= min(len(X_tr), len(embedding_columns)):
                for p in penalizers:
                    scores[(k, p)].append(np.nan)
                continue
            prep = _preprocessor(clinical_columns, embedding_columns, k, seed).fit(X_tr)
            train_frame = _to_frame(prep.transform(X_tr), X_tr.index)
            train_frame[["time", "event"]] = y_tr[["time", "event"]].values
            val_matrix = _to_frame(prep.transform(X_va), X_va.index)
            for p in penalizers:
                try:
                    fitter = _fit_cox(train_frame, p)
                    risk = fitter.predict_partial_hazard(val_matrix).to_numpy()
                    c, pairs = ev.harrell_c(
                        y_va["time"].to_numpy(), y_va["event"].to_numpy(), risk
                    )
                    scores[(k, p)].append(c if pairs else np.nan)
                # lifelines raises assorted exception types on convergence
                # failure; all of them just mean this (k, p) is unusable here.
                except Exception as error:  # noqa: BLE001
                    logger.debug(f"inner fit failed at k={k} penalizer={p}: {error}")
                    scores[(k, p)].append(np.nan)

    means = {
        kp: (np.mean(finite) if (finite := [v for v in vals if not np.isnan(v)]) else np.nan)
        for kp, vals in scores.items()
    }
    # Ties and the all-failed fallback resolve toward more shrinkage / fewer
    # components — the safest defaults when the inner CV cannot distinguish.
    best_k, best_p = max(
        means,
        key=lambda kp: ((-np.inf if np.isnan(means[kp]) else means[kp]), -kp[0], kp[1]),
    )

    preprocessor = _preprocessor(clinical_columns, embedding_columns, best_k, seed).fit(X_train)
    train_frame = _to_frame(preprocessor.transform(X_train), X_train.index)
    train_frame[["time", "event"]] = y_train[["time", "event"]].values
    model = _fit_cox(train_frame, best_p)
    return {"preprocessor": preprocessor, "model": model, "k": best_k, "penalizer": best_p}


# --------------------------------------------------------------------------
# The out-of-fold loop, one model at a time
# --------------------------------------------------------------------------
def _load_cohort(endpoint: str):
    spec = cov.load_spec()
    entry = spec["sets"].get(endpoint)
    if entry is None:
        raise ValueError(f"{endpoint!r} has no covariate set in covariates.yaml")
    clinical = pd.read_parquet(CLINICAL_PARQUET).set_index("patient_key")
    labels = pd.read_parquet(LABELS_PARQUET)
    embeddings = load_embeddings()
    eligible = folds_mod.eligible_labels(labels, endpoint).set_index("patient_key")
    keys = sorted(set(eligible.index) & set(embeddings.index))
    return entry, clinical, eligible, embeddings, keys


def run_arm(endpoint: str, arm: str, *, root: Path | None = None) -> str:
    """Train one model out of fold on the imaged subset; returns the run tag."""
    if arm not in ARM_SPECS:
        raise ValueError(f"unknown model {arm!r}; expected one of {MODELS}")
    root = RUNS_ROOT if root is None else root
    arm_spec = ARM_SPECS[arm]
    entry, clinical, eligible, embeddings, keys = _load_cohort(endpoint)

    # Models on a non-TITAN embedding table swap the features but keep the
    # cohort anchored to the TITAN cohort; a table missing a few of those
    # patients shrinks the model's cohort with a warning.
    table = arm_spec.get("table", "titan")
    if table != "titan":
        table_path, hint = EMBEDDING_TABLES[table]
        embeddings = load_embeddings(table_path, hint=hint)
        paired = sorted(set(keys) & set(embeddings.index))
        if len(paired) < len(keys):
            logger.warning(
                f"{arm}: {len(keys) - len(paired)} patient(s) missing from {table_path.name}"
            )
        keys = paired

    folds, cohort_sha = folds_mod.load_folds(endpoint)
    logger.info(
        f"{endpoint}/{arm}: {len(keys)} imaged eligible patient(s) of {len(eligible)} eligible overall"
    )
    folds_r = restrict_folds(folds, set(keys))

    truth = eligible.loc[keys, [f"{endpoint}_time", f"{endpoint}_event"]].astype(float)
    truth.columns = ["time", "event"]

    clinical_columns = list(entry["covariates"]) if arm_spec["clinical"] else []
    embedding_columns = list(embeddings.columns) if arm_spec["embeddings"] else []
    parts = []
    if clinical_columns:
        parts.append(cov.design_matrix(clinical.loc[keys], clinical_columns))
    if embedding_columns:
        parts.append(embeddings.loc[keys].astype(float))
    X_all = pd.concat(parts, axis=1)

    records: list[pd.DataFrame] = []
    chosen: dict[str, dict] = {}
    for seed in sorted(folds_r["seed"].unique()):
        at_seed = folds_r.loc[folds_r["seed"] == seed]
        for fold in sorted(at_seed.loc[at_seed["fold"].notna(), "fold"].unique()):
            train_keys, val_keys = folds_mod.train_val_split(folds_r, seed, fold)
            prediction, choice = _fit_predict_cox(
                arm,
                X_all.loc[train_keys],
                truth.loc[train_keys],
                X_all.loc[val_keys],
                clinical_columns,
                embedding_columns,
                arm_spec,
                int(seed),
            )
            chosen[f"seed{seed}_fold{fold}"] = choice
            records.append(
                pd.DataFrame(
                    {
                        "patient_key": val_keys,
                        "endpoint": endpoint,
                        "scheme": SCHEME,
                        "seed": int(seed),
                        "fold": fold,
                        "time": truth.loc[val_keys, "time"].to_numpy(),
                        "event": truth.loc[val_keys, "event"].to_numpy(dtype=int),
                        "risk": prediction,
                        # prospective risk group: above the training fold's median risk
                        "high_risk": (
                            np.asarray(prediction) > choice["train_median_risk"]
                        ).astype(int),
                    }
                )
            )

    oof = pd.concat(records, ignore_index=True)
    tag = run_tag(endpoint, arm, SCHEME)
    write_run(
        tag,
        oof,
        {
            "endpoint": endpoint,
            "endpoint_description": folds_mod.ENDPOINTS_BY_NAME[endpoint].description,
            "model": arm,
            "model_family": "cox",
            "scheme": SCHEME,
            "clinical_covariates": clinical_columns,
            "n_embedding_dims": len(embedding_columns),
            "k_grid": list(arm_spec["k_grid"]) if arm_spec["k_grid"] else None,
            "chosen": chosen,
            "folds_cohort_sha256": cohort_sha,
            "n_patients": len(keys),
            "n_events": int((truth["event"] == 1).sum()),
        },
        root=root,
    )
    return tag


def _fit_predict_cox(
    arm, X_train, y_train, X_val, clinical_columns, embedding_columns, arm_spec, seed
):
    if arm == "clinical":
        fitted = fit_cox_fold(X_train, y_train, seed=seed)
        val_matrix = _clinical_transform(fitted["preprocessor"], X_val)
        train_matrix = _clinical_transform(fitted["preprocessor"], X_train)
        choice = {"penalizer": fitted["penalizer"]}
    else:
        fitted = fit_embedding_cox_fold(
            X_train,
            y_train,
            clinical_columns=clinical_columns,
            embedding_columns=embedding_columns,
            k_grid=arm_spec["k_grid"],
            seed=seed,
        )
        val_matrix = _to_frame(fitted["preprocessor"].transform(X_val), X_val.index)
        train_matrix = _to_frame(fitted["preprocessor"].transform(X_train), X_train.index)
        choice = {"penalizer": fitted["penalizer"], "pca_components": fitted["k"]}
    # The prospective risk-group cutoff: the median partial hazard of the TRAINING fold.
    # A split at the pooled out-of-fold median is a split on the evaluation set.
    choice["train_median_risk"] = float(
        np.median(fitted["model"].predict_partial_hazard(train_matrix).to_numpy())
    )
    return fitted["model"].predict_partial_hazard(val_matrix).to_numpy(), choice

