"""Build, persist and serve the cross-validation folds.

The folds are an *artifact*, not a runtime construction: they are written once
to ``data/processed/folds.csv``, described by a manifest carrying a hash of the
cohort they were built from, and every model reads them. This is what makes the
comparison of each imaging or fusion model with the clinical model a paired one
(identical patients in identical test folds). ``load_folds`` recomputes the
cohort hash against the current labels table and refuses to serve stale folds.

Scheme: 5 folds x 5 repeats of ``StratifiedGroupKFold`` grouped by
``patient_key``. The stratification key starts as ``institution x treatment arm
x event`` and is coarsened until every stratum holds at least five patients;
for PFS and OS the key actually used is **treatment arm x event** (the
``stratum`` column of ``folds.csv``).

A fourth site (n = 14, no slides) is pinned to training (``role="train_only"``):
its patients never appear in a test fold and receive no out-of-fold
prediction. It has no slides, so it drops out of every model trained on the
imaged subset; pinning it keeps the folds of the imaged patients identical to
those the clinical baseline was first built on.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ec_wsi_2026.clinical.endpoints import ENDPOINTS_BY_NAME
from ec_wsi_2026.config import DEFAULT_SEED, FOLDS_CSV, FOLDS_MANIFEST_JSON, LABELS_PARQUET

TRAIN_ONLY_SITES = ("Wuerzburg",)
N_SPLITS = 5
N_REPEATS = 5

# Endpoints that get folds (PFS primary, OS secondary).
MODELLED_ENDPOINTS = ("pfs", "os")
SCHEME = "stratified"

FOLD_COLUMNS = ["endpoint", "scheme", "seed", "fold", "patient_key", "stratum", "role"]


def eligible_labels(labels: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    """Rows usable for *endpoint*: observed event (and time, where applicable)."""
    spec = ENDPOINTS_BY_NAME[endpoint]
    mask = labels[f"{endpoint}_event"].notna()
    if spec.kind == "time_to_event":
        mask &= labels[f"{endpoint}_time"].notna()
    return labels.loc[mask]


def cohort_hash(labels: pd.DataFrame, endpoint: str) -> str:
    """SHA-256 over the sorted ``patient_key|event|time`` strings of the eligible set.

    Any change to who is eligible or to their ground truth changes the hash,
    which is exactly what must invalidate previously built folds.
    """
    spec = ENDPOINTS_BY_NAME[endpoint]
    rows = eligible_labels(labels, endpoint)
    parts = []
    for _, row in rows.iterrows():
        event = int(row[f"{endpoint}_event"])
        time = float(row[f"{endpoint}_time"]) if spec.kind == "time_to_event" else ""
        parts.append(f"{row['patient_key']}|{event}|{time}")
    return hashlib.sha256("\n".join(sorted(parts)).encode()).hexdigest()


def _strata(rows: pd.DataFrame, endpoint: str, n_splits: int) -> pd.Series:
    """``institution x cohort x event`` labels, coarsened until every stratum >= n_splits."""
    event = rows[f"{endpoint}_event"].astype(int).astype(str)
    candidates = {
        "institution x cohort x event": rows["institution"] + "|" + rows["cohort"] + "|" + event,
        "cohort x event": rows["cohort"] + "|" + event,
        "event": event,
    }
    for name, strata in candidates.items():
        if strata.value_counts().min() >= n_splits:
            if name != "institution x cohort x event":
                logger.warning(
                    f"{endpoint}: smallest 'institution x cohort x event' stratum is below "
                    f"{n_splits}; coarsened stratification to '{name}'"
                )
            return strata
    logger.warning(f"{endpoint}: even 'event' strata are below {n_splits}; using them anyway")
    return candidates["event"]


def build_stratified(
    labels: pd.DataFrame,
    endpoint: str,
    *,
    n_splits: int = N_SPLITS,
    n_repeats: int = N_REPEATS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Repeated stratified group folds; one row per (patient, repeat)."""
    rows = eligible_labels(labels, endpoint)
    pinned = rows["institution"].isin(TRAIN_ONLY_SITES)
    rotating = rows.loc[~pinned]
    strata = _strata(rotating, endpoint, n_splits)

    records: list[dict] = []
    for repeat in range(n_repeats):
        repeat_seed = seed + repeat
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=repeat_seed)
        groups = rotating["patient_key"].to_numpy()
        for fold, (_, val_idx) in enumerate(
            splitter.split(rotating, strata.to_numpy(), groups=groups)
        ):
            for position in val_idx:
                records.append(
                    {
                        "endpoint": endpoint,
                        "scheme": SCHEME,
                        "seed": repeat_seed,
                        "fold": str(fold),
                        "patient_key": rotating["patient_key"].iloc[position],
                        "stratum": strata.iloc[position],
                        "role": "rotating",
                    }
                )
        for _, row in rows.loc[pinned].iterrows():
            records.append(
                {
                    "endpoint": endpoint,
                    "scheme": SCHEME,
                    "seed": repeat_seed,
                    "fold": None,
                    "patient_key": row["patient_key"],
                    "stratum": None,
                    "role": "train_only",
                }
            )
    return pd.DataFrame.from_records(records, columns=FOLD_COLUMNS)


def train_val_split(
    folds: pd.DataFrame, seed: int, fold: str
) -> tuple[np.ndarray, np.ndarray]:
    """Materialise one split: validation = the fold; training = everything else.

    ``train_only`` rows (Wuerzburg) join every training set and no validation set.
    """
    at_seed = folds.loc[folds["seed"] == seed]
    # `fold` is None/NA for train_only rows; equality against NA must count as
    # False (train side), not propagate as missing.
    is_val = (at_seed["fold"] == fold).fillna(False).astype(bool)
    val = at_seed.loc[is_val, "patient_key"].to_numpy()
    train = at_seed.loc[~is_val, "patient_key"].unique()
    if not len(val):
        raise ValueError(f"no validation patients for seed={seed} fold={fold!r}")
    return train, val


def assert_folds(folds: pd.DataFrame, labels: pd.DataFrame) -> None:
    """Leakage and coverage invariants; raises on the first violation."""
    for (endpoint, scheme), frame in folds.groupby(["endpoint", "scheme"]):
        eligible = eligible_labels(labels, endpoint)
        eligible_keys = set(eligible["patient_key"])
        pinned_keys = set(
            eligible.loc[eligible["institution"].isin(TRAIN_ONLY_SITES), "patient_key"]
        )

        # No train_only patient may ever be a validation patient.
        val_keys = set(frame.loc[frame["fold"].notna(), "patient_key"])
        leaked = val_keys & pinned_keys
        if leaked:
            raise AssertionError(f"{endpoint}/{scheme}: pinned patient(s) in validation: {leaked}")

        for seed, at_seed in frame.groupby("seed"):
            # The seed's rows must partition the eligible set exactly.
            keys = at_seed["patient_key"]
            if keys.duplicated().any():
                dupes = keys[keys.duplicated()].tolist()
                raise AssertionError(f"{endpoint}/{scheme}/seed {seed}: duplicated {dupes}")
            if set(keys) != eligible_keys:
                raise AssertionError(
                    f"{endpoint}/{scheme}/seed {seed}: fold patients != eligible patients "
                    f"(missing {len(eligible_keys - set(keys))}, "
                    f"extra {len(set(keys) - eligible_keys)})"
                )
            # Every split must keep train and validation disjoint.
            for fold in at_seed.loc[at_seed["fold"].notna(), "fold"].unique():
                train, val = train_val_split(frame, seed, fold)
                overlap = set(train) & set(val)
                if overlap:
                    raise AssertionError(
                        f"{endpoint}/{scheme}/seed {seed}/fold {fold}: leakage {overlap}"
                    )
        per_seed = frame.groupby("seed")["patient_key"].nunique()
        if per_seed.nunique() != 1:
            raise AssertionError(f"{endpoint}: unequal patient counts across seeds")
    logger.success("fold invariants hold: no leakage, full coverage, pinned site never held out")


def write_folds(
    labels: pd.DataFrame,
    *,
    endpoints: tuple[str, ...] = MODELLED_ENDPOINTS,
    folds_path: Path = FOLDS_CSV,
    manifest_path: Path = FOLDS_MANIFEST_JSON,
) -> pd.DataFrame:
    """Build the folds for every modelled endpoint and persist them."""
    frames = []
    manifest: dict = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed_base": DEFAULT_SEED,
        "n_splits": N_SPLITS,
        "n_repeats": N_REPEATS,
        "train_only_sites": list(TRAIN_ONLY_SITES),
        "endpoints": {},
    }
    for endpoint in endpoints:
        frames.append(build_stratified(labels, endpoint))
        eligible = eligible_labels(labels, endpoint)
        manifest["endpoints"][endpoint] = {
            "n_patients": len(eligible),
            "n_events": int((eligible[f"{endpoint}_event"] == 1).sum()),
            "n_train_only": int(eligible["institution"].isin(TRAIN_ONLY_SITES).sum()),
            "cohort_sha256": cohort_hash(labels, endpoint),
        }

    folds = pd.concat(frames, ignore_index=True)
    assert_folds(folds, labels)

    folds_path.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(folds_path, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.success(
        f"wrote {folds_path.name} ({len(folds)} rows, {len(manifest['endpoints'])} endpoints) "
        f"and {manifest_path.name}"
    )
    return folds


def load_folds(
    endpoint: str,
    *,
    folds_path: Path = FOLDS_CSV,
    manifest_path: Path = FOLDS_MANIFEST_JSON,
    labels_path: Path = LABELS_PARQUET,
) -> tuple[pd.DataFrame, str]:
    """Serve the persisted folds for one endpoint, or refuse.

    Returns ``(folds, cohort_sha256)``; the hash goes into every metrics JSON so
    results are traceable to the exact cohort the folds were built on.
    """
    if not folds_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"folds not found ({folds_path}, {manifest_path}); run `modeling folds` first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if endpoint not in manifest["endpoints"]:
        raise KeyError(f"no folds recorded for endpoint {endpoint!r}; run `modeling folds`")

    labels = pd.read_parquet(labels_path)
    current = cohort_hash(labels, endpoint)
    stored = manifest["endpoints"][endpoint]["cohort_sha256"]
    if current != stored:
        raise ValueError(
            f"folds for {endpoint!r} are stale: the eligible cohort changed since they were "
            "built (labels hash mismatch). Rerun `modeling folds` deliberately - this "
            "invalidates paired comparisons with any results computed on the old folds."
        )

    folds = pd.read_csv(folds_path, dtype={"fold": "string"})
    subset = folds.loc[(folds["endpoint"] == endpoint) & (folds["scheme"] == SCHEME)].copy()
    if subset.empty:
        raise KeyError(f"no rows for endpoint={endpoint!r} in {folds_path}")
    return subset, stored
