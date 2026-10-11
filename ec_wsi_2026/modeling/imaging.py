"""Consolidate TRIDENT's per-patient slide-embedding output into one table.

TRIDENT writes one ``.h5`` file per patient under a directory named
``slide_features_<encoder>/`` (e.g. ``slide_features_titan/``), each holding a
``features`` dataset — the fixed-length slide embedding — and a ``coords``
dataset recording the tile grid the slide encoder attended over. The filename
stem is the ``patient_key`` (``scripts/_stage_trident_input.py`` names the
staged symlinks that way), so no separate identifier lookup is needed.

This module turns that raw, multi-file tree into a single small parquet — one
row per patient, one column per embedding dimension — so it plugs into the
same modelling machinery as the clinical covariates in
``ec_wsi_2026/modeling/covariates.py``. The raw tree (large, on the extraction
machine) stays where TRIDENT put it; the consolidated parquet is what the
models read.
"""

from __future__ import annotations

from pathlib import Path

import h5py
from loguru import logger
import numpy as np
import pandas as pd


def find_slide_features_dir(job_dir: Path, *, encoder: str) -> Path:
    """Locate ``slide_features_<encoder>/`` anywhere under *job_dir*.

    TRIDENT nests it under a run-parameter-encoded directory name (e.g.
    ``20x_512px_0px_overlap/``) that this deliberately does not hardcode.
    """
    match = next(job_dir.glob(f"**/slide_features_{encoder}"), None)
    if match is None:
        raise FileNotFoundError(
            f"no slide_features_{encoder}/ directory found under {job_dir} "
            "(has the slide-encoder run completed?)"
        )
    return match


def consolidate_mean_patch_embeddings(
    features_dir: Path,
    *,
    prefix: str,
    expected_patient_keys: set[str] | None = None,
) -> pd.DataFrame:
    """Mean-pool each patient's patch-feature matrix into one embedding row.

    The benchmark-standard baseline (Patho-Bench; the TITAN paper's
    mean-pooling comparator): no attention, no training — just the average
    patch embedding. Same output shape and contracts as
    :func:`consolidate_slide_embeddings`: ``patient_key`` +
    ``{prefix}_{i:03d}`` columns, a raised error on inconsistent dims, logged
    (not fatal) mismatches against *expected_patient_keys*.
    """
    paths = sorted(features_dir.glob("*.h5"))
    if not paths:
        raise FileNotFoundError(f"no .h5 files under {features_dir}")

    rows: list[dict] = []
    dim: int | None = None
    for path in paths:
        with h5py.File(path, "r") as handle:
            features = np.asarray(handle["features"][()], dtype=np.float64)
        if features.ndim != 2:
            raise ValueError(f"{path.stem}: expected (n_patches, dim), got {features.shape}")
        if dim is None:
            dim = features.shape[1]
        elif features.shape[1] != dim:
            raise ValueError(
                f"{path.stem}: feature dim {features.shape[1]} != {dim} seen in earlier "
                f"files under {features_dir} — inconsistent run"
            )
        mean = features.mean(axis=0)
        row = {"patient_key": path.stem}
        row.update({f"{prefix}_{i:03d}": float(v) for i, v in enumerate(mean)})
        rows.append(row)

    table = pd.DataFrame(rows)
    logger.info(
        f"mean-pooled {len(table)} patient bag(s) ({dim} dims) from {features_dir}"
    )
    if expected_patient_keys is not None:
        found = set(table["patient_key"])
        missing = expected_patient_keys - found
        extra = found - expected_patient_keys
        if missing:
            logger.warning(
                f"{len(missing)} staged patient(s) have no {prefix} bag: {sorted(missing)}"
            )
        if extra:
            logger.warning(
                f"{len(extra)} {prefix} bag(s) have no matching staged slide: {sorted(extra)}"
            )
    return table


def find_patch_features_dir(job_dir: Path, *, encoder: str) -> Path:
    """Locate ``features_<encoder>/`` (patch-level features) under *job_dir*.

    Same nesting caveat as :func:`find_slide_features_dir`; patch encoders
    write ``features_<name>/`` where slide encoders write
    ``slide_features_<name>/``.
    """
    match = next(job_dir.glob(f"**/features_{encoder}"), None)
    if match is None:
        raise FileNotFoundError(
            f"no features_{encoder}/ directory found under {job_dir} "
            "(has the patch-encoder run completed?)"
        )
    return match


def consolidate_slide_embeddings(
    slide_features_dir: Path,
    *,
    prefix: str,
    expected_patient_keys: set[str] | None = None,
) -> pd.DataFrame:
    """Read every ``<patient_key>.h5`` in *slide_features_dir* into one wide table.

    Columns: ``patient_key``, then ``{prefix}_000`` .. ``{prefix}_{dim-1}``.
    Raises if embedding dimensions disagree across patients (a TRIDENT/run
    inconsistency, not something to average or drop silently). Logs, but does
    not raise on, a mismatch against *expected_patient_keys* — a handful of
    slides failing extraction is expected occasionally and already visible in
    TRIDENT's own run log; the point here is to name them, not hide them.
    """
    paths = sorted(slide_features_dir.glob("*.h5"))
    if not paths:
        raise FileNotFoundError(f"no .h5 files under {slide_features_dir}")

    rows: list[dict] = []
    dim: int | None = None
    for path in paths:
        patient_key = path.stem
        with h5py.File(path, "r") as handle:
            features = np.asarray(handle["features"][()]).reshape(-1)
        if dim is None:
            dim = features.shape[0]
        elif features.shape[0] != dim:
            raise ValueError(
                f"{patient_key}: embedding dim {features.shape[0]} != {dim} seen in "
                f"earlier files under {slide_features_dir} — inconsistent run"
            )
        row = {"patient_key": patient_key}
        row.update({f"{prefix}_{i:03d}": float(v) for i, v in enumerate(features)})
        rows.append(row)

    table = pd.DataFrame(rows)
    logger.info(f"consolidated {len(table)} slide embedding(s) ({dim} dims) from {slide_features_dir}")

    if expected_patient_keys is not None:
        found = set(table["patient_key"])
        missing = expected_patient_keys - found
        extra = found - expected_patient_keys
        if missing:
            logger.warning(
                f"{len(missing)} staged patient(s) have no {prefix} embedding — check "
                f"the extraction log for these: {sorted(missing)}"
            )
        if extra:
            logger.warning(
                f"{len(extra)} {prefix} embedding(s) have no matching staged slide: "
                f"{sorted(extra)}"
            )

    return table
