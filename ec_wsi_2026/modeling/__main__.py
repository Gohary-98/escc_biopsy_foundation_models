"""CLI: folds, embedding consolidation and model training.

    python -m ec_wsi_2026.modeling folds              # build + persist data/processed/folds.csv
    python -m ec_wsi_2026.modeling consolidate-titan  # TRIDENT's TITAN slide embeddings -> one parquet
    python -m ec_wsi_2026.modeling consolidate-mean   # mean-pooled UNI2-h / CONCH v1.5 tile features -> parquets
    python -m ec_wsi_2026.modeling train              # the 7 models x PFS / OS, out of fold
    python -m ec_wsi_2026.modeling train --endpoint pfs --model fusion

Requires the curated tables (`python -m ec_wsi_2026.dataset all` first). Every
model reads the persisted folds, so all comparisons are paired.
"""

from __future__ import annotations

from loguru import logger
import pandas as pd
import typer

from ec_wsi_2026.config import (
    CONCH_MEAN_EMBEDDINGS_PARQUET,
    LABELS_PARQUET,
    SLIDE_MANIFEST_CSV,
    TITAN_EMBEDDINGS_PARQUET,
    TITAN_JOB_DIR,
    UNI_JOB_DIR,
    UNI_MEAN_EMBEDDINGS_PARQUET,
)
from ec_wsi_2026.modeling import folds as folds_mod
from ec_wsi_2026.modeling import imaging as imaging_mod
from ec_wsi_2026.modeling import models as models_mod

app = typer.Typer(add_completion=False, help=__doc__)


def _expected_patient_keys() -> set[str]:
    manifest = pd.read_csv(SLIDE_MANIFEST_CSV)
    return set(manifest.loc[(manifest["status"] == "ok") & manifest["primary"], "patient_key"])


@app.command()
def folds() -> None:
    """Build and persist the cross-validation folds (PFS and OS)."""
    labels = pd.read_parquet(LABELS_PARQUET)
    folds_mod.write_folds(labels)


@app.command(name="consolidate-titan")
def consolidate_titan() -> None:
    """Turn TRIDENT's raw per-patient TITAN .h5 output into one small parquet."""
    slide_features_dir = imaging_mod.find_slide_features_dir(TITAN_JOB_DIR, encoder="titan")
    table = imaging_mod.consolidate_slide_embeddings(
        slide_features_dir, prefix="titan", expected_patient_keys=_expected_patient_keys()
    )
    TITAN_EMBEDDINGS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(TITAN_EMBEDDINGS_PARQUET, index=False)
    logger.success(f"wrote {TITAN_EMBEDDINGS_PARQUET} ({len(table)} patients)")


@app.command(name="consolidate-mean")
def consolidate_mean() -> None:
    """Mean-pool the UNI2-h and CONCH v1.5 tile features into per-patient parquets."""
    expected = _expected_patient_keys()
    jobs = {
        "uni_v2": (UNI_JOB_DIR, UNI_MEAN_EMBEDDINGS_PARQUET, "uni_mean"),
        # CONCH v1.5 tile features are a byproduct of the TITAN run
        "conch_v15": (TITAN_JOB_DIR, CONCH_MEAN_EMBEDDINGS_PARQUET, "conch_mean"),
    }
    for encoder, (job_dir, out_path, prefix) in jobs.items():
        features_dir = imaging_mod.find_patch_features_dir(job_dir, encoder=encoder)
        table = imaging_mod.consolidate_mean_patch_embeddings(
            features_dir, prefix=prefix, expected_patient_keys=expected
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(out_path, index=False)
        logger.success(f"wrote {out_path} ({len(table)} patients)")


@app.command()
def train(
    endpoint: str = typer.Option(None, help="pfs or os (default: both)."),
    model: str = typer.Option(None, help=f"One of {models_mod.MODELS} (default: all)."),
) -> None:
    """Train the models out of fold on the persisted folds."""
    endpoints = (endpoint,) if endpoint else models_mod.ENDPOINTS
    arms = (model,) if model else models_mod.MODELS
    for ep in endpoints:
        for arm in arms:
            models_mod.run_arm(ep, arm)


if __name__ == "__main__":
    app()
