"""Shared IO for model runs: tags and out-of-fold artifacts.

Every training run writes two files under ``models/runs/paper/<tag>/``:

- ``oof.csv``: one row per (patient, repeat) with the ground truth and the
  out-of-fold prediction: ``patient_key, endpoint, scheme, seed, fold, time,
  event, risk, high_risk``.
- ``model_meta.json``: covariates, per-fold hyperparameters and the folds
  cohort hash.

The tag is ``{endpoint}_{model}_{scheme}``. Runs are regenerable and not
committed; the paper scripts under ``paper/`` read them.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from loguru import logger
import pandas as pd

from ec_wsi_2026.config import RUNS_DIR


def run_tag(endpoint: str, model: str, scheme: str) -> str:
    return f"{endpoint}_{model}_{scheme}"


def run_dir(tag: str, *, root: Path = RUNS_DIR) -> Path:
    return root / tag


def write_run(tag: str, oof: pd.DataFrame, meta: dict, *, root: Path = RUNS_DIR) -> Path:
    directory = run_dir(tag, root=root)
    directory.mkdir(parents=True, exist_ok=True)
    oof.to_csv(directory / "oof.csv", index=False)
    meta = {**meta, "tag": tag, "written": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (directory / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.success(f"wrote {directory / 'oof.csv'} ({len(oof)} rows)")
    return directory


def load_run(tag: str, *, root: Path = RUNS_DIR) -> tuple[pd.DataFrame, dict]:
    directory = run_dir(tag, root=root)
    oof_path = directory / "oof.csv"
    meta_path = directory / "model_meta.json"
    if not oof_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"run {tag!r} not found under {root}; train it first")
    oof = pd.read_csv(oof_path, dtype={"fold": "string"})
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return oof, meta
