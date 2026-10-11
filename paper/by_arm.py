"""Section 3.6 (exploratory): fusion minus clinical C-index within each treatment arm.

The out-of-fold predictions of Table 2 are restricted to one treatment arm and
scored exactly as Table 2 (Harrell's C within each of the 25 test folds, mean
over folds); the paired 95% CI comes from 1000 patient bootstrap resamples within the
arm.

    python paper/by_arm.py [--n-boot 1000] [--runs-root models/runs/paper]
    -> paper/output/by_arm.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from loguru import logger
import numpy as np
import pandas as pd
from per_fold_tables import (
    OUT,
    Scorer,
    boot_indices,
    boot_p,
    load_frame,
    percentile_ci,
)

from ec_wsi_2026.config import PROCESSED_DATA_DIR
from ec_wsi_2026.modeling import models

MODELS = ("clinical", "fusion", "fusion-uni", "fusion-conch")


def arm_metrics(endpoint: str, arm: str, runs_root: Path, n_boot: int) -> dict:
    keys, truth, oofs = load_frame(endpoint, endpoint, runs_root)
    cohort = pd.read_parquet(PROCESSED_DATA_DIR / "labels.parquet").set_index("patient_key")["cohort"]
    keys = [k for k in keys if cohort[k] == arm]
    truth = truth.loc[keys]
    scorers = {m: Scorer(oofs[m][oofs[m]["patient_key"].isin(keys)], keys, truth) for m in MODELS}
    n = len(keys)
    point = {m: float(np.nanmean(scorers[m].cell_cs(np.ones(n)))) for m in MODELS}
    boot = {m: np.empty(n_boot) for m in MODELS}
    for b, idx in enumerate(boot_indices(n, n_boot, seed=0)):
        w = np.bincount(idx, minlength=n).astype(float)
        for m in MODELS:
            boot[m][b] = np.nanmean(scorers[m].cell_cs(w))
    out = {"n": n, "events": int(truth["event"].sum()), "c": point, "diff_vs_clinical": {}}
    for m in MODELS[1:]:
        d = boot[m] - boot["clinical"]
        lo, hi = percentile_ci(d)
        out["diff_vs_clinical"][m] = {"diff": point[m] - point["clinical"], "lo": lo, "hi": hi, "p": boot_p(d)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--runs-root", type=Path, default=models.RUNS_ROOT)
    args = ap.parse_args()
    res = {
        ep: {arm: arm_metrics(ep, arm, args.runs_root, args.n_boot) for arm in ("definitive", "neoadjuvant")}
        for ep in ("pfs", "os")
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "by_arm.json").write_text(json.dumps(res, indent=2))
    for ep, arms in res.items():
        for arm, r in arms.items():
            for m, d in r["diff_vs_clinical"].items():
                logger.info(f"{ep} {arm} (n={r['n']}, events={r['events']}) {m}: {d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]")


if __name__ == "__main__":
    main()
