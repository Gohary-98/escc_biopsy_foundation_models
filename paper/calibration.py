"""Calibration (TRIPOD+AI item 12e) for the seven models on PFS and OS, out of fold.

The persisted OOF files hold partial hazards only, so every outer cell's model is refitted here
exactly as ``models.run_arm`` fits it (same folds, seeds, inner-CV grids, preprocessing, and the arm's OWN
cohort — the refit reproduces the stored risks to machine precision) and its own Breslow
baseline is applied to its held-out patients (evaluated on the intersection of all arms): absolute risk of the event by 12 / 24 / 48 months
for each patient in each of the 5 fold repeats, averaged over repeats. Nothing is fitted on the
pooled out-of-fold data, so the calibration measures below are honest out-of-fold measures.

Per model and endpoint (``calibration_readouts.csv``, bootstrap CIs over patients):
- calibration-in-the-large at t: mean predicted risk vs Kaplan-Meier observed risk (1 - S_KM(t));
- calibration slope at t: coefficient of cloglog(predicted risk at t) in a Cox model on follow-up
  truncated at t (ideal 1; < 1 = predictions too extreme);
- Brier score at t (IPCW, scikit-survival) against the Kaplan-Meier reference Brier;
- integrated Brier score over 0-48 months;
- a PROSPECTIVE risk group per patient: the cutoff is the median linear predictor of each cell's
  TRAINING fold, applied to that cell's held-out patients; majority vote over the five repeats
  (``high_risk`` in calibration_predictions_<ep>.csv; per-seed flags in risk_groups_<ep>.csv).
  It reproduces the ``high_risk`` flags ``models.run_arm`` stores in oof.csv.

Supplementary Table E4 (12 months, PFS and OS) is rendered from ``calibration_readouts.csv``.

    python paper/calibration.py [--endpoints pfs os] [--n-boot 1000]
    -> paper/output/calibration_readouts.csv, calibration_predictions_<ep>.csv, tabE4_calibration.tex
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import warnings

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lifelines import CoxPHFitter, KaplanMeierFitter
from loguru import logger
import numpy as np
import pandas as pd
from sksurv.metrics import brier_score, integrated_brier_score
from sksurv.util import Surv

from ec_wsi_2026.modeling import covariates as cov
from ec_wsi_2026.modeling import folds as folds_mod
from ec_wsi_2026.modeling import models
from ec_wsi_2026.modeling.evaluation import DAYS_PER_MONTH
from ec_wsi_2026.modeling.survival import _transform as _clinical_transform
from ec_wsi_2026.modeling.survival import fit_cox_fold

OUT = Path(__file__).resolve().parent / "output"
MODELS = ("clinical", "titan", "uni-mean", "conch-mean", "fusion", "fusion-uni", "fusion-conch")
LABEL = {"clinical": "Clinical", "titan": "TITAN", "uni-mean": "UNI2-h (mean)", "conch-mean": "CONCH v1.5 (mean)", "fusion": "Fusion TITAN", "fusion-uni": "Fusion UNI", "fusion-conch": "Fusion CONCH"}
HORIZONS = (12, 24, 48)


# --------------------------------------------------------------------------- per-cell absolute risks


def cohort(endpoint: str):
    entry, clinical, eligible, titan, keys = models._load_cohort(endpoint)
    tables = {"titan": titan}
    for arm in MODELS:
        table = models.ARM_SPECS[arm].get("table", "titan")
        if table not in tables:
            path, hint = models.EMBEDDING_TABLES[table]
            tables[table] = models.load_embeddings(path, hint=hint)
    # Each model is refitted on its own cohort, exactly as models.run_arm does (the TITAN
    # cohort intersected with that model's embedding table), which reproduces the stored
    # out-of-fold risks. Calibration is evaluated on the intersection of all models, so
    # every model is scored on the same patients.
    cohort_keys = list(keys)
    eval_keys = sorted(set(cohort_keys) & set.intersection(*(set(t.index) for t in tables.values())))
    cc = list(entry["covariates"])
    designs = {}
    for arm in MODELS:
        spec = models.ARM_SPECS[arm]
        table = tables[spec.get("table", "titan")]
        arm_keys = sorted(set(cohort_keys) & set(table.index)) if spec["embeddings"] else sorted(cohort_keys)
        parts = []
        if spec["clinical"]:
            parts.append(cov.design_matrix(clinical.loc[arm_keys], cc))
        if spec["embeddings"]:
            parts.append(table.loc[arm_keys].astype(float))
        designs[arm] = (pd.concat(parts, axis=1), cc if spec["clinical"] else [], list(table.columns) if spec["embeddings"] else [], spec)
    truth = eligible.loc[sorted(cohort_keys), [f"{endpoint}_time", f"{endpoint}_event"]].astype(float)
    truth.columns = ["time", "event"]
    return eval_keys, truth, designs


def fit_predict_survival(arm, X_tr, y_tr, X_te, cc, ec, spec, seed, times):
    """Refit one cell as models.run_arm does; return S(t | x) for the test rows at *times* (days)."""
    if arm == "clinical":
        fitted = fit_cox_fold(X_tr, y_tr, seed=seed)
        val = _clinical_transform(fitted["preprocessor"], X_te)
    else:
        fitted = models.fit_embedding_cox_fold(X_tr, y_tr, clinical_columns=cc, embedding_columns=ec, k_grid=spec["k_grid"], seed=seed)
        val = models._to_frame(fitted["preprocessor"].transform(X_te), X_te.index)
    sf = fitted["model"].predict_survival_function(val, times=times)  # rows = times, cols = patients
    # prospective risk grouping: the cutoff is the median linear predictor of the TRAINING fold
    tr = _clinical_transform(fitted["preprocessor"], X_tr) if arm == "clinical" else models._to_frame(fitted["preprocessor"].transform(X_tr), X_tr.index)
    cut = float(np.median(fitted["model"].predict_log_partial_hazard(tr).to_numpy()))
    lp_te = fitted["model"].predict_log_partial_hazard(val).to_numpy()
    return sf.to_numpy().T, (lp_te > cut).astype(int)


def absolute_risks(endpoint: str) -> pd.DataFrame:
    eval_keys, truth, designs = cohort(endpoint)
    folds, _ = folds_mod.load_folds(endpoint)
    times = np.array([h * DAYS_PER_MONTH for h in HORIZONS])
    rows = []
    for arm in MODELS:
        X, cc, ec, spec = designs[arm]
        arm_keys = set(X.index)
        folds_r = models.restrict_folds(folds, arm_keys)  # the arm's own folds, as in models.run_arm
        for seed in sorted(folds_r["seed"].unique()):
            at_seed = folds_r.loc[folds_r["seed"] == seed]
            for fold in sorted(at_seed.loc[at_seed["fold"].notna(), "fold"].unique()):
                t0 = time.time()
                train_keys, val_keys = folds_mod.train_val_split(folds_r, seed, fold)
                train_keys = [k for k in train_keys if k in arm_keys]
                val_keys = [k for k in val_keys if k in arm_keys and k in eval_keys]
                S, high = fit_predict_survival(arm, X.loc[train_keys], truth.loc[train_keys], X.loc[val_keys], cc, ec, spec, int(seed), times)
                for i, k in enumerate(val_keys):
                    rows.append({"patient_key": k, "seed": int(seed), "fold": str(fold), "model": arm, "high_risk": int(high[i]), **{f"risk_{h}m": 1 - S[i, j] for j, h in enumerate(HORIZONS)}})
                logger.info(f"{endpoint}/{arm} seed {seed} fold {fold}: refit on {len(train_keys)} ({time.time() - t0:.0f}s)")
    pred = pd.DataFrame(rows)
    ens = pred.groupby(["model", "patient_key"])[[f"risk_{h}m" for h in HORIZONS]].mean().reset_index()
    # majority vote over the five fold repeats of the training-fold-median flag
    votes = pred.groupby(["model", "patient_key"])["high_risk"].agg(["sum", "count"]).reset_index()
    votes["high_risk"] = (votes["sum"] * 2 > votes["count"]).astype(int)
    ens = ens.merge(votes[["model", "patient_key", "high_risk"]], on=["model", "patient_key"])
    ens = ens.merge(truth.reset_index().rename(columns={"index": "patient_key"}), on="patient_key")
    ens.to_csv(OUT / f"calibration_predictions_{endpoint}.csv", index=False)
    pred[["patient_key", "seed", "fold", "model", "high_risk"]].to_csv(OUT / f"risk_groups_{endpoint}.csv", index=False)
    return ens


# --------------------------------------------------------------------------- calibration measures


def km_risk(t, e, d):
    km = KaplanMeierFitter().fit(t, e)
    return 1 - float(km.survival_function_at_times(d).iloc[0])


def slope_at(t, e, risk, d):
    cal = pd.DataFrame({"time": np.minimum(t, d), "event": ((e == 1) & (t <= d)).astype(int), "cll": np.log(-np.log(np.clip(1 - risk, 1e-6, 1 - 1e-6)))})
    try:
        return float(CoxPHFitter(penalizer=1e-4).fit(cal, "time", "event").params_["cll"])
    except Exception:  # noqa: BLE001  (lifelines raises assorted convergence errors)
        return float("nan")


def brier_at(t, e, risk, d):
    y = Surv.from_arrays(e.astype(bool), t)
    try:
        return float(brier_score(y, y, (1 - risk)[:, None], np.array([d]))[1][0])
    except ValueError:
        return float("nan")


def ibs(t, e, risks: np.ndarray, days: np.ndarray):
    y = Surv.from_arrays(e.astype(bool), t)
    ok = days < t[e == 0].max()
    try:
        return float(integrated_brier_score(y, y, 1 - risks[:, ok], days[ok]))
    except ValueError:
        return float("nan")


def readouts(ens: pd.DataFrame, endpoint: str, n_boot: int) -> list[dict]:
    rows = []
    for m in MODELS:
        g = ens[ens.model == m].reset_index(drop=True)
        t, e = g["time"].to_numpy(float), g["event"].to_numpy(int)
        rng = np.random.default_rng(0)
        boots = [rng.choice(len(g), len(g), replace=True) for _ in range(n_boot)]
        r = {"endpoint": endpoint, "model": m, "n": len(g), "events": int(e.sum())}
        for h in HORIZONS:
            d = h * DAYS_PER_MONTH
            risk = g[f"risk_{h}m"].to_numpy(float)
            r[f"mean_pred_{h}m"] = float(risk.mean())
            r[f"km_obs_{h}m"] = km_risk(t, e, d)
            r[f"itl_{h}m"] = r[f"mean_pred_{h}m"] - r[f"km_obs_{h}m"]
            r[f"slope_{h}m"] = slope_at(t, e, risk, d)
            r[f"brier_{h}m"] = brier_at(t, e, risk, d)
            r[f"brier_km_{h}m"] = brier_at(t, e, np.full(len(g), km_risk(t, e, d)), d)
            b_itl, b_slope = [], []
            for b in boots:
                b_itl.append(risk[b].mean() - km_risk(t[b], e[b], d))
                b_slope.append(slope_at(t[b], e[b], risk[b], d))
            b_itl, b_slope = np.array(b_itl), np.array(b_slope)
            r[f"itl_{h}m_lo"], r[f"itl_{h}m_hi"] = np.nanpercentile(b_itl, [2.5, 97.5])
            r[f"slope_{h}m_lo"], r[f"slope_{h}m_hi"] = np.nanpercentile(b_slope, [2.5, 97.5])
        grid_m = np.arange(6, 49, 3)  # start at 6 m: the earliest OS time is 95 d and sksurv rejects grid points before it
        grid_d = grid_m * DAYS_PER_MONTH
        # IBS needs S(t) on a grid: interpolate the three horizons linearly in log-survival (adequate for a summary)
        S3 = 1 - g[[f"risk_{h}m" for h in HORIZONS]].to_numpy(float)
        logS = np.log(np.clip(S3, 1e-6, 1))
        grid_logS = np.array([np.interp(grid_d, [0] + [h * DAYS_PER_MONTH for h in HORIZONS], [0] + list(row)) for row in logS])
        r["ibs_0_48m"] = ibs(t, e, 1 - np.exp(grid_logS), grid_d)
        S_km = np.array([1 - km_risk(t, e, d) for d in grid_d])
        r["ibs_km_0_48m"] = ibs(t, e, 1 - np.tile(S_km, (len(g), 1)), grid_d)
        rows.append(r)
        logger.info(f"{endpoint}/{m}: slope12 {r['slope_12m']:.2f} ITL12 {r['itl_12m']:+.3f} Brier12 {r['brier_12m']:.3f} (KM {r['brier_km_12m']:.3f}) IBS {r['ibs_0_48m']:.3f}")
    return rows


def render_e4(read: pd.DataFrame) -> str:
    """Table E4: calibration at 12 months and the integrated Brier score, PFS and OS."""
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        (
            r"Model & Mean predicted & Observed (KM) & Calibration in the large [95\,\% CI]"
            r" & Slope [95\,\% CI] & Brier (KM reference) & IBS 6--48 months (KM reference) \\"
        ),
        r"\midrule",
    ]
    for ep, title in (("pfs", "PFS"), ("os", "OS")):
        g = read[read.endpoint == ep]
        if g.empty:
            continue
        r0 = g.iloc[0]
        lines.append(
            rf"\multicolumn{{7}}{{l}}{{\textit{{{title} (n = {r0['n']}, events = {r0['events']})}}}} \\"
        )
        for _, r in g.iterrows():
            lines.append(
                f"{LABEL[r['model']]} & {r['mean_pred_12m']:.3f} & {r['km_obs_12m']:.3f} & "
                f"{r['itl_12m']:+.3f} [{r['itl_12m_lo']:+.3f}, {r['itl_12m_hi']:+.3f}] & "
                f"{r['slope_12m']:.2f} [{r['slope_12m_lo']:.2f}, {r['slope_12m_hi']:.2f}] & "
                f"{r['brier_12m']:.3f} ({r['brier_km_12m']:.3f}) & "
                f"{r['ibs_0_48m']:.3f} ({r['ibs_km_0_48m']:.3f}) " + r"\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--endpoints", nargs="+", default=["pfs", "os"])
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for ep in args.endpoints:
        f = OUT / f"calibration_predictions_{ep}.csv"
        ens = pd.read_csv(f) if f.exists() else absolute_risks(ep)
        all_rows += readouts(ens, ep, args.n_boot)
    read = pd.DataFrame(all_rows)
    read.to_csv(OUT / "calibration_readouts.csv", index=False)
    (OUT / "tabE4_calibration.tex").write_text(render_e4(read))
    logger.success("CALIBRATION DONE")


if __name__ == "__main__":
    main()
