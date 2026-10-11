"""Table 2 and Supplementary Tables E1-E3: C-index per fold and time-dependent AUC per repeat.

Three frames: PFS, OS, and OS scored with the PFS-trained models (no refit; every
PFS out-of-fold risk is also out of fold for OS). Patients are the intersection
over the seven models (PFS 180, OS 179).

- C-index: Harrell's C within each of the 25 (repeat, fold) test folds, mean and
  SD over the 25 folds.
- td-AUC: per repeat, the five test folds pooled (risk rank-normalised within
  fold), IPCW cumulative/dynamic AUC at 6 / 12 / 24 / 48 months, mean over the
  five repeats.
- 95% CI and the paired difference vs the clinical model: 2000 patient-level
  bootstrap resamples shared by all models; each resample re-scores every fold
  (C) or every repeat (AUC) on the resampled patients. Two-sided bootstrap p.
- Paired t-test over the 25 folds (C) or 5 repeats (AUC), descriptive only.
- Log-rank p: high vs low risk, each fold model's training-fold median as the
  cutoff (``high_risk`` in oof.csv), majority vote over the repeats.

    python paper/per_fold_tables.py [--n-boot 2000] [--runs-root models/runs/paper]
    -> paper/output/per_fold_tables.json, table2_cindex.tex, tabE{1,2,3}_tdauc_*.tex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from loguru import logger
import numpy as np
import pandas as pd
from scipy import stats

from ec_wsi_2026.modeling import models
from ec_wsi_2026.modeling.evaluation import (
    DAYS_PER_MONTH,
    ipcw_auc_pooled,
    rank_normal,
)

OUT = HERE / "output"
MODELS = ("clinical", "titan", "uni-mean", "conch-mean", "fusion", "fusion-uni", "fusion-conch")
LABEL = {
    "clinical": "Clinical",
    "titan": "TITAN",
    "uni-mean": "UNI2-h (mean)",
    "conch-mean": "CONCH v1.5 (mean)",
    "fusion": "Fusion TITAN",
    "fusion-uni": "Fusion UNI",
    "fusion-conch": "Fusion CONCH",
}
HORIZONS = (6, 12, 24, 48)
# frame name -> (endpoint the models were trained on, endpoint they are scored against)
FRAMES = {"pfs": ("pfs", "pfs"), "os": ("os", "os"), "os_pfshead": ("pfs", "os")}


# --------------------------------------------------------------------------- data


def load_frame(train_ep: str, score_ep: str, runs_root: Path):
    """Per-model OOF (patient_key, seed, fold, risk, high_risk) and the truth of *score_ep*."""
    oofs = {m: pd.read_csv(runs_root / f"{train_ep}_{m}_stratified" / "oof.csv") for m in MODELS}
    truth_oof = pd.read_csv(runs_root / f"{score_ep}_clinical_stratified" / "oof.csv")
    truth = truth_oof.groupby("patient_key")[["time", "event"]].first()
    keys = set(truth.index)
    for o in oofs.values():
        keys &= set(o["patient_key"])
    keys = sorted(keys)
    truth = truth.loc[keys]
    for m, o in oofs.items():
        o = o[o["patient_key"].isin(keys)].drop(columns=["time", "event"], errors="ignore")
        oofs[m] = o
    return keys, truth, oofs


# --------------------------------------------------------------------------- statistics


def weighted_c(t: np.ndarray, e: np.ndarray, r: np.ndarray, w: np.ndarray) -> float:
    """Harrell's C (strict time ties, half credit for risk ties) with patient multiplicities w.

    Equal to ``evaluation.harrell_c`` on the sample with each patient repeated w times.
    """
    comparable = (e[:, None] == 1) & (t[:, None] < t[None, :])
    ww = w[:, None] * w[None, :] * comparable
    den = ww.sum()
    if den == 0:
        return float("nan")
    conc = (r[:, None] > r[None, :]) + 0.5 * (r[:, None] == r[None, :])
    return float((ww * conc).sum() / den)


def boot_indices(n: int, n_boot: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.choice(n, size=n, replace=True) for _ in range(n_boot)]


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    v = values[np.isfinite(values)]
    if len(v) == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


def boot_p(values: np.ndarray) -> float:
    v = values[np.isfinite(values)]
    if len(v) == 0:
        return float("nan")
    return float(max(2 * min((v <= 0).mean(), (v >= 0).mean()), 1 / len(v)))


class Scorer:
    """Pre-indexed folds and repeats of one model for fast bootstrap re-scoring."""

    def __init__(self, oof: pd.DataFrame, keys: list[str], truth: pd.DataFrame):
        pos = {k: i for i, k in enumerate(keys)}
        self.t = truth["time"].to_numpy(float)
        self.e = truth["event"].to_numpy(int)
        self.cells = []
        for _, g in oof.groupby(["seed", "fold"]):
            idx = g["patient_key"].map(pos).to_numpy()
            self.cells.append((idx, g["risk"].to_numpy(float)))
        self.seeds = []
        o = oof.copy()
        o["rn"] = o.groupby(["seed", "fold"])["risk"].transform(lambda r: rank_normal(r.to_numpy()))
        for _, g in o.groupby("seed"):
            idx = g["patient_key"].map(pos).to_numpy()
            order = np.argsort(idx)
            self.seeds.append((idx[order], g["rn"].to_numpy(float)[order]))

    def cell_cs(self, w: np.ndarray) -> np.ndarray:
        return np.array([weighted_c(self.t[i], self.e[i], r, w[i]) for i, r in self.cells])

    def seed_aucs(self, sample: np.ndarray | None, months: int) -> np.ndarray:
        out = []
        for idx, r in self.seeds:
            # every repeat covers every patient once, so idx is a permutation of 0..n-1
            lookup = np.empty(len(idx), dtype=float)
            lookup[idx] = r
            s = np.arange(len(idx)) if sample is None else sample
            frame = pd.DataFrame({"time": self.t[s], "event": self.e[s]})
            out.append(ipcw_auc_pooled(frame, lookup[s], months * DAYS_PER_MONTH))
        return np.array(out)


def risk_groups(oof: pd.DataFrame, truth: pd.DataFrame) -> dict:
    votes = oof.groupby("patient_key")["high_risk"].agg(["sum", "count"])
    high = (votes["sum"] * 2 > votes["count"]).astype(int).reindex(truth.index)
    t, e = truth["time"].to_numpy(), truth["event"].to_numpy()
    h = high.to_numpy()
    lr = logrank_test(t[h == 1], t[h == 0], e[h == 1], e[h == 0])
    s = (
        CoxPHFitter()
        .fit(pd.DataFrame({"time": t, "event": e, "high": h}), "time", "event")
        .summary.loc["high"]
    )
    return {
        "logrank_p": float(lr.p_value),
        "hr": float(s["exp(coef)"]),
        "hr_lo": float(s["exp(coef) lower 95%"]),
        "hr_hi": float(s["exp(coef) upper 95%"]),
        "n_high": int(h.sum()),
        "n_low": int((1 - h).sum()),
    }


def frame_metrics(train_ep: str, score_ep: str, runs_root: Path, n_boot: int) -> dict:
    keys, truth, oofs = load_frame(train_ep, score_ep, runs_root)
    n = len(keys)
    scorers = {m: Scorer(oofs[m], keys, truth) for m in MODELS}
    ones = np.ones(n)
    point_c = {m: scorers[m].cell_cs(ones) for m in MODELS}
    point_auc = {m: {h: scorers[m].seed_aucs(None, h) for h in HORIZONS} for m in MODELS}

    boots = boot_indices(n, n_boot, seed=0)
    boot_c = {m: np.empty(n_boot) for m in MODELS}
    boot_auc = {m: {h: np.empty(n_boot) for h in HORIZONS} for m in MODELS}
    for b, idx in enumerate(boots):
        w = np.bincount(idx, minlength=n).astype(float)
        for m in MODELS:
            boot_c[m][b] = np.nanmean(scorers[m].cell_cs(w))
            for h in HORIZONS:
                boot_auc[m][h][b] = np.nanmean(scorers[m].seed_aucs(idx, h))
        if (b + 1) % 200 == 0:
            logger.info(f"{train_ep}->{score_ep}: {b + 1}/{n_boot} resamples")

    out = {"n": n, "events": int(truth["event"].sum()), "train_endpoint": train_ep,
           "score_endpoint": score_ep, "models": {}}
    out["events_by_horizon"] = {
        str(h): int(((truth["time"] <= h * DAYS_PER_MONTH) & (truth["event"] == 1)).sum())
        for h in HORIZONS
    }
    for m in MODELS:
        c = point_c[m]
        d = {"c_mean": float(np.nanmean(c)), "c_sd": float(np.nanstd(c, ddof=1))}
        d["c_lo"], d["c_hi"] = percentile_ci(boot_c[m])
        d.update(risk_groups(oofs[m], truth))
        d["auc"] = {}
        for h in HORIZONS:
            a = point_auc[m][h]
            ad = {"mean": float(a.mean()), "sd": float(a.std(ddof=1))}
            ad["lo"], ad["hi"] = percentile_ci(boot_auc[m][h])
            if m != "clinical":
                ref = point_auc["clinical"][h]
                diff = boot_auc[m][h] - boot_auc["clinical"][h]
                ad["diff"] = float(a.mean() - ref.mean())
                ad["diff_lo"], ad["diff_hi"] = percentile_ci(diff)
                ad["diff_p"] = boot_p(diff)
                ad["n_positive"] = int(((a - ref) > 0).sum())
                ad["t_p"] = float(stats.ttest_rel(a, ref).pvalue)
            d["auc"][str(h)] = ad
        if m != "clinical":
            ref = point_c["clinical"]
            diff = boot_c[m] - boot_c["clinical"]
            d["diff"] = float(np.nanmean(c) - np.nanmean(ref))
            d["diff_lo"], d["diff_hi"] = percentile_ci(diff)
            d["diff_p"] = boot_p(diff)
            d["t_p"] = float(stats.ttest_rel(c, ref, nan_policy="omit").pvalue)
        out["models"][m] = d
    return out


# --------------------------------------------------------------------------- rendering


def _p(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def _s(x: float) -> str:
    return f"{x:+.3f}".replace("+0.000", "+0.000")


def render_cindex(metrics: dict) -> str:
    titles = {
        "pfs": "A. PFS",
        "os": "B. OS",
        "os_pfshead": "C. OS scored with the PFS-trained models",
    }
    lines = [
        r"\begin{tabular}{lccccc c}",
        r"\toprule",
        r"Model & Mean $\pm$ SD & 95\,\% CI & Difference vs clinical [95\,\% CI] & $p$ (bootstrap) & $p$ ($t$-test) & Log-rank $p$ \\",
        r"\midrule",
    ]
    for name, title in titles.items():
        f = metrics[name]
        lines.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{title} (n = {f['n']}, events = {f['events']})}}}} \\")
        for i, m in enumerate(MODELS):
            d = f["models"][m]
            if m == "fusion":
                lines.append(r"\addlinespace")
            ci = f"[{d['c_lo']:.3f}, {d['c_hi']:.3f}]"
            if m == "clinical":
                diff, pb, pt = "--", "--", "--"
            else:
                diff = f"{_s(d['diff'])} [{_s(d['diff_lo'])}, {_s(d['diff_hi'])}]"
                pb, pt = _p(d["diff_p"]), _p(d["t_p"])
            lines.append(
                f"{LABEL[m]} & {d['c_mean']:.3f} $\\pm$ {d['c_sd']:.3f} & {ci} & {diff} & {pb} & {pt} & {_p(d['logrank_p'])} \\\\"
            )
        lines.append(r"\midrule" if name != "os_pfshead" else r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def render_tdauc(f: dict) -> str:
    lines = [
        r"\begin{tabular}{l" + "c" * len(HORIZONS) + "}",
        r"\toprule",
        "Model & " + " & ".join(f"{h} months ({f['events_by_horizon'][str(h)]} events)" for h in HORIZONS) + r" \\",
        r"\midrule",
    ]
    for m in MODELS:
        cells = []
        for h in HORIZONS:
            a = f["models"][m]["auc"][str(h)]
            cell = f"{a['mean']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}]"
            if m != "clinical":
                cell += f"; {_s(a['diff'])} [{_s(a['diff_lo'])}, {_s(a['diff_hi'])}], {a['n_positive']}/5"
            cells.append(cell)
        lines.append(f"{LABEL[m]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--runs-root", type=Path, default=models.RUNS_ROOT)
    ap.add_argument("--render-only", action="store_true", help="re-render the tables from the JSON")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "per_fold_tables.json"
    if args.render_only:
        metrics = json.loads(path.read_text())
    else:
        metrics = {"n_boot": args.n_boot}
        for name, (train_ep, score_ep) in FRAMES.items():
            metrics[name] = frame_metrics(train_ep, score_ep, args.runs_root, args.n_boot)
        path.write_text(json.dumps(metrics, indent=2))
        logger.success(f"wrote {path}")
    (OUT / "table2_cindex.tex").write_text(render_cindex(metrics))
    for tab, name in (("tabE1_tdauc_pfs", "pfs"), ("tabE2_tdauc_os", "os"), ("tabE3_tdauc_os_pfshead", "os_pfshead")):
        (OUT / f"{tab}.tex").write_text(render_tdauc(metrics[name]))
    logger.success(f"tables in {OUT}")


if __name__ == "__main__":
    main()
