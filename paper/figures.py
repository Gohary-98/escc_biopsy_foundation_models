"""Figure 2 (time-dependent AUC) and Figure 3 (PFS by predicted risk group).

Rendered from ``per_fold_tables.json`` and the out-of-fold ``high_risk`` flags
only; no number is computed here that is not in that file, apart from the
Kaplan-Meier curves themselves.

    python paper/figures.py [--runs-root models/runs/paper]
    -> paper/output/figure2_auc_time.{png,pdf}, figure3_km_pfs.{png,pdf}
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

import matplotlib

matplotlib.use("Agg")
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
from per_fold_tables import HORIZONS, OUT, load_frame

from ec_wsi_2026.modeling import models
from ec_wsi_2026.modeling.evaluation import DAYS_PER_MONTH

FIG_MODELS = ("clinical", "fusion", "fusion-uni", "fusion-conch")
LABEL = {"clinical": "Clinical", "fusion": "Fusion TITAN", "fusion-uni": "Fusion UNI", "fusion-conch": "Fusion CONCH"}
COLOR = {"clinical": "#2a78d6", "fusion": "#eb6834", "fusion-uni": "#eda100", "fusion-conch": "#1baf7a"}
STYLE = {"clinical": ("-", "s"), "fusion": ("--", "o"), "fusion-uni": (":", "^"), "fusion-conch": ("-.", "D")}
COLOR_RISK = {"low": "#008300", "high": "#d55181"}
KM_XMAX_M = 60

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "axes.axisbelow": True})


def figure2(metrics: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.5))
    frames = (("pfs", "Progression-free survival"), ("os", "Overall survival"))
    offsets = dict(zip(FIG_MODELS, (-0.6, -0.2, 0.2, 0.6)))
    x0 = np.arange(len(HORIZONS)) * 3.0
    for row, (name, title) in enumerate(frames):
        f = metrics[name]
        ax_abs, ax_diff = axes[row]
        ticklabels = [f"{h}\n({f['events_by_horizon'][str(h)]})" for h in HORIZONS]
        for m in FIG_MODELS:
            a = [f["models"][m]["auc"][str(h)] for h in HORIZONS]
            ls, mk = STYLE[m]
            x = x0 + offsets[m]
            pts = np.array([d["mean"] for d in a])
            ax_abs.errorbar(x, pts, yerr=[pts - [d["lo"] for d in a], np.array([d["hi"] for d in a]) - pts],
                            color=COLOR[m], linestyle=ls, marker=mk, ms=4.5, capsize=2, elinewidth=0.8, label=LABEL[m])
            if m != "clinical":
                dd = np.array([d["diff"] for d in a])
                ax_diff.errorbar(x, dd, yerr=[dd - [d["diff_lo"] for d in a], np.array([d["diff_hi"] for d in a]) - dd],
                                 color=COLOR[m], linestyle=ls, marker=mk, ms=4.5, capsize=2, elinewidth=0.8, label=LABEL[m])
        ax_abs.axhline(0.5, color="#999999", lw=0.8, ls=":")
        ax_diff.axhline(0.0, color="black", lw=0.8)
        ax_abs.set_ylim(0.3, 0.95)
        ax_diff.set_ylim(-0.35, 0.4)
        ax_abs.set_ylabel("Time-dependent AUC")
        ax_diff.set_ylabel("Difference in AUC vs clinical model")
        ax_abs.set_title(f"{title} (n = {f['n']}, events = {f['events']})", fontsize=9, loc="left")
        ax_diff.set_title("Fusion minus clinical (paired)", fontsize=9, loc="left")
        for ax, letter in ((ax_abs, "AC"[row]), (ax_diff, "BD"[row])):
            ax.set_xticks(x0)
            ax.set_xticklabels(ticklabels)
            ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontweight="bold", fontsize=11)
            if row == 1:
                ax.set_xlabel("Months from start of CRT (events by that time)")
        if row == 0:
            ax_abs.legend(frameon=False, fontsize=7.5)
            ax_diff.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"figure2_auc_time.{ext}", dpi=300)
    plt.close(fig)


def figure3(metrics: dict, runs_root: Path) -> None:
    keys, truth, oofs = load_frame("pfs", "pfs", runs_root)
    t = truth["time"].to_numpy() / DAYS_PER_MONTH
    e = truth["event"].to_numpy()

    def high_flags(m):
        votes = oofs[m].groupby("patient_key")["high_risk"].agg(["sum", "count"]).reindex(keys)
        return (votes["sum"] * 2 > votes["count"]).to_numpy()

    clin_high = high_flags("clinical")
    ticks = list(range(0, KM_XMAX_M + 1, 12))
    fig, axes = plt.subplots(2, 2, figsize=(9, 9.5))
    for ax, m, letter in zip(axes.ravel(), FIG_MODELS, "ABCD"):
        high = high_flags(m)
        if m != "clinical":
            for mask in (~clin_high, clin_high):
                KaplanMeierFitter().fit(t[mask], e[mask]).plot_survival_function(
                    ax=ax, ci_show=False, show_censors=False, color="#9a9a9a", lw=0.8, ls=":", label="_nolegend_")
        fitters = []
        for name, mask, ls in (("low", ~high, "-"), ("high", high, "--")):
            kmf = KaplanMeierFitter().fit(t[mask], e[mask], label=f"{name.capitalize()} risk (n = {mask.sum()})")
            kmf.plot_survival_function(ax=ax, ci_show=False, show_censors=True, color=COLOR_RISK[name], lw=1.6, ls=ls,
                                       censor_styles={"ms": 4, "marker": "|"})
            fitters.append(kmf)
        d = metrics["pfs"]["models"][m]
        ax.text(0.03, 0.05, f"HR {d['hr']:.2f} (95% CI {d['hr_lo']:.2f}-{d['hr_hi']:.2f})\nlog-rank p = {d['logrank_p']:.3f}",
                transform=ax.transAxes, fontsize=7.5, bbox={"boxstyle": "square,pad=0.3", "fc": "white", "ec": "#bbbbbb", "lw": 0.6})
        ax.set_title(f"{LABEL[m]} model", fontsize=9, loc="left")
        ax.text(-0.12, 1.05, letter, transform=ax.transAxes, fontweight="bold", fontsize=11)
        ax.set_xlim(0, KM_XMAX_M)
        ax.set_ylim(0, 1.02)
        ax.set_xticks(ticks)
        ax.set_xlabel("Months from start of CRT")
        ax.set_ylabel("Progression-free survival")
        ax.legend(frameon=False, fontsize=7.5, loc="upper right")
        add_at_risk_counts(*fitters, ax=ax, rows_to_show=["At risk"], xticks=ticks, fontsize=6.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"figure3_km_pfs.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", type=Path, default=models.RUNS_ROOT)
    args = ap.parse_args()
    metrics = json.loads((OUT / "per_fold_tables.json").read_text())
    figure2(metrics)
    figure3(metrics, args.runs_root)
    logger.success(f"figures in {OUT}")


if __name__ == "__main__":
    main()
