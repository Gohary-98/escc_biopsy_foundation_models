"""Table 1 (patient characteristics by treatment arm) and the treatment numbers of Section 2.2.

Cohort: the patients scored in Table 2 (PFS frame, n = 180). Centers are
anonymized A-C as in the manuscript. Median follow-up by the reverse
Kaplan-Meier method on PFS time (and OS time in the JSON).

    python paper/table1.py [--runs-root models/runs/paper]
    -> paper/output/table1.tex, table1.json
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

from lifelines import KaplanMeierFitter
from loguru import logger
import numpy as np
import pandas as pd
from per_fold_tables import OUT, load_frame

from ec_wsi_2026.config import PROCESSED_DATA_DIR
from ec_wsi_2026.modeling import models
from ec_wsi_2026.modeling.evaluation import DAYS_PER_MONTH

ARMS = ("definitive", "neoadjuvant")
# manuscript labels: A = TUM, B = LMU, C = Erlangen
CENTER_ORDER = ("TUM", "LMU", "Erlangen")


def median_iqr(x: pd.Series, d: int = 1) -> str:
    x = x.dropna()
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    return f"{q2:.{d}f} ({q1:.{d}f}--{q3:.{d}f})"


def n_pct(mask: pd.Series, denom: int) -> str:
    return f"{int(mask.sum())} ({100 * mask.sum() / denom:.0f})"


def reverse_km_median_months(time: pd.Series, event: pd.Series) -> float:
    ok = time.notna() & event.notna()
    km = KaplanMeierFitter().fit(time[ok], 1 - event[ok])
    return float(km.median_survival_time_ / DAYS_PER_MONTH)


def build(c: pd.DataFrame) -> tuple[list[tuple[str, list[str]]], dict]:
    groups = {"All": c, **{a: c[c["cohort"] == a] for a in ARMS}}
    centers = list(CENTER_ORDER)
    center_label = {inst: chr(ord("A") + i) for i, inst in enumerate(centers)}
    rows: list[tuple[str, list[str]]] = []

    def add(label, fn):
        rows.append((label, [fn(g) for g in groups.values()]))

    rows.append((r"\textit{Patient}", []))
    add("Age, years", lambda g: median_iqr(g["age_at_crt"]))
    rows.append(("Sex", []))
    add(r"\quad Male", lambda g: n_pct(g["sex"] == 1, len(g)))
    add(r"\quad Female", lambda g: n_pct(g["sex"] == 2, len(g)))
    rows.append(("Center", []))
    for inst in centers:
        add(rf"\quad {center_label[inst]}", lambda g, i=inst: n_pct(g["institution"] == i, len(g)))
    rows.append((r"\textit{Tumor}", []))
    rows.append(("Clinical T stage", []))
    for t in (1, 2, 3, 4):
        add(rf"\quad cT{t}", lambda g, t=t: n_pct(g["ct_stage_num"] == t, g["ct_stage_num"].notna().sum()))
    rows.append(("Clinical nodal status", []))
    add(r"\quad cN0", lambda g: n_pct(g["cn_involved"] == 0, g["cn_involved"].notna().sum()))
    add(r"\quad cN+", lambda g: n_pct(g["cn_involved"] == 1, g["cn_involved"].notna().sum()))
    rows.append((r"Histologic grade\textsuperscript{a}", []))
    for gr in (1, 2, 3, 4):
        add(
            rf"\quad G{gr}",
            lambda g, gr=gr: n_pct(g["histologic_grade_num"] == gr, g["histologic_grade_num"].notna().sum()),
        )
    add(r"Tumor length, cm\textsuperscript{b}", lambda g: median_iqr(g["tumor_length_cm"]))
    rows.append((r"\textit{Treatment}", []))
    add(r"Radiotherapy dose, Gy\textsuperscript{c}", lambda g: median_iqr(g["total_dose_gy"]))
    add("CRT duration, days", lambda g: median_iqr(g["crt_duration_days"], 0))
    rows.append((r"\textit{Outcome}", []))
    add("PFS events, n", lambda g: str(int(g["pfs_event"].sum())))
    add("Deaths, n/N", lambda g: f"{int(g['os_event'].sum())}/{int(g['os_event'].notna().sum())}")
    add(
        r"Median follow-up, months\textsuperscript{d}",
        lambda g: f"{reverse_km_median_months(g['pfs_time'], g['pfs_event']):.1f}",
    )

    missing = {
        "histologic_grade": int(c["histologic_grade_num"].isna().sum()),
        "tumor_length": int(c["tumor_length_cm"].isna().sum()),
        "total_dose": int(c["total_dose_gy"].isna().sum()),
    }
    treatment = {}
    for a in ARMS:
        g = c[c["cohort"] == a]
        treatment[a] = {
            "n": len(g),
            "dose_recorded": int(g["total_dose_gy"].notna().sum()),
            "total_dose_median": float(g["total_dose_gy"].median()),
            "total_dose_range": [float(g["total_dose_gy"].min()), float(g["total_dose_gy"].max())],
            "fraction_dose_median": float(g["ed_gy"].median()),
            "fraction_dose_range": [float(g["ed_gy"].min()), float(g["ed_gy"].max())],
            "boost_n": int(g["boost_gd_gy"].notna().sum()),
            "boost_median": float(g["boost_gd_gy"].median()) if g["boost_gd_gy"].notna().any() else None,
            "brachy_boost_n": int(g["brachyboost_gy"].notna().sum()),
            "crt_duration_median_days": float(g["crt_duration_days"].median()),
            "end_crt_to_surgery_days": (
                [float(v) for v in np.percentile(g["crt_end_to_op_days"].dropna(), [50, 25, 75])]
                if g["crt_end_to_op_days"].notna().any()
                else None
            ),
        }
    summary = {
        "n": len(c),
        "center_labels": center_label,
        "missing": missing,
        "median_followup_months": {
            "pfs": reverse_km_median_months(c["pfs_time"], c["pfs_event"]),
            "os": reverse_km_median_months(c["os_time"], c["os_event"]),
        },
        "treatment": treatment,
    }
    return rows, summary


def render(rows, groups_n: list[int]) -> str:
    head = " & ".join(["Characteristic", f"All (n = {groups_n[0]})", f"Definitive (n = {groups_n[1]})", f"Neoadjuvant (n = {groups_n[2]})"])
    out = [r"\begin{tabular}{lccc}", r"\toprule", head + r" \\", r"\midrule"]
    for label, vals in rows:
        out.append(label + (" & " + " & ".join(vals) if vals else " & & &") + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", type=Path, default=models.RUNS_ROOT)
    args = ap.parse_args()
    keys, _, _ = load_frame("pfs", "pfs", args.runs_root)
    c = pd.read_parquet(PROCESSED_DATA_DIR / "clinical.parquet").set_index("patient_key").loc[keys]
    rows, summary = build(c)
    OUT.mkdir(parents=True, exist_ok=True)
    n = [len(c)] + [int((c["cohort"] == a).sum()) for a in ARMS]
    (OUT / "table1.tex").write_text(render(rows, n))
    (OUT / "table1.json").write_text(json.dumps(summary, indent=2))
    logger.success(f"Table 1 -> {OUT / 'table1.tex'}; {summary['median_followup_months']}")


if __name__ == "__main__":
    main()
