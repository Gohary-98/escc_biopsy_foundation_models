"""Section 3.6: what the first-year PFS events are.

Counts, in the analysed cohort (the patients scored in Table 2), the PFS events
within 12 months, how many of them are deaths without documented progression,
and how many of those deaths followed esophagectomy within 16-49 days.

    python paper/early_events.py [--runs-root models/runs/paper]
    -> paper/output/early_events.json
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
import pandas as pd
from per_fold_tables import OUT, load_frame

from ec_wsi_2026.config import PROCESSED_DATA_DIR
from ec_wsi_2026.modeling import models
from ec_wsi_2026.modeling.evaluation import DAYS_PER_MONTH

POSTOP_WINDOW_DAYS = (16, 49)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", type=Path, default=models.RUNS_ROOT)
    args = ap.parse_args()
    keys, _, _ = load_frame("pfs", "pfs", args.runs_root)
    c = pd.read_parquet(PROCESSED_DATA_DIR / "clinical.parquet").set_index("patient_key").loc[keys]

    early = c[(c["pfs_event"] == 1) & (c["pfs_time"] <= 12 * DAYS_PER_MONTH)]
    # a PFS event without progression at or before it is a death
    death = early[(early["ttp_event"] == 0) | (early["ttp_time"] > early["pfs_time"])]
    after_op = (pd.to_datetime(death["date_death"]) - pd.to_datetime(death["date_op"])).dt.days
    lo, hi = POSTOP_WINDOW_DAYS
    out = {
        "n": len(c),
        "pfs_events": int(c["pfs_event"].sum()),
        "first_year_pfs_events": len(early),
        "first_year_deaths_without_progression": len(death),
        "of_which_neoadjuvant": int((death["cohort"] == "neoadjuvant").sum()),
        f"of_which_{lo}_{hi}_days_after_surgery": int(after_op.between(lo, hi).sum()),
        "reason_of_death_counts": {
            str(k): int(v) for k, v in death["reason_of_death"].value_counts(dropna=False).items()
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "early_events.json").write_text(json.dumps(out, indent=2))
    logger.success(out)


if __name__ == "__main__":
    main()
