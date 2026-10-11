"""Build data/interim/trident_input from the slide manifest.

One symlink per usable primary slide, named ``<patient_key><suffix>`` so
TRIDENT's per-slide output filenames join straight back to patient_key. Called
by scripts/run_trident.sh; not part of the ec_wsi_2026 package.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

manifest = pd.read_csv("data/processed/slide_manifest.csv")
usable = manifest[(manifest["status"] == "ok") & (manifest["primary"])]

staging = Path("data/interim/trident_input")
staging.mkdir(parents=True, exist_ok=True)

created = 0
for _, row in usable.iterrows():
    src = Path(row["file_path"])
    dst = staging / f"{row['patient_key']}{src.suffix}"
    if not dst.exists():
        dst.symlink_to(src.resolve())
        created += 1

print(f"staged {len(usable)} slides ({created} new symlinks) -> {staging}")
