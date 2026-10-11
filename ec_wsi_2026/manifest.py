"""Build the slide manifest and the imaging-coverage analysis.

    python -m ec_wsi_2026.manifest build       # -> data/interim/slide_manifest.csv
    python -m ec_wsi_2026.manifest coverage    # -> reports/slide_coverage.md
    python -m ec_wsi_2026.manifest all

There is no slide identifier anywhere in the clinical workbooks. The only link
between a slide and a patient is the directory name -- ``tum_def_1``,
``erl_neo_78`` -- so this manifest *is* the join, and it is the single artifact
every downstream stage keys on. The workbooks stay a read-only source.

The coverage half is the more important half. Not every patient has a slide, and
which patients do is not random until proven otherwise: if slides were retrieved
preferentially for patients who recurred, or only from the sites with digital
archives, then models trained on the imaged subset inherit that selection and
every estimate is biased. :func:`coverage_report` quantifies it by comparing
event rates and baseline characteristics between the imaged and non-imaged
groups.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re

from loguru import logger
import pandas as pd
import typer

from ec_wsi_2026.clinical import endpoints as ep
from ec_wsi_2026.clinical import schema as sch
from ec_wsi_2026.config import (
    CLINICAL_PARQUET,
    SLIDE_COVERAGE_MD,
    SLIDE_MANIFEST_CSV,
    WSI_DIR,
    add_openslide_dll_directory,
)

app = typer.Typer(add_completion=False, help=__doc__)

# Formats OpenSlide can serve region reads from. `.tif` is included because
# vendors export pyramidal TIFFs, but a flat TIFF will fail to open and is
# caught as `unreadable`.
SLIDE_EXTENSIONS = (".svs", ".ndpi", ".mrxs", ".scn", ".tif", ".tiff", ".qptiff", ".bif")

# When one specimen has been saved in several formats, keep the vendor-native
# one: it carries the full pyramid and the acquisition metadata, which a
# re-export typically loses.
FORMAT_PREFERENCE = (".ndpi", ".mrxs", ".svs", ".scn", ".bif", ".qptiff", ".tiff", ".tif")

FOLDER_RE = re.compile(r"^(?P<inst>[a-z]+)_(?P<cohort>def|neo)_(?P<patient_no>\d+)$", re.IGNORECASE)

COHORT_FROM_FOLDER = {"def": "definitive", "neo": "neoadjuvant"}


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------
def parse_folder_name(name: str) -> dict | None:
    """Parse ``{institution}_{cohort}_{patient_no}`` into its parts."""
    match = FOLDER_RE.match(name.strip())
    if not match:
        return None
    code = match.group("inst").lower()
    return {
        "institution_from_folder": sch.SLIDE_CODE_TO_INSTITUTION.get(code),
        "institution_code": code,
        "cohort": COHORT_FROM_FOLDER[match.group("cohort").lower()],
        "patient_no": int(match.group("patient_no")),
    }


def _hash_head_tail(path: Path, block: int = 1 << 20) -> str:
    """MD5 of the first and last megabyte, for cheap transfer verification.

    Hashing 2.4 GB slides in full costs minutes each and buys nothing here: the
    purpose is to detect a truncated or partially-copied file after moving the
    cohort between machines, and a truncated file differs in its tail.
    """
    size = path.stat().st_size
    digest = hashlib.md5()
    with path.open("rb") as handle:
        digest.update(handle.read(block))
        if size > block:
            handle.seek(max(size - block, block))
            digest.update(handle.read(block))
    return digest.hexdigest()


def _read_slide_properties(path: Path) -> dict:
    """Open the slide and pull the acquisition metadata, or report why not."""
    import openslide

    try:
        with openslide.OpenSlide(str(path)) as slide:
            properties = slide.properties
            width, height = slide.level_dimensions[0]
            return {
                "readable": True,
                "unreadable_reason": None,
                "vendor": properties.get(openslide.PROPERTY_NAME_VENDOR),
                "objective_power": _as_float(properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)),
                "mpp_x": _as_float(properties.get(openslide.PROPERTY_NAME_MPP_X)),
                "mpp_y": _as_float(properties.get(openslide.PROPERTY_NAME_MPP_Y)),
                "level_count": slide.level_count,
                "level0_width": width,
                "level0_height": height,
                "level_downsamples": ",".join(f"{d:.4g}" for d in slide.level_downsamples),
            }
    except Exception as exc:  # noqa: BLE001 - any failure means we cannot use it
        return {
            "readable": False,
            "unreadable_reason": f"{type(exc).__name__}: {exc}",
            "vendor": None,
            "objective_power": None,
            "mpp_x": None,
            "mpp_y": None,
            "level_count": None,
            "level0_width": None,
            "level0_height": None,
            "level_downsamples": None,
        }


def _as_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def discover_slides(wsi_dir: Path = WSI_DIR, *, read_properties: bool = True) -> pd.DataFrame:
    """Walk the slide directory and emit one row per file, plus empty folders."""
    add_openslide_dll_directory()
    wsi_dir = Path(wsi_dir)
    if not wsi_dir.is_dir():
        raise FileNotFoundError(f"slide directory not found: {wsi_dir}")

    rows: list[dict] = []
    for folder in sorted(p for p in wsi_dir.iterdir() if p.is_dir()):
        parsed = parse_folder_name(folder.name)
        files = sorted(
            p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SLIDE_EXTENSIONS
        )

        if not files:
            # A folder with no slide is a transfer that did not complete, and it
            # must appear in the manifest -- silently skipping it is how a
            # missing patient goes unnoticed.
            rows.append(
                {
                    "slide_id": f"{folder.name}__MISSING",
                    "file_id": f"{folder.name}__MISSING",
                    "folder": folder.name,
                    "file_name": None,
                    "file_path": None,
                    "file_format": None,
                    "file_size_bytes": None,
                    "md5_head_tail": None,
                    "status": "missing_file",
                    "primary": False,
                    **(parsed or _unparsed()),
                    **_empty_properties(),
                }
            )
            logger.warning(f"{folder.name}: no slide file in this directory")
            continue

        for path in files:
            row = {
                # slide_id identifies the *specimen*, so the same tissue saved as
                # both .ndpi and .tif shares one id -- that is what makes the
                # duplicate detectable. file_id is the unique per-row key.
                "slide_id": f"{folder.name}__{path.stem}",
                "file_id": f"{folder.name}__{path.name}",
                "folder": folder.name,
                "file_name": path.name,
                "file_path": str(path),
                "file_format": path.suffix.lower(),
                "file_size_bytes": path.stat().st_size,
                "md5_head_tail": _hash_head_tail(path),
                **(parsed or _unparsed()),
            }
            row.update(_read_slide_properties(path) if read_properties else _empty_properties())
            rows.append(row)

        if parsed is None:
            logger.warning(f"{folder.name}: does not match the expected naming convention")

    frame = pd.DataFrame(rows)
    logger.info(f"discovered {len(frame)} slide file(s) across {frame['folder'].nunique()} folder(s)")
    return frame


def _unparsed() -> dict:
    return {
        "institution_from_folder": None,
        "institution_code": None,
        "cohort": None,
        "patient_no": None,
    }


def _empty_properties() -> dict:
    return {
        "readable": False,
        "unreadable_reason": None,
        "vendor": None,
        "objective_power": None,
        "mpp_x": None,
        "mpp_y": None,
        "level_count": None,
        "level0_width": None,
        "level0_height": None,
        "level_downsamples": None,
    }


# --------------------------------------------------------------------------
# Status and primary selection
# --------------------------------------------------------------------------
def assign_status(frame: pd.DataFrame) -> pd.DataFrame:
    """Classify each file and pick one primary slide per specimen.

    A specimen is identified by ``(folder, stem)``: the same tissue saved as both
    ``96.ndpi`` and ``96.tif`` is one specimen in two formats, not two slides.
    """
    out = frame.copy()
    if "status" not in out.columns:
        out["status"] = None
    if "primary" not in out.columns:
        out["primary"] = False

    for (folder, stem), group in out[out["file_name"].notna()].groupby(
        ["folder", out.loc[out["file_name"].notna(), "file_name"].map(lambda n: Path(n).stem)]
    ):
        # Rank the formats, then take the best one that actually opens.
        ranked = group.assign(
            rank=group["file_format"].map(
                lambda f: FORMAT_PREFERENCE.index(f) if f in FORMAT_PREFERENCE else len(FORMAT_PREFERENCE)
            )
        ).sort_values(["readable", "rank"], ascending=[False, True])

        chosen = ranked.index[0] if bool(ranked.iloc[0]["readable"]) else None
        for index, row in ranked.iterrows():
            if index == chosen:
                out.loc[index, "status"] = "ok"
                out.loc[index, "primary"] = True
            elif len(group) > 1:
                out.loc[index, "status"] = "duplicate_conversion"
                out.loc[index, "primary"] = False
            else:
                out.loc[index, "status"] = "unreadable"
                out.loc[index, "primary"] = False
        if chosen is None:
            logger.warning(f"{folder}/{stem}: no readable file among {len(group)} candidate(s)")

    return out


def reconcile_with_clinical(frame: pd.DataFrame, clinical: pd.DataFrame) -> pd.DataFrame:
    """Attach the patient key and cross-check the institution.

    The folder name encodes the institution and the patient number, and the
    clinical table encodes the institution independently. They are redundant on
    purpose: a disagreement means a slide is filed under the wrong patient, which
    would silently mislabel it.
    """
    out = frame.copy()
    out["patient_key"] = [
        sch.patient_key(cohort, int(no)) if cohort and pd.notna(no) else None
        for cohort, no in zip(out["cohort"], out["patient_no"])
    ]

    lookup = clinical.set_index("patient_key")["institution"].to_dict()
    out["institution_from_clinical"] = out["patient_key"].map(lookup)
    out["matched_clinical"] = out["institution_from_clinical"].notna()
    # Three states, so a nullable boolean: True (agree), False (disagree), and NA
    # (no comparison possible because one side is missing). Plain bool would
    # collapse NA into False and manufacture mismatches.
    out["institution_match"] = pd.array(
        [
            None if (a is None or pd.isna(a) or pd.isna(b)) else (a == b)
            for a, b in zip(out["institution_from_folder"], out["institution_from_clinical"])
        ],
        dtype="boolean",
    )

    unmatched = out[out["file_name"].notna() & ~out["matched_clinical"]]
    if len(unmatched):
        # Not a filter: a slide with no clinical record is either a folder
        # naming bug or a patient missing from the workbooks, and both need
        # resolving rather than dropping.
        logger.error(
            f"{len(unmatched)} slide file(s) do not match any clinical patient: "
            f"{_describe(unmatched)}"
        )
        out.loc[unmatched.index, "status"] = "unmatched"
        out.loc[unmatched.index, "primary"] = False

    mismatched = out[out["institution_match"].eq(False).fillna(False)]
    if len(mismatched):
        logger.error(
            f"{len(mismatched)} slide(s) disagree with the clinical institution: "
            f"{_describe(mismatched)}"
        )
    else:
        logger.info("institution cross-check passed for every matched slide")

    # One slide per patient today. Assert it rather than assume it: the moment a
    # patient has two specimens, averaging them silently would be wrong.
    primaries = out[out["primary"]]
    duplicated = primaries["patient_key"][primaries["patient_key"].duplicated()].tolist()
    if duplicated:
        logger.warning(
            f"{len(set(duplicated))} patient(s) have more than one primary slide: "
            f"{sorted(set(duplicated))}. Bag construction must concatenate, not average."
        )
    out["slide_index"] = primaries.groupby("patient_key").cumcount().reindex(out.index)

    return out


def _describe(frame: pd.DataFrame) -> list[str]:
    """Identify rows for a log message, preferring whichever key is available."""
    for column in ("slide_id", "folder", "patient_key"):
        if column in frame.columns:
            return frame[column].astype(str).tolist()
    return [f"row {i}" for i in frame.index]


def build_manifest(wsi_dir: Path = WSI_DIR, clinical_path: Path = CLINICAL_PARQUET) -> pd.DataFrame:
    """Discover, classify and reconcile. The full manifest build."""
    if not Path(clinical_path).exists():
        raise FileNotFoundError(
            f"{clinical_path} not found. Run `python -m ec_wsi_2026.dataset all` first -- the "
            "manifest is a join against the curated clinical table."
        )
    clinical = pd.read_parquet(clinical_path)
    frame = discover_slides(wsi_dir)
    frame = assign_status(frame)
    frame = reconcile_with_clinical(frame, clinical)

    # file_id is the row key; slide_id is the specimen key and is unique only
    # among primaries, since duplicates share it by design.
    duplicated_files = frame["file_id"][frame["file_id"].duplicated()].tolist()
    if duplicated_files:
        raise ValueError(f"manifest file_id is not unique: {sorted(set(duplicated_files))}")
    primaries = frame[frame["primary"]]
    duplicated_slides = primaries["slide_id"][primaries["slide_id"].duplicated()].tolist()
    if duplicated_slides:
        raise ValueError(f"two primaries share a slide_id: {sorted(set(duplicated_slides))}")

    columns = [
        "file_id", "slide_id", "patient_key", "cohort", "patient_no",
        "institution_from_folder", "institution_from_clinical", "institution_match",
        "matched_clinical", "status", "primary", "slide_index",
        "folder", "file_name", "file_path", "file_format", "file_size_bytes", "md5_head_tail",
        "readable", "unreadable_reason", "vendor", "objective_power", "mpp_x", "mpp_y",
        "level_count", "level0_width", "level0_height", "level_downsamples",
    ]
    frame = frame[[c for c in columns if c in frame.columns]]
    counts = frame["status"].value_counts().to_dict()
    logger.info(f"manifest: {len(frame)} row(s), {int(frame['primary'].sum())} primary; {counts}")
    return frame


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------
def standardised_mean_difference(a: pd.Series, b: pd.Series) -> float:
    """SMD between two groups. Scale-free, and unlike a p-value it does not
    shrink towards "no difference" just because the groups are small."""
    a, b = a.dropna().astype(float), b.dropna().astype(float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = ((a.var(ddof=1) + b.var(ddof=1)) / 2) ** 0.5
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def coverage_report(manifest: pd.DataFrame, clinical: pd.DataFrame) -> str:
    """Compare the imaged and non-imaged patients. Renders markdown."""
    imaged_keys = set(manifest.loc[manifest["primary"], "patient_key"].dropna())
    clinical = clinical.copy()
    clinical["imaged"] = clinical["patient_key"].isin(imaged_keys)

    n_imaged = int(clinical["imaged"].sum())
    lines: list[str] = []
    add = lines.append

    add("# Slide coverage")
    add("")
    add(f"Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}.")
    add("")
    add(
        f"**{n_imaged} of {len(clinical)} patients ({n_imaged / len(clinical):.0%}) have a usable "
        f"whole-slide image.**"
    )
    add("")
    add(
        "Whether that subset is representative determines what the imaging models can claim. "
        "The comparisons below are the evidence; the *reason* some patients lack a slide is not "
        "in the data and must come from the data owner. Administrative causes (archive age, site "
        "policy, failed transfers) leave the subset usable. Selective retrieval -- pulling slides "
        "for patients who recurred, or only from sites with digital archives -- does not."
    )
    add("")

    _coverage_by_stratum(add, clinical)
    _coverage_by_endpoint(add, clinical)
    _coverage_baseline(add, clinical)
    _coverage_slides(add, manifest)

    return "\n".join(lines) + "\n"


def _coverage_by_stratum(add, clinical: pd.DataFrame) -> None:
    add("## Coverage by cohort and institution")
    add("")
    table = clinical.pivot_table(
        index="institution", columns="cohort", values="imaged", aggfunc=["sum", "count"]
    )
    rows = []
    for institution in sorted(clinical["institution"].dropna().unique()):
        row = {"institution": institution}
        for cohort in sch.COHORTS:
            subset = clinical[(clinical["institution"] == institution) & (clinical["cohort"] == cohort)]
            imaged = int(subset["imaged"].sum())
            row[cohort] = f"{imaged}/{len(subset)}" if len(subset) else "-"
        rows.append(row)
    total = {"institution": "**Total**"}
    for cohort in sch.COHORTS:
        subset = clinical[clinical["cohort"] == cohort]
        total[cohort] = f"**{int(subset['imaged'].sum())}/{len(subset)}**"
    rows.append(total)
    add(pd.DataFrame(rows).to_markdown(index=False))
    add("")
    _ = table
    add(
        "A site contributing disproportionately many slides dominates the imaging model."
    )
    add("")


def _coverage_by_endpoint(add, clinical: pd.DataFrame) -> None:
    add("## Event counts: imaged subset vs full cohort")
    add("")
    add(
        "**These are the numbers that govern what is analysable**, not the full-cohort counts in "
        "`clinical_audit.md`. An endpoint with adequate events overall can be underpowered in the "
        "imaged subset."
    )
    add("")
    rows = []
    for endpoint in ep.ENDPOINTS:
        event_col = f"{endpoint.name}_event"
        if event_col not in clinical.columns:
            continue
        full = clinical[clinical[event_col].notna()]
        imaged = full[full["imaged"]]
        not_imaged = full[~full["imaged"]]
        events_imaged = int((imaged[event_col] == 1).sum())
        rate_imaged = events_imaged / max(len(imaged), 1)
        rate_other = int((not_imaged[event_col] == 1).sum()) / max(len(not_imaged), 1)
        rows.append(
            {
                "endpoint": endpoint.name,
                "full n": len(full),
                "full events": int((full[event_col] == 1).sum()),
                "imaged n": len(imaged),
                "imaged events": events_imaged,
                "imaged rate": f"{rate_imaged:.0%}",
                "non-imaged rate": f"{rate_other:.0%}",
                "rate diff": f"{rate_imaged - rate_other:+.0%}",
                "max covariates": events_imaged // 10,
            }
        )
    add(pd.DataFrame(rows).to_markdown(index=False))
    add("")
    add(
        "A large **rate diff** is the warning sign: it means slide availability is associated with "
        "the outcome, which is selection bias rather than a sampling accident."
    )
    add("")


def _coverage_baseline(add, clinical: pd.DataFrame) -> None:
    add("## Baseline characteristics: imaged vs non-imaged")
    add("")
    imaged = clinical[clinical["imaged"]]
    other = clinical[~clinical["imaged"]]
    if not len(other):
        add("Every patient has a slide; there is no non-imaged group to compare against.")
        add("")
        return

    continuous = [
        c
        for c in ("age_at_crt", "tumor_length_cm", "total_dose_gy", "treatment_year", "followup_days")
        if c in clinical.columns
    ]
    ordinal = [c for c in ("ct_stage_num", "cn_stage_num", "uicc_stage_num", "histologic_grade_num") if c in clinical.columns]

    rows = []
    for column in continuous + ordinal:
        smd = standardised_mean_difference(imaged[column], other[column])
        rows.append(
            {
                "variable": column,
                "imaged (mean)": f"{imaged[column].astype(float).mean():.1f}",
                "non-imaged (mean)": f"{other[column].astype(float).mean():.1f}",
                "SMD": f"{smd:+.2f}",
                "imbalanced": "**yes**" if abs(smd) > 0.1 else "",
            }
        )
    add(pd.DataFrame(rows).to_markdown(index=False))
    add("")
    add(
        "|SMD| > 0.1 is the conventional threshold for a meaningful imbalance. Several imbalanced "
        "variables mean the imaged subset is not a random sample of the cohort, and any model "
        "trained on it needs that stated as a limitation."
    )
    add("")


def _coverage_slides(add, manifest: pd.DataFrame) -> None:
    add("## Slide inventory")
    add("")
    counts = manifest["status"].value_counts()
    add(pd.DataFrame({"status": counts.index, "n": counts.values}).to_markdown(index=False))
    add("")

    excluded = manifest[manifest["status"] != "ok"]
    if len(excluded):
        add("### Excluded files")
        add("")
        for _, row in excluded.iterrows():
            # A CSV round-trip turns missing values into NaN, which is truthy --
            # `x or default` silently renders "nan".
            name = row["file_name"] if pd.notna(row["file_name"]) else "(empty directory)"
            reason = row.get("unreadable_reason")
            detail = f": {reason}" if pd.notna(reason) else ""
            add(f"- `{row['folder']}/{name}` — **{row['status']}**{detail}")
        add("")

    usable = manifest[manifest["primary"]]
    if len(usable):
        add("### Acquisition metadata (usable slides)")
        add("")
        by_vendor = usable.groupby("vendor").agg(
            n=("slide_id", "count"),
            mpp_min=("mpp_x", "min"),
            mpp_max=("mpp_x", "max"),
            levels_min=("level_count", "min"),
            levels_max=("level_count", "max"),
        )
        add(by_vendor.to_markdown())
        add("")
        add(
            "Scanner vendor is a candidate confounder: it correlates with site, and site "
            "correlates with outcome. If a classifier can predict the institution from tile "
            "colour alone, stain normalisation moves from optional to required."
        )
        add("")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
@app.command()
def build(
    wsi_dir: Path = typer.Option(WSI_DIR, help="Directory of per-slide folders."),
    out: Path = typer.Option(SLIDE_MANIFEST_CSV, help="Where to write the manifest."),
) -> None:
    """Discover slides, read their metadata, and reconcile against the clinical table."""
    frame = build_manifest(wsi_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False, encoding="utf-8")
    logger.success(f"wrote {out}")


@app.command()
def coverage(
    manifest_path: Path = typer.Option(SLIDE_MANIFEST_CSV, help="Manifest to analyse."),
    out: Path = typer.Option(SLIDE_COVERAGE_MD, help="Where to write the report."),
) -> None:
    """Compare imaged and non-imaged patients."""
    if not manifest_path.exists():
        raise typer.BadParameter(f"{manifest_path} not found; run `manifest build` first")
    manifest = pd.read_csv(manifest_path)
    clinical = pd.read_parquet(CLINICAL_PARQUET)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(coverage_report(manifest, clinical), encoding="utf-8")
    logger.success(f"wrote {out}")


@app.command()
def all(
    wsi_dir: Path = typer.Option(WSI_DIR, help="Directory of per-slide folders."),
) -> None:
    """Run build then coverage."""
    build(wsi_dir=wsi_dir, out=SLIDE_MANIFEST_CSV)
    coverage(manifest_path=SLIDE_MANIFEST_CSV, out=SLIDE_COVERAGE_MD)


if __name__ == "__main__":
    app()
