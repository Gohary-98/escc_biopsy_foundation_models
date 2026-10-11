import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# The data root is env-overridable (`.env`): the slides and the feature
# extraction usually live on a GPU machine, the modelling can run anywhere.
DATA_DIR = Path(os.getenv("EC_WSI_DATA_DIR", PROJ_ROOT / "data"))
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --------------------------------------------------------------------------
# Project-specific paths
# --------------------------------------------------------------------------
REFERENCES_DIR = PROJ_ROOT / "references"
RUNS_DIR = MODELS_DIR / "runs"

# Slides may live outside data/, so this is resolved separately.
WSI_DIR = Path(os.getenv("EC_WSI_SLIDE_DIR", RAW_DATA_DIR / "histopathology"))

# The clinical workbooks are source data, not derived output.
CLINICAL_DIR = RAW_DATA_DIR / "clinical_data"
CLINICAL_DEF_XLSX = CLINICAL_DIR / "definitive_clean.xlsx"
CLINICAL_NEO_XLSX = CLINICAL_DIR / "neoadjuvant_clean.xlsx"

# Interim clinical artifacts (includes PHI — never promoted to processed/)
CLINICAL_LONG_RAW_PARQUET = INTERIM_DATA_DIR / "clinical_long_raw.parquet"
CLINICAL_COERCED_PARQUET = INTERIM_DATA_DIR / "clinical_coerced.parquet"
CLINICAL_PARSE_FAILURES_CSV = INTERIM_DATA_DIR / "clinical_parse_failures.csv"
PATIENT_ID_MAP_PARQUET = INTERIM_DATA_DIR / "patient_id_map.parquet"

# De-identified analysis tables (pseudonymous patient keys only).
CLINICAL_PARQUET = PROCESSED_DATA_DIR / "clinical.parquet"
LABELS_PARQUET = PROCESSED_DATA_DIR / "labels.parquet"
SLIDE_MANIFEST_CSV = PROCESSED_DATA_DIR / "slide_manifest.csv"

# Cross-validation artifacts. folds.csv is written once; every model reads the
# same folds so comparisons are paired.
FOLDS_CSV = PROCESSED_DATA_DIR / "folds.csv"
FOLDS_MANIFEST_JSON = PROCESSED_DATA_DIR / "folds_manifest.json"

# Histopathology artifacts. TITAN_JOB_DIR and UNI_JOB_DIR hold TRIDENT's raw,
# multi-file-per-patient output trees (scripts/run_trident.sh writes there);
# SLIDE_EMB_DIR holds the small, consolidated, one-row-per-patient tables the
# models read.
HISTO_DIR = PROCESSED_DATA_DIR / "histopathology"
SLIDE_EMB_DIR = HISTO_DIR / "slide_embeddings"
TILES_DIR = INTERIM_DATA_DIR / "tiles"

TITAN_JOB_DIR = HISTO_DIR / "titan"
TITAN_EMBEDDINGS_PARQUET = SLIDE_EMB_DIR / "titan.parquet"

# Mean-pooled tile features of UNI2-h and CONCH v1.5 (`modeling consolidate-mean`).
UNI_MEAN_EMBEDDINGS_PARQUET = SLIDE_EMB_DIR / "uni_mean.parquet"
CONCH_MEAN_EMBEDDINGS_PARQUET = SLIDE_EMB_DIR / "conch_mean.parquet"

# Staged TRIDENT input: one symlink per usable primary slide, named
# <patient_key><suffix> (scripts/_stage_trident_input.py), so TRIDENT's output
# filenames are patient keys.
TRIDENT_INPUT_DIR = INTERIM_DATA_DIR / "trident_input"

# UNI2-h tile features live in the TRIDENT job dir shared with the resnet50
# segmentation/patching run (scripts/run_trident.sh points both at
# data/interim/tiles so UNI reuses the same tile coordinates).
UNI_JOB_DIR = TILES_DIR

# Reference documents authored by this project
DATA_DICTIONARY_MD = REFERENCES_DIR / "data_dictionary.md"
CODE_LEGENDS_YAML = REFERENCES_DIR / "code_legends.yaml"
ENDPOINT_DEFINITIONS_MD = REFERENCES_DIR / "endpoint_definitions.md"
COVARIATES_YAML = REFERENCES_DIR / "covariates.yaml"

# Reports
CLINICAL_AUDIT_MD = REPORTS_DIR / "clinical_audit.md"
SLIDE_COVERAGE_MD = REPORTS_DIR / "slide_coverage.md"

# --------------------------------------------------------------------------
# Secrets and runtime settings
# --------------------------------------------------------------------------
# Foundation model weights (UNI, UNI2-h, CONCH, TITAN) are gated on Hugging Face
# and need an approved account.
HF_TOKEN = os.getenv("HF_TOKEN")

DEFAULT_SEED = 20260101

# OpenSlide on Windows needs its DLL directory registered before the first
# `import openslide`. The `openslide-bin` wheel normally handles this; set
# OPENSLIDE_PATH in .env only when falling back to a manual binary install.
OPENSLIDE_PATH = os.getenv("OPENSLIDE_PATH")


def add_openslide_dll_directory() -> None:
    """Register the OpenSlide DLL directory on Windows, if OPENSLIDE_PATH is set.

    No-op on POSIX and when the `openslide-bin` wheel is doing its job. Call this
    before importing `openslide`, not after — `os.add_dll_directory` has no effect
    once the extension module has been loaded.
    """
    if not OPENSLIDE_PATH or not hasattr(os, "add_dll_directory"):
        return
    path = Path(OPENSLIDE_PATH)
    if not path.is_dir():
        logger.warning(f"OPENSLIDE_PATH is set but not a directory: {path}")
        return
    os.add_dll_directory(str(path))
    logger.debug(f"Registered OpenSlide DLL directory: {path}")


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
# The sink must target stderr and flush per message: tqdm.write defaults to
# stdout, which is block-buffered under `nohup ... > log` redirection, so
# INFO lines sat in an 8 KB buffer for hours while stderr (warnings,
# tracebacks) skipped ahead — healthy runs looked hung.
try:
    from tqdm import tqdm

    def _tqdm_sink(message: str) -> None:
        tqdm.write(message, end="", file=sys.stderr)
        sys.stderr.flush()

    logger.remove(0)
    logger.add(_tqdm_sink, colorize=True)
except ModuleNotFoundError:
    pass
