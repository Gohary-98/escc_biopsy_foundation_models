# Pathology foundation-model embeddings of the pretreatment biopsy for survival prediction in ESCC treated with chemoradiotherapy

Analysis code for the paper (*Advances in Radiation Oncology*, submitted). It
compares Cox models built on **routine clinical variables**, on **frozen
pathology foundation-model embeddings** of one pretreatment H&E biopsy slide
(TITAN, CONCH v1.5, UNI2-h), and on an **early fusion** of both, for
progression-free survival (primary) and overall survival (secondary) in 180
patients with esophageal squamous cell carcinoma treated with definitive or
neoadjuvant chemoradiotherapy at three German university hospitals.

**Data availability.** Patient data and whole-slide images cannot be shared
because of the ethics approvals, so this repository contains code only. Every
script runs on the de-identified tables the pipeline derives from the clinical
workbooks and the slides.

## Pipeline

```
clinical workbooks --dataset all-->  data/processed/{clinical,labels}.parquet
slides             --manifest all--> data/processed/slide_manifest.csv
                   --folds------->   data/processed/folds.csv (5 x 5 stratified, persisted, read by every model)
slides --TRIDENT (run_trident.sh)--> tile features / TITAN slide embeddings
       --consolidate-------------->  data/processed/histopathology/slide_embeddings/{titan,uni_mean,conch_mean}.parquet
       --train------------------->   models/runs/paper/<endpoint>_<model>_stratified/oof.csv  (7 models x PFS / OS)
       --paper/*.py-------------->   paper/output/  (every table, figure and number of the paper)
```

## Installation

Python 3.10.

```bash
conda create -n ec_wsi_2026 python=3.10 -y && conda activate ec_wsi_2026
pip install -r requirements.txt          # the exact versions the results were produced with
pip install -e ".[analysis,wsi,dev]"
cp .env.example .env                     # data and slide paths, Hugging Face token
pytest -q
```

Feature extraction uses [TRIDENT](https://github.com/mahmoodlab/TRIDENT) in its
own conda environment (`trident`) on a GPU machine; `scripts/run_trident.sh`
expects a clone at `~/trident_repo` (override with `TRIDENT_DIR`). TITAN,
CONCH v1.5 and UNI2-h are gated on Hugging Face and need an approved account.

## Reproducing the paper

`make help` lists every target. In order:

| Step | Command | Output |
|---|---|---|
| Clinical curation | `make data` | `data/processed/clinical.parquet`, `labels.parquet`, curation audit |
| Slide manifest | `make manifest` | `data/processed/slide_manifest.csv` (slide-to-patient join, pixel size, scanner) |
| Folds | `make folds` | `data/processed/folds.csv`, `folds_manifest.json` |
| Feature extraction (GPU) | `make trident-stage trident-resnet50 trident-titan trident-uni` | TRIDENT output trees |
| Consolidation | `make consolidate` | one embedding parquet per encoder, one row per patient |
| Models | `make train` | out-of-fold predictions of the 7 models for PFS and OS |
| Tables and figures | `make paper` | `paper/output/` |

### Where each result comes from

| Paper item | Script | Output in `paper/output/` |
|---|---|---|
| Table 1; treatment numbers and follow-up in Sections 2.2 and 3.1 | `paper/table1.py` | `table1.tex`, `table1.json` |
| Table 2 (C-index, PFS / OS / OS with the PFS-trained models; paired bootstrap; t-test; log-rank) | `paper/per_fold_tables.py` | `table2_cindex.tex`, `per_fold_tables.json` |
| Supplementary Tables E1–E3 (time-dependent AUC) | `paper/per_fold_tables.py` | `tabE1_tdauc_pfs.tex`, `tabE2_tdauc_os.tex`, `tabE3_tdauc_os_pfshead.tex` |
| Figure 2 (time-dependent AUC) and Figure 3 (Kaplan–Meier by risk group) | `paper/figures.py` | `figure2_auc_time.*`, `figure3_km_pfs.*` |
| Section 3.5 and Supplementary Table E4 (calibration at 12 months) | `paper/calibration.py` | `calibration_readouts.csv`, `tabE4_calibration.tex` |
| Section 3.6 (fusion vs clinical by treatment arm) | `paper/by_arm.py` | `by_arm.json` |
| Section 3.6 (first-year PFS events, deaths without progression) | `paper/early_events.py` | `early_events.json` |

## Methods in code

- **Models** (`ec_wsi_2026/modeling/models.py`): ridge Cox proportional hazards
  (lifelines). *Clinical*: age, sex, treatment arm, clinical T stage, nodal
  involvement, histologic grade, tumour length (`references/covariates.yaml`),
  median imputation and standardisation. *Imaging-only*: one embedding per
  slide (TITAN 768-d; mean-pooled CONCH v1.5 768-d; mean-pooled UNI2-h 1536-d),
  standardised and reduced by PCA. *Fusion*: the clinical block and the
  principal components in one Cox model with one shared penalty.
- **Validation**: 5-fold cross-validation stratified by treatment arm and event,
  repeated 5 times (`ec_wsi_2026/modeling/folds.py`); the folds are built once
  and read by every model, so all comparisons are paired. Inside each training
  fold, a five-fold stratified inner cross-validation selects the number of
  components (8, 16, 32) and the penalty (0.01, 0.1, 1, 10) by Harrell's C.
  Imputation, scaling and PCA are fitted inside every training split.
- **Evaluation** (`paper/per_fold_tables.py`): Harrell's C in each of the 25 test
  folds, mean ± SD over folds; time-dependent AUC (Uno, IPCW) at 6, 12, 24 and
  48 months on the pooled test folds of each repeat, mean over repeats;
  95% CIs and the paired difference vs the clinical model from 2000 patient
  bootstrap resamples; risk groups split at each fold model's training-fold
  median risk (majority vote over repeats), log-rank test.
- **Calibration** (`paper/calibration.py`): each fold model refitted with its own
  Breslow baseline hazard and applied to its test patients; calibration in the
  large, calibration slope and Brier score at 12 months.

## Repository layout

```
├── ec_wsi_2026/
│   ├── clinical/            <- workbook loading, coercion, endpoint derivation, audit
│   ├── dataset.py           <- `python -m ec_wsi_2026.dataset all`
│   ├── manifest.py          <- `python -m ec_wsi_2026.manifest all`
│   ├── config.py            <- paths (overridable in .env)
│   └── modeling/
│       ├── folds.py         <- persisted 5 x 5 folds with a cohort hash
│       ├── covariates.py    <- clinical covariate specification and encoding
│       ├── survival.py      <- clinical ridge Cox fit (inner-CV penalty)
│       ├── models.py        <- the seven models, out-of-fold training
│       ├── imaging.py       <- TRIDENT output -> per-patient embedding tables
│       ├── evaluation.py    <- Harrell's C, IPCW time-dependent AUC
│       └── runs.py          <- out-of-fold run IO
├── paper/                   <- one script per table / figure / reported number
├── references/              <- endpoint definitions, covariate set, code legends, data dictionary
├── scripts/                 <- TRIDENT wrapper and slide staging
└── tests/
```

## Citation

[MISSING: citation once published]

## License

The code in this repository is released under the [MIT License](LICENSE).
The pathology foundation models it uses (TITAN, CONCH v1.5, UNI2-h) are not
part of this repository and are distributed by their authors under their own
terms; their weights are downloaded through TRIDENT from Hugging Face.
