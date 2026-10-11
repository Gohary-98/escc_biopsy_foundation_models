# Endpoint definitions

The outcome definitions the code implements (`ec_wsi_2026/clinical/endpoints.py`).
The duration columns of the source workbooks are empty, so every endpoint is
derived from dates.

## Time origin

**`Date Start CRT`, for both treatment arms.** It is the only index date
recorded for every patient, and the only choice that permits pooling the arms.
Anchoring the neoadjuvant arm at surgery would introduce immortal time (a
patient anchored at surgery must have survived from CRT start to surgery) and
bias the comparison in favour of the surgical arm.

## Censoring date

**The maximum over every non-null follow-up date**, not a preference-ordered
coalesce:

```
censor_date = max(last_date_alive, last_fu_date, date_progress,
                  date_local_failure, date_death, date_ct_post_crt, date_op)
```

Neither follow-up column dominates the other, so a coalesce in either direction
would discard a known contact and understate follow-up. The column that
supplied the maximum is recorded in `censor_source`.

## Endpoints used in the paper

| Name | Event | Origin | Role |
|---|---|---|---|
| `pfs` | documented progression **or** death from any cause, whichever first | CRT start | primary |
| `os` | death from any cause | CRT start | secondary |

Patients without an event are censored at `censor_date`. A patient without a
recorded vital status is excluded from `os` only.

**PFS counts death as an event.** If the registry never recorded progression
for patients who died without a restaging scan, treating those deaths as
censored would make the censoring informative and bias every estimate
optimistically.

`dataset all` additionally derives further endpoints from the same dates
(time to progression, local-failure-free survival, landmark variants and the
pathological response labels); they are not used in the paper.

## Derived covariates

| Column | Definition |
|---|---|
| `age_at_crt` | `(Date Start CRT − Date of Birth) / 365.25`; the recorded age where the birth date is absent |
| `cn_involved` | `cn_stage_num > 0`; patients recorded only as node-positive (`'+'`) count as involved |
| `crt_duration_days` | `Date End CRT − Date Start CRT` (nulled where negative) |
| `crt_end_to_op_days` | `Date OP − Date End CRT` (neoadjuvant arm; nulled where negative) |
| `followup_days` | `censor_date − Date Start CRT` |

## Validation

`endpoints.validate` raises on any negative endpoint duration, an observed event
without an event date, local failure without progression, a censoring date
before the time origin, or an unresolvable or implausible (< 18, > 100) age.
An impossible intra-treatment interval (a transcription error in the source
workbook) is set to missing and flagged in `date_anomaly` rather than corrected;
it does not touch any endpoint, which anchor on `Date Start CRT` only.

## Modelling specification

**Covariates** are pre-specified in [covariates.yaml](covariates.yaml):
age, sex, treatment arm, clinical T stage, clinical nodal involvement,
histologic grade and tumour length. `modeling/covariates.py` encodes them.

**Missing data.** Median imputation for numeric covariates, fitted inside each
training fold (and each inner-CV training split) — never on pooled data.
Outcomes are never imputed: patients with a missing event or time are excluded
at fold construction. Columns that are constant within a training fold are
dropped inside that fold's pipeline.

**Cross-validation.** Folds are a persisted artifact
(`data/processed/folds.csv` + `folds_manifest.json`), built once by
`python -m ec_wsi_2026.modeling folds` and read by every model, so the clinical,
imaging-only and fusion models are evaluated on identical splits — the
precondition for paired differences. The manifest stores a hash of each
endpoint's eligible cohort; stale folds are refused, not silently rebuilt.
`StratifiedGroupKFold` (grouped by `patient_key`), 5 folds × 5 repeats; the
stratification key is coarsened from institution × treatment arm × event until
every stratum holds at least five patients, which for PFS and OS gives
**treatment arm × event**.

A fourth site (14 patients, no slides) is pinned to the training folds: its
patients never appear in a test fold. Having no slides, it drops out of every
model trained on the imaged subset.

**Inference.** The comparison of each imaging-only or fusion model with the
clinical model is the paired bootstrap difference on identical patients, never
two overlapping independent confidence intervals.
