#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = ec_wsi_2026
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# SETUP AND CHECKS                                                              #
#################################################################################

## Set up the conda environment
.PHONY: create_environment
create_environment:
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"

## Install the package with the analysis, slide and test dependencies
.PHONY: requirements
requirements:
	pip install -r requirements.txt
	pip install -e ".[analysis,wsi,dev]"

## Run the test suite
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest -q

## Lint with ruff
.PHONY: lint
lint:
	ruff check ec_wsi_2026 tests paper scripts

## Delete compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

#################################################################################
# 1. DATA                                                                       #
#################################################################################

## Clinical curation: workbooks -> data/processed/{clinical,labels}.parquet + audit
.PHONY: data
data:
	$(PYTHON_INTERPRETER) -m ec_wsi_2026.dataset all

## Slide manifest (slide -> patient join, MPP, vendor) + imaging-coverage report
.PHONY: manifest
manifest:
	$(PYTHON_INTERPRETER) -m ec_wsi_2026.manifest all

## Build and persist the 5 x 5 cross-validation folds (PFS and OS)
.PHONY: folds
folds:
	$(PYTHON_INTERPRETER) -m ec_wsi_2026.modeling folds

#################################################################################
# 2. FEATURE EXTRACTION (GPU machine with the slides and TRIDENT)               #
#################################################################################

## Stage the usable primary slides for TRIDENT as <patient_key>.<ext> symlinks
.PHONY: trident-stage
trident-stage:
	bash scripts/run_trident.sh stage

## TRIDENT: tissue segmentation + 20x / 256 px tiling (tile grid reused by UNI2-h)
.PHONY: trident-resnet50
trident-resnet50:
	bash scripts/run_trident.sh resnet50

## TRIDENT: CONCH v1.5 tile features (20x / 512 px) and TITAN slide embeddings
.PHONY: trident-titan
trident-titan:
	bash scripts/run_trident.sh titan

## TRIDENT: UNI2-h tile features on the 256 px grid (after trident-resnet50)
.PHONY: trident-uni
trident-uni:
	bash scripts/run_trident.sh uni

## Consolidate TITAN and mean-pooled UNI2-h / CONCH v1.5 embeddings into per-patient parquets
.PHONY: consolidate
consolidate:
	$(PYTHON_INTERPRETER) -m ec_wsi_2026.modeling consolidate-titan
	$(PYTHON_INTERPRETER) -m ec_wsi_2026.modeling consolidate-mean

#################################################################################
# 3. MODELS                                                                     #
#################################################################################

## Train the 7 models (clinical, 3 imaging-only, 3 fusion) x PFS / OS out of fold
.PHONY: train
train:
	$(PYTHON_INTERPRETER) -m ec_wsi_2026.modeling train

#################################################################################
# 4. PAPER TABLES AND FIGURES (-> paper/output/)                                #
#################################################################################

## Table 2 and Supplementary Tables E1-E3 (C-index, td-AUC, paired bootstrap, log-rank)
.PHONY: paper-tables
paper-tables:
	$(PYTHON_INTERPRETER) paper/per_fold_tables.py

## Table 1 and the treatment numbers of Section 2.2
.PHONY: paper-table1
paper-table1:
	$(PYTHON_INTERPRETER) paper/table1.py

## Calibration at 12 months (Section 3.5, Supplementary Table E4); refits every fold model
.PHONY: paper-calibration
paper-calibration:
	$(PYTHON_INTERPRETER) paper/calibration.py

## Figures 2 and 3 (after paper-tables)
.PHONY: paper-figures
paper-figures:
	$(PYTHON_INTERPRETER) paper/figures.py

## Section 3.6: fusion vs clinical by treatment arm, and the first-year PFS events
.PHONY: paper-exploratory
paper-exploratory:
	$(PYTHON_INTERPRETER) paper/by_arm.py
	$(PYTHON_INTERPRETER) paper/early_events.py

## Every table, figure and number of the paper from the trained models
.PHONY: paper
paper: paper-table1 paper-tables paper-figures paper-calibration paper-exploratory

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z0-9_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

## Show this help
help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
