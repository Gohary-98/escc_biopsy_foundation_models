#!/usr/bin/env bash
# Wrapper around TRIDENT feature extraction.
#
# Usage (run from the repo root, on the GPU machine holding the slides):
#   bash scripts/run_trident.sh stage        # build data/interim/trident_input from the manifest
#   bash scripts/run_trident.sh smoke        # 4-slide sanity check (2 Aperio, 2 Hamamatsu), foreground
#   bash scripts/run_trident.sh resnet50     # tissue segmentation + 20x/256 px tiling (coords for UNI), background
#   bash scripts/run_trident.sh titan        # CONCH v1.5 tile features (20x/512 px) -> TITAN slide embeddings, background
#   bash scripts/run_trident.sh uni          # UNI2-h tile features on the resnet50 tile grid, background
#   bash scripts/run_trident.sh status       # which of the above are currently running
#   bash scripts/run_trident.sh inspect <resnet50|titan|uni>
#                                             # output dirs, .h5 count, and one file's internal structure
#
# resnet50/titan/uni run detached and log to logs/trident_<name>_<timestamp>.log.
# Running jobs are identified in the process list by --job_dir and encoder flag.
#
# Override TRIDENT_DIR if your clone isn't at ~/trident_repo:
#   TRIDENT_DIR=/path/to/trident bash scripts/run_trident.sh resnet50

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRIDENT_DIR="${TRIDENT_DIR:-$HOME/trident_repo}"
INPUT_DIR="$REPO_DIR/data/interim/trident_input"
SMOKE_DIR="$REPO_DIR/data/interim/trident_smoke"
LOG_DIR="$REPO_DIR/logs"
mkdir -p "$LOG_DIR"

# Output location per encoder. uni shares resnet50's job_dir: both use 20x and
# 256 px tiles, so `--task feat` reuses the segmentation and tile coordinates of
# the resnet50 run; TRIDENT writes each encoder to its own features_<name>/.
declare -A JOB_DIRS=(
  [resnet50]="$REPO_DIR/data/interim/tiles"
  [titan]="$REPO_DIR/data/processed/histopathology/titan"
  [uni]="$REPO_DIR/data/interim/tiles"
)

# Encoder flag used to tell runs that share a job_dir apart.
declare -A MATCH_EXTRA=(
  [resnet50]="patch_encoder resnet50"
  [titan]="slide_encoder titan"
  [uni]="patch_encoder uni_v2"
)

stage() {
  conda run -n ec_wsi_2026 --no-capture-output python "$REPO_DIR/scripts/_stage_trident_input.py"
}

run_encoder() {
  # run_encoder <name> <wsi_dir> <job_dir> <extra run_batch_of_slides.py flags...>
  local name="$1"; shift
  local wsi_dir="$1"; shift
  local job_dir="$1"; shift
  local match="run_batch_of_slides.py.*--${MATCH_EXTRA[$name]}.*--job_dir $job_dir"
  local timestamp logfile

  if pgrep -f "$match" > /dev/null 2>&1; then
    echo "a '$name' run already appears active (matched on --job_dir $job_dir) - not starting another."
    echo "  see:   pgrep -af '$match'"
    echo "  stop:  pkill -9 -f '$match'"
    return 1
  fi

  timestamp="$(date +%Y%m%d_%H%M%S)"
  logfile="$LOG_DIR/trident_${name}_${timestamp}.log"
  mkdir -p "$job_dir"

  (
    cd "$TRIDENT_DIR"
    conda run -n trident --no-capture-output python run_batch_of_slides.py "$@" \
      --wsi_dir "$wsi_dir" --job_dir "$job_dir" --gpus 0
  ) > "$logfile" 2>&1 &
  disown

  echo "started '$name'"
  echo "  watch: tail -f $logfile"
  echo "  stop:  pkill -9 -f '$match'"
}

status() {
  local any=0
  for name in "${!JOB_DIRS[@]}"; do
    if pgrep -f "run_batch_of_slides.py.*--${MATCH_EXTRA[$name]}.*--job_dir ${JOB_DIRS[$name]}" > /dev/null 2>&1; then
      echo "$name: RUNNING"
      any=1
    fi
  done
  [[ "$any" -eq 0 ]] && echo "nothing running"
  return 0
}

inspect() {
  local name="${1:-}"
  local job_dir="${JOB_DIRS[$name]:-}"
  if [[ -z "$job_dir" ]]; then
    echo "usage: $0 inspect <resnet50|titan|uni>"
    return 1
  fi
  if [[ ! -d "$job_dir" ]]; then
    echo "no output directory yet at $job_dir (has '$name' been run?)"
    return 1
  fi

  echo "=== directories under $job_dir ==="
  find "$job_dir" -maxdepth 3 -type d

  echo
  echo "=== .h5 file count (every subdirectory) ==="
  find "$job_dir" -name "*.h5" | wc -l

  # Prefer the slide-level output (slide_features_<name>/) when it exists.
  local target_dir sample
  target_dir="$(find "$job_dir" -type d -name "slide_features_*" -print -quit)"
  if [[ -n "$target_dir" ]]; then
    echo
    echo "=== slide-level output: $target_dir ==="
    find "$target_dir" -name "*.h5" | wc -l
    echo "file(s) here"
    sample="$(find "$target_dir" -name "*.h5" -print -quit)"
  else
    sample="$(find "$job_dir" -name "*.h5" -print -quit)"
  fi

  if [[ -z "$sample" ]]; then
    echo "no .h5 files found under $job_dir"
    return 0
  fi

  echo
  echo "=== structure of: $sample ==="
  conda run -n trident --no-capture-output python -c '
import sys
import h5py

with h5py.File(sys.argv[1], "r") as handle:
    handle.visititems(
        lambda name, obj: print(" ", name, getattr(obj, "shape", ""), getattr(obj, "dtype", ""))
    )
    print("attrs:", dict(handle.attrs))
' "$sample"
}

case "${1:-}" in
  stage)
    stage
    ;;
  smoke)
    if [[ ! -d "$INPUT_DIR" ]] || [[ -z "$(ls -A "$INPUT_DIR" 2>/dev/null)" ]]; then
      echo "run '$0 stage' first (no staged slides found in $INPUT_DIR)"
      exit 1
    fi
    rm -rf "$SMOKE_DIR"
    mkdir -p "$SMOKE_DIR"
    # globbing rather than `find | head`, which fails under pipefail (SIGPIPE)
    svs_files=("$INPUT_DIR"/*.svs)
    ndpi_files=("$INPUT_DIR"/*.ndpi)
    for f in "${svs_files[@]:0:2}" "${ndpi_files[@]:0:2}"; do
      [[ -e "$f" ]] && ln -sf "$f" "$SMOKE_DIR/"
    done
    echo "staged $(ls "$SMOKE_DIR" | wc -l) smoke-test slide(s) -> $SMOKE_DIR"
    echo "running in the foreground (it's only 4 slides) ..."
    (
      cd "$TRIDENT_DIR"
      conda run -n trident --no-capture-output python run_batch_of_slides.py \
        --task all --patch_encoder resnet50 --mag 20 --patch_size 256 \
        --wsi_dir "$SMOKE_DIR" --job_dir "$REPO_DIR/data/interim/trident_smoke_out" --gpus 0
    )
    ;;
  resnet50)
    run_encoder resnet50 "$INPUT_DIR" "${JOB_DIRS[resnet50]}" \
      --task all --patch_encoder resnet50 --mag 20 --patch_size 256
    ;;
  titan)
    run_encoder titan "$INPUT_DIR" "${JOB_DIRS[titan]}" \
      --task all --slide_encoder titan --patch_size 512 --mag 20
    ;;
  uni)
    run_encoder uni "$INPUT_DIR" "${JOB_DIRS[uni]}" \
      --task feat --patch_encoder uni_v2 --mag 20 --patch_size 256
    ;;
  status)
    status
    ;;
  inspect)
    inspect "${2:-}"
    ;;
  *)
    echo "usage: $0 {stage|smoke|resnet50|titan|uni|status|inspect <name>}"
    exit 1
    ;;
esac
