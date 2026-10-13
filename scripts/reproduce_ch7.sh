#!/usr/bin/env bash
# Reproduce the Chapter 7 results (results/ch7_v2) from the two upstream
# repositories.
#
# Steps, in order:
#   train     (a) train the four branch models in the upstream repositories
#   assemble  (b) copy their test predictions into $RESULTS_DIR/<cell>/
#   fusion    (c) fusion metrics -> $RESULTS_DIR/fusion_recompute_matched.csv
#   saliency  (d) Integrated Gradients (fMRI) and Grad-CAM (sMRI) per cell
#   llm       (e) optional LLM reports, explainer and clinician styles
#                 (dry run unless OPENAI_API_KEY is set)
#
# Usage:
#   scripts/reproduce_ch7.sh all                       # train assemble fusion saliency
#   scripts/reproduce_ch7.sh assemble fusion saliency  # models already trained
#   scripts/reproduce_ch7.sh llm
#
# Environment:
#   ANXIETY_SMRI_REPO  anxiety-smri-classification checkout (default: sibling directory)
#   ANXIETY_FMRI_REPO  anxiety-fmri-classification checkout (default: sibling directory)
#   RESULTS_DIR        output root (default: <this repo>/results/ch7_v2)
#   PYTHON             Python interpreter (default: python3)
#
# The training step expects the upstream data to be prepared as described in
# the upstream READMEs: GSP slices and participants table and ABIDE ANTs
# slices in anxiety-smri-classification (README.md, abide/README.md); GSP and
# ABIDE voxel fingerprints, data/gsp_split.json (scripts/make_gsp_inputs.py)
# and data/abide/abide_split_seed42.json in anxiety-fmri-classification.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT="$(dirname "$REPO")"
export ANXIETY_SMRI_REPO="${ANXIETY_SMRI_REPO:-$PARENT/anxiety-smri-classification}"
export ANXIETY_FMRI_REPO="${ANXIETY_FMRI_REPO:-$PARENT/anxiety-fmri-classification}"
SMRI="$ANXIETY_SMRI_REPO"
FMRI="$ANXIETY_FMRI_REPO"
RESULTS_DIR="${RESULTS_DIR:-$REPO/results/ch7_v2}"
PY="${PYTHON:-python3}"

# Upstream run directories (relative to their repositories).
SMRI_RUN_GSP=results/ch7_gsp_full_preprocessed_axial_ResNet50
SMRI_RUN_ABIDE=results/ch7_abide_ants_axial_ResNet50
SLICES_GSP="$SMRI/data/GSP/slices/preprocessed/axial"
SLICES_ABIDE="$SMRI/data/ABIDE/slices/raw/axial"

# cell                              dataset  atlas                fMRI run (in $FMRI)
CELLS="
gsp_aal_voxel                       gsp      aal_roi_atlas        results/ch5/gsp_classification_voxel_aal_roi_atlas
gsp_schaefer_voxel                  gsp      schaefer_200_7n_yeo  results/ch5/gsp_classification_voxel_schaefer_200_7n_yeo
abide_aal_voxel_sharedsplit         abide    aal_roi_atlas        results/ch7/abide_voxel_aal_sharedsplit
abide_schaefer_voxel_sharedsplit    abide    schaefer_200_7n_yeo  results/ch7/abide_voxel_schaefer_sharedsplit
"

smri_run () {   # dataset -> structural run directory
  case "$1" in
    gsp)   echo "$SMRI/$SMRI_RUN_GSP" ;;
    abide) echo "$SMRI/$SMRI_RUN_ABIDE" ;;
  esac
}

slices_dir () {  # dataset -> structural slice directory
  case "$1" in
    gsp)   echo "$SLICES_GSP" ;;
    abide) echo "$SLICES_ABIDE" ;;
  esac
}

# (a) Train the branch models ------------------------------------------------
step_train () {
  if [ ! -f "$FMRI/data/gsp_split.json" ]; then
    echo "missing $FMRI/data/gsp_split.json: run scripts/make_gsp_inputs.py in anxiety-fmri-classification first" >&2
    exit 1
  fi

  echo "== (a1) GSP structural branch: full cohort, GSP split shared with the fMRI branch"
  ( cd "$SMRI" && \
    EXPERIMENT_TYPE=preprocessed VIEW_MODE=axial BASE_MODEL=ResNet50 \
    USE_GSP_SUBSET=0 SPLIT_FILE="$FMRI/data/gsp_split.json" \
    EXPERIMENT_NAME=ch7_gsp_full_preprocessed_axial_ResNet50 \
    "$PY" src/train.py )

  echo "== (a2) ABIDE structural branch: ANTs cortical thickness, shared ABIDE split"
  ( cd "$SMRI" && \
    EXPERIMENT_TYPE=raw VIEW_MODE=axial BASE_MODEL=ResNet50 \
    EXPERIMENT_NAME=ch7_abide_ants_axial_ResNet50 \
    "$PY" -m abide.train )

  echo "== (a3) GSP functional branch: voxel 3D CNN, AAL and Schaefer-200"
  for atlas in aal_roi_atlas schaefer_200_7n_yeo; do
    ( cd "$FMRI" && "$PY" run_experiments.py classify --dataset gsp --atlas "$atlas" --connectivity voxel )
  done

  echo "== (a4) ABIDE functional branch: voxel 3D CNN on the shared ABIDE split"
  bash "$FMRI/run_abide_sharedsplit.sh"
}

# (b) Assemble the per-cell prediction bundles -------------------------------
step_assemble () {
  while read -r cell dataset atlas fmri_run <&3; do
    [ -z "$cell" ] && continue
    echo "== (b) $cell"
    mkdir -p "$RESULTS_DIR/$cell/smri" "$RESULTS_DIR/$cell/fmri"
    cp "$(smri_run "$dataset")/predictions_slice.csv" "$RESULTS_DIR/$cell/smri/"
    cp "$FMRI/$fmri_run/predictions_subject.csv" "$RESULTS_DIR/$cell/fmri/"
  done 3<<< "$CELLS"
}

# (c) Fusion metrics ----------------------------------------------------------
step_fusion () {
  echo "== (c) fusion metrics"
  "$PY" "$REPO/src/recompute_fusion.py" --results-dir "$RESULTS_DIR" \
      --out "$RESULTS_DIR/fusion_recompute_matched.csv"
}

# (d) Saliency ----------------------------------------------------------------
step_saliency () {
  # Structural branch: one Grad-CAM run per dataset (the model does not
  # depend on the atlas), copied into both cells of that dataset.
  for dataset in gsp abide; do
    echo "== (d) $dataset structural Grad-CAM"
    run="$(smri_run "$dataset")"
    "$PY" "$REPO/src/report_saliency.py" \
        --model           "$run/best.keras" \
        --predictions-csv "$run/predictions_slice.csv" \
        --modality        smri \
        --slices-dir      "$(slices_dir "$dataset")" \
        --out             "$RESULTS_DIR/${dataset}_smri_saliency"
  done

  while read -r cell dataset atlas fmri_run <&3; do
    [ -z "$cell" ] && continue
    echo "== (d) $cell functional Integrated Gradients"
    "$PY" "$REPO/src/report_saliency.py" \
        --model           "$FMRI/$fmri_run/best.keras" \
        --predictions-csv "$FMRI/$fmri_run/predictions_subject.csv" \
        --dataset         "$dataset" \
        --modality        fmri \
        --connectivity    voxel \
        --atlas-name      "$atlas" \
        --atlas-nifti     "$FMRI/src/templates/atlases/$atlas.nii.gz" \
        --cache-dir       "$FMRI/data/cache" \
        --out             "$RESULTS_DIR/$cell/fmri/saliency"
    mkdir -p "$RESULTS_DIR/$cell/smri/saliency"
    cp "$RESULTS_DIR/${dataset}_smri_saliency"/* "$RESULTS_DIR/$cell/smri/saliency/"
  done 3<<< "$CELLS"
}

# (e) Optional LLM reports ----------------------------------------------------
step_llm () {
  local runs=()
  while read -r cell _ <&3; do
    [ -n "$cell" ] && runs+=("$RESULTS_DIR/$cell")
  done 3<<< "$CELLS"
  for style in explainer clinician; do
    echo "== (e) LLM report, $style style"
    "$PY" "$REPO/src/llm_report.py" --run "${runs[@]}" --style "$style"
  done
}

usage () {
  awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"
}

if [ "$#" -eq 0 ]; then
  usage
  exit 1
fi

steps=("$@")
if [ "${steps[0]}" = "all" ]; then
  steps=(train assemble fusion saliency)
fi

for step in "${steps[@]}"; do
  case "$step" in
    train)    step_train ;;
    assemble) step_assemble ;;
    fusion)   step_fusion ;;
    saliency) step_saliency ;;
    llm)      step_llm ;;
    *) echo "unknown step: $step" >&2; usage; exit 1 ;;
  esac
done
