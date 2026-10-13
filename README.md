# Multimodal sMRI + fMRI Fusion with Explainability

Score-level (late) fusion of a structural MRI slice classifier and a
resting-state fMRI connectivity classifier, with attribution maps for each
branch, on two cohorts: trait anxiety (STAI-T) in the GSP and autism in
ABIDE I. The two branches are trained independently, on the same subject
splits, in the Chapter 4 and Chapter 5 repositories. This repository
combines their per-subject test predictions, evaluates the fusion, and
produces the explainability outputs: Integrated Gradients for the fMRI
model, Grad-CAM for the sMRI model, Yeo network summaries, and an optional
LLM-written report.

This repository contains the code for **Chapter 7** of a thesis on anxiety
prediction from MRI:

- Ch4: structural MRI ([anxiety-smri-classification](https://github.com/ethancaldeira/anxiety-smri-classification));
  it also trains the Chapter 7 structural branches (GSP full cohort, and
  ABIDE ANTs cortical thickness in `abide/`)
- Ch5: rs-fMRI with standard atlases ([anxiety-fmri-classification](https://github.com/ethancaldeira/anxiety-fmri-classification));
  it also trains the Chapter 7 functional branches
- Ch6: rs-fMRI with custom parcellations; models trained in
  [anxiety-fmri-classification](https://github.com/ethancaldeira/anxiety-fmri-classification),
  group and individual parcellations built in
  [anxiety-fmri-parcellations](https://github.com/ethancaldeira/anxiety-fmri-parcellations)
- **Ch7 (this repo)**: multimodal sMRI + fMRI fusion and explainability

---

## Repository layout

```
anxiety-multimodal-fusion/
├── scripts/
│   └── reproduce_ch7.sh            # thesis reproduction, step by step
├── src/
│   ├── paths.py                    # locations of the upstream repositories
│   ├── recompute_fusion.py         # fusion metrics on the matched test subjects
│   ├── report_saliency.py          # Integrated Gradients / Grad-CAM reports
│   ├── run_pipeline.py             # orchestrator: train → fuse → saliency for one run
│   ├── llm_report.py               # optional LLM report
│   └── explain/
│       ├── methods.py              # Grad-CAM, Grad-CAM++, SmoothGrad, Integrated Gradients
│       ├── network_attribution.py  # Schaefer ROI scores → Yeo 7 / 17 networks
│       ├── reporting.py            # network bar chart
│       ├── llm_narrative.py        # evidence, reliability gate, prompts
│       ├── atlas_glossary.py       # plain-anatomy names for atlas codes
│       ├── labels/aal_spm12.csv    # AAL code → region name (SPM12 AAL)
│       └── README.md
├── requirements.txt
├── LICENSE
└── results/                        # not tracked; results/ch7_v2/
```

---

## Installation

Tested with Python 3.12 on Ubuntu with a single NVIDIA T4 GPU (16 GB).
The code reads models, predictions, features, atlases and slice images from
the two upstream repositories, which are expected as sibling directories:

```bash
git clone https://github.com/ethancaldeira/anxiety-smri-classification.git
git clone https://github.com/ethancaldeira/anxiety-fmri-classification.git
git clone https://github.com/ethancaldeira/anxiety-multimodal-fusion.git
cd anxiety-multimodal-fusion
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions used for the thesis results
(TensorFlow 2.20, Keras 3.12). Other locations are set through environment
variables:

| Variable | Use | Default |
|---|---|---|
| `ANXIETY_SMRI_REPO` | anxiety-smri-classification checkout | `../anxiety-smri-classification` |
| `ANXIETY_FMRI_REPO` | anxiety-fmri-classification checkout | `../anxiety-fmri-classification` |
| `SCHAEFER_LABELS_DIR` | CBIG Schaefer label tables | `$ANXIETY_FMRI_REPO/src/templates/atlases/labels` |
| `OPENAI_API_KEY` | OpenAI API key, for the optional LLM report only | none (dry run) |
| `OPENAI_MODEL` | OpenAI model for the LLM report | `gpt-5` |

From the upstream repositories the code reads the branch runs
(`best.keras`, `predictions_slice.csv`, `predictions_subject.csv`), the fMRI
feature cache (`$ANXIETY_FMRI_REPO/data/cache/`), the atlases and Schaefer
label tables (`$ANXIETY_FMRI_REPO/src/templates/atlases/`), and the
structural slice images (`$ANXIETY_SMRI_REPO/data/<DATASET>/slices/`).

---

## Data

| | GSP | ABIDE I |
|---|---|---|
| Target | STAI-T ≥ 36 (cohort median) | diagnosis: ASD = 1, control = 0 |
| Subjects | 877 | 774 |
| Structural input | CAT12 modulated grey-matter density, axial slices | PCP ANTs cortical-thickness maps, axial slices |
| Functional input | voxel connectivity fingerprints, C-PAC-preprocessed BOLD | voxel connectivity fingerprints, PCP C-PAC BOLD (`nofilt_noglobal`) |
| Split (train / val / test) | 595 / 106 / 176, the GSP split of anxiety-fmri-classification, shared by both branches | 503 / 116 / 155, the shared ABIDE split, used by both branches |

The Brain Genomics Superstruct Project Open Access release
(Holmes et al., 2015) is available from
[neuroinformatics.harvard.edu/gsp](http://neuroinformatics.harvard.edu/gsp/)
under the GSP Open Access Data Use Terms. ABIDE I is used through the
[Preprocessed Connectomes Project](http://preprocessed-connectomes-project.org/abide/)
release (Craddock et al., 2013). No imaging data, phenotypes or subject
lists are distributed in this repository; the inputs are prepared in the
upstream repositories as described in their READMEs (the GSP split is
generated by `scripts/make_gsp_inputs.py` in anxiety-fmri-classification, the
ABIDE split is `abide/data_split_ABIDE_seed42.json` in
anxiety-smri-classification and `data/abide/abide_split_seed42.json` in
anxiety-fmri-classification).

---

## Method

### Branches

Each branch is trained independently and outputs one probability per test
subject.

- **Structural (sMRI):** ImageNet ResNet50 → global average pooling →
  Dense(256, ReLU) → Dropout(0.5) → Dense(1, sigmoid), trained in two
  stages as in Chapter 4, on the 40 axial slices around each subject's
  median slice (224 × 224, replicated to three channels, ResNet50
  `preprocess_input`). The subject probability is the mean over slices.
  GSP: CAT12 grey-matter density, all 877 subjects on the GSP split
  (`results/ch7_gsp_full_preprocessed_axial_ResNet50` in
  anxiety-smri-classification). ABIDE: ANTs cortical thickness
  (`results/ch7_abide_ants_axial_ResNet50`, trained with `abide/`).
- **Functional (fMRI):** the voxel 3D CNN of Chapter 5 on
  31 × 37 × 31 × N_ROIs connectivity fingerprints, one model per atlas:
  AAL (116 regions) and Schaefer-200 (7-network order). GSP: the Chapter 5
  GSP voxel runs. ABIDE: the same model trained on the shared ABIDE split
  (`run_abide_sharedsplit.sh` in anxiety-fmri-classification).

The structural branch does not depend on the atlas, so the design is
2 targets × 2 atlases, with the sMRI result shared by the two atlas cells
of each target.

### Fusion

`src/recompute_fusion.py` matches the two branches on a normalised subject
ID and keeps the test subjects present in both (176 GSP, 155 ABIDE);
ground-truth labels must agree for every matched subject.

- **Fixed mean (primary):** p = (p_sMRI + p_fMRI) / 2, thresholded at 0.5.
  No parameters are fitted.
- **Logistic-regression stacker (secondary):**
  p = σ(w_s · p_sMRI + w_f · p_fMRI + b), L2 logistic regression with
  C = 1.0 and max_iter = 1000, thresholded at 0.5. Because the branches
  save predictions for the held-out test sets only, the stacker is fitted
  with leave-one-out over the matched test subjects: each subject's fused
  probability comes from a model fitted on the other n − 1. The modality
  weights are reported from one refit on all matched subjects.

Each branch alone and both combiners are evaluated on the same matched
subjects: ROC AUC, accuracy, balanced accuracy, precision and recall, plus
diagnostics for degenerate predictors (predicted-positive rate, base rate,
SD / min / max of the probabilities, and `degenerate_flag`, set when the
probability SD is below 0.01, or when AUC < 0.45 with SD < 0.05). AUC is
reported as computed.

### Explainability

- **fMRI, Integrated Gradients** (Sundararajan et al., 2017) on the voxel
  3D CNN input, all-zeros baseline, 50 steps, for every test subject. The
  absolute attribution is averaged over the atlas channels and over
  subjects, giving a voxel map on the fingerprint grid (glass brain and
  orthogonal stat map). The per-ROI score is the mean of that map over the
  supervoxels of each atlas region (atlas resampled to the grid by nearest
  neighbour); the scores are painted onto the atlas in MNI152 space and
  shown as a glass brain. The attributions are not normalised.
- **Yeo networks:** for Schaefer-200, the per-ROI scores are averaged
  within the Yeo et al. (2011) networks. The thesis reports the 7-network
  scores; 17-network scores are written alongside, with the 17-network
  names matched to the atlas parcels by centroid.
- **sMRI, Grad-CAM** (Selvaraju et al., 2017) on the ResNet50 feature map,
  computed on each test subject's median slice, min-max normalised per
  subject, upsampled to 224 × 224 and averaged over the test set, with an
  overlay for one sample test subject.

`report_saliency.py` also supports the ROI-ROI MLP of Chapter 5
(`--connectivity roi_roi`: per-edge attribution averaged per ROI, and the
Haufe et al. (2014) forward pattern); Chapter 7 uses the voxel 3D CNN only.

### Optional LLM report

`src/llm_report.py` writes a plain-language report of one cell's
attribution maps in three stages (`src/explain/llm_narrative.py`):

1. **Evidence (code):** subject-level AUC with a bootstrap 95% CI
   (2,000 resamples) per branch; the top 15 ROIs with region names (AAL
   label table; Schaefer 7- and 17-network names) and plain-anatomy names
   from `atlas_glossary.py`; Yeo network scores; and the fusion results
   (fixed mean primary, stacker secondary) computed with the functions of
   `recompute_fusion.py`.
2. **Reliability gate (code):** a branch is interpretable only if its AUC
   CI excludes 0.5. For a branch that fails, the region and network content
   is removed from the prompt. Fusion notes flag no gain over fMRI alone
   (< 0.02 AUC), class collapse, a degenerate stacker, and fewer than 50
   matched subjects.
3. **Report (LLM):** one OpenAI Responses API call constrained to the
   given evidence. `--style explainer` (default) writes a plain-language
   summary with performance and fusion to `<cell>/llm/`; `--style clinician`
   writes a region-only report in the style of an imaging report to
   `<cell>/llm_clinician/`.

Only cohort-level aggregates are sent: no subject IDs, images or
per-subject values. The API key is read from `OPENAI_API_KEY`; without it,
or with `--dry-run`, the evidence, gate decision and exact prompt are
written and no call is made. The model is `--model`, else `$OPENAI_MODEL`,
else `gpt-5`. `prompt.json` records the exact prompt, and `report.md`
places the code-written gate table above the LLM text. The thesis shows an
excerpt of the clinician-style report for the ABIDE Schaefer-200 cell.

---

## Reproducing the thesis results

`scripts/reproduce_ch7.sh` runs the Chapter 7 steps in order; each step can
be run on its own:

```bash
scripts/reproduce_ch7.sh train      # (a) train the four branch models upstream
scripts/reproduce_ch7.sh assemble   # (b) copy test predictions into results/ch7_v2/<cell>/
scripts/reproduce_ch7.sh fusion     # (c) results/ch7_v2/fusion_recompute_matched.csv
scripts/reproduce_ch7.sh saliency   # (d) Integrated Gradients and Grad-CAM per cell
scripts/reproduce_ch7.sh llm        # (e) optional LLM reports, both styles

scripts/reproduce_ch7.sh all        # (a)–(d)
```

The training step runs, in the upstream repositories:

| Branch | Repository | Command | Run directory |
|---|---|---|---|
| GSP sMRI | anxiety-smri-classification | `EXPERIMENT_TYPE=preprocessed VIEW_MODE=axial BASE_MODEL=ResNet50 USE_GSP_SUBSET=0 SPLIT_FILE=$ANXIETY_FMRI_REPO/data/gsp_split.json EXPERIMENT_NAME=ch7_gsp_full_preprocessed_axial_ResNet50 python src/train.py` | `results/ch7_gsp_full_preprocessed_axial_ResNet50` |
| ABIDE sMRI | anxiety-smri-classification | `EXPERIMENT_TYPE=raw VIEW_MODE=axial BASE_MODEL=ResNet50 EXPERIMENT_NAME=ch7_abide_ants_axial_ResNet50 python -m abide.train` | `results/ch7_abide_ants_axial_ResNet50` |
| GSP fMRI | anxiety-fmri-classification | `python run_experiments.py classify --dataset gsp --atlas <atlas> --connectivity voxel` | `results/ch5/gsp_classification_voxel_<atlas>` |
| ABIDE fMRI | anxiety-fmri-classification | `./run_abide_sharedsplit.sh` | `results/ch7/abide_voxel_{aal,schaefer}_sharedsplit` |

with `<atlas>` = `aal_roi_atlas`, `schaefer_200_7n_yeo`. The upstream data
(slices, participants tables, features and splits) must be prepared first,
as described in the upstream READMEs.

The fusion table can be recomputed from the assembled predictions at any
time:

```bash
python src/recompute_fusion.py --results-dir results/ch7_v2 \
    --out results/ch7_v2/fusion_recompute_matched.csv
```

### Other runs

`src/run_pipeline.py` runs one dataset × atlas configuration end to end:
it trains the branches upstream (GSP structural branch via
`src/train.py` of anxiety-smri-classification; functional branch via
`src/train.py` of anxiety-fmri-classification), copies the run bundles into
`results/ch7_v2/<run>/{smri,fmri}/`, writes the fusion results to
`<run>/fusion/` with the same functions as `recompute_fusion.py`, and runs
the saliency reports. A branch whose bundle already contains `best.keras`
is not retrained, so a run trained elsewhere (for example the ABIDE
structural branch) can be copied into place first. The training
subprocesses inherit the environment, so upstream settings apply to both
branches; for example, `USE_GSP_SUBSET=0
SPLIT_FILE=$ANXIETY_FMRI_REPO/data/gsp_split.json` trains the GSP
structural branch on the full cohort and the split used by the functional
branch, as in the thesis.

```bash
python src/run_pipeline.py --dataset gsp --modalities smri fmri \
    --atlas schaefer_200_7n_yeo --connectivity voxel \
    --smri-cell exp1_preprocessed_axial_ResNet50 --out <run> \
    [--force-retrain] [--llm-report --llm-style clinician]
```

`src/report_saliency.py` runs a single saliency report; see its `--help`
for the fMRI voxel, fMRI ROI-ROI and sMRI modes.

---

## Outputs

```
results/ch7_v2/
├── fusion_recompute_matched.csv        # fusion table (see Results)
├── {gsp,abide}_smri_saliency/          # Grad-CAM of each structural branch
└── <cell>/                             # gsp_{aal,schaefer}_voxel,
    │                                   # abide_{aal,schaefer}_voxel_sharedsplit
    ├── smri/
    │   ├── predictions_slice.csv       # subject_id, slice_idx, true, prob, pred
    │   └── saliency/                   # mean_gradcam.{npy,png}, sample_gradcam.png, summary.json
    ├── fmri/
    │   ├── predictions_subject.csv     # subject_id, true, prob, pred
    │   └── saliency/
    │       ├── mean_attribution_integrated_gradients.npy   # voxel map
    │       ├── brain_volume_integrated_gradients.nii.gz
    │       ├── glass_brain_integrated_gradients.png
    │       ├── stat_map_integrated_gradients.png
    │       ├── roi_importance_integrated_gradients.csv     # per-ROI score
    │       ├── brain_volume_roi_integrated_gradients.nii.gz
    │       ├── glass_brain_roi_integrated_gradients.png
    │       ├── network_scores_{7n,17n}_integrated_gradients.csv   # Schaefer only
    │       ├── network_bar_{7n,17n}_integrated_gradients.png      # Schaefer only
    │       └── summary.json
    ├── llm/                            # optional: evidence, reliability, prompt (.json), report.md
    └── llm_clinician/                  # optional: same, clinician style
```

`fusion_recompute_matched.csv` has one row per dataset × atlas × model ×
combiner with the columns `n_matched`, `auc`, `accuracy`,
`balanced_accuracy`, `precision`, `recall`, `pred_pos_rate`, `base_rate`,
`prob_std`, `prob_min`, `prob_max` and `degenerate_flag`; the structural
branch is listed once per dataset with `atlas = all`.
`run_pipeline.py` runs additionally write `<run>/fusion/`
(`fusion_metrics.csv` with the same columns, `fused_predictions.csv`,
`stacker_weights.json`) and `<run>/summary.json`.

---

## Results

From `results/ch7_v2/fusion_recompute_matched.csv`, on the matched test
subjects. The sMRI branch does not depend on the atlas.
† `degenerate_flag = True` (near-constant probabilities, or AUC < 0.45 with
a probability SD below 0.05).

**GSP, trait anxiety (176 test subjects)**

| Model | Atlas | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|
| sMRI † | — | 0.458 | 0.528 | 0.524 | 0.945 |
| fMRI | AAL | 0.540 | 0.534 | 0.539 | 0.681 |
| Fixed mean (primary) | AAL | 0.536 | 0.523 | 0.528 | 0.714 |
| LR stacker (secondary) † | AAL | 0.018 | 0.517 | 0.517 | 1.000 |
| fMRI | Schaefer-200 | 0.552 | 0.511 | 0.517 | 0.813 |
| Fixed mean (primary) | Schaefer-200 | 0.554 | 0.500 | 0.510 | 0.824 |
| LR stacker (secondary) † | Schaefer-200 | 0.016 | 0.517 | 0.517 | 1.000 |

**ABIDE I, autism (155 test subjects)**

| Model | Atlas | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|
| sMRI | — | 0.576 | 0.568 | 0.610 | 0.329 |
| fMRI | AAL | 0.739 | 0.690 | 0.671 | 0.724 |
| Fixed mean (primary) | AAL | 0.743 | 0.684 | 0.671 | 0.697 |
| LR stacker (secondary) | AAL | 0.716 | 0.671 | 0.662 | 0.671 |
| fMRI | Schaefer-200 | 0.735 | 0.671 | 0.662 | 0.671 |
| Fixed mean (primary) | Schaefer-200 | 0.738 | 0.677 | 0.671 | 0.671 |
| LR stacker (secondary) | Schaefer-200 | 0.720 | 0.664 | 0.662 | 0.645 |

---

## Data use and acknowledgements

GSP data are used under the GSP Open Access Data Use Terms. Data were
provided by the Brain Genomics Superstruct Project of Harvard University and
the Massachusetts General Hospital (Principal Investigators: Randy Buckner,
Joshua Roffman, and Jordan Smoller), with support from the Center for Brain
Science Neuroinformatics Research Group, the Athinoula A. Martinos Center for
Biomedical Imaging, and the Center for Human Genetic Research. 20 individual
investigators at Harvard and MGH generously contributed data to the overall
project.

ABIDE I data are used under the ABIDE data usage terms, via the
Preprocessed Connectomes Project.

- Holmes, A. J. et al. (2015). Brain Genomics Superstruct Project initial
  data release with structural, functional, and behavioral measures.
  *Scientific Data*, 2, 150031.
- Di Martino, A. et al. (2014). The autism brain imaging data exchange:
  towards a large-scale evaluation of the intrinsic brain architecture in
  autism. *Molecular Psychiatry*, 19(6), 659–667.
- Craddock, C. et al. (2013). The Neuro Bureau Preprocessing Initiative:
  open sharing of preprocessed neuroimaging data and derivatives.
  *Frontiers in Neuroinformatics*.

Methods and atlases:

- Selvaraju, R. R. et al. (2017). Grad-CAM: visual explanations from deep
  networks via gradient-based localization. *ICCV*.
- Sundararajan, M., Taly, A. and Yan, Q. (2017). Axiomatic attribution for
  deep networks. *ICML*.
- Haufe, S. et al. (2014). On the interpretation of weight vectors of
  linear models in multivariate neuroimaging. *NeuroImage*, 87, 96–110.
- Schaefer, A. et al. (2018). Local-global parcellation of the human
  cerebral cortex from intrinsic functional connectivity MRI. *Cerebral
  Cortex*, 28(9), 3095–3114.
- Yeo, B. T. T. et al. (2011). The organization of the human cerebral
  cortex estimated by intrinsic functional connectivity. *Journal of
  Neurophysiology*, 106(3), 1125–1165.
- Tzourio-Mazoyer, N. et al. (2002). Automated anatomical labeling of
  activations in SPM using a macroscopic anatomical parcellation of the MNI
  MRI single-subject brain. *NeuroImage*, 15(1), 273–289. The AAL label
  table in `src/explain/labels/aal_spm12.csv` follows the SPM12 AAL release.

## Citation

If you use this code, please cite the thesis: Ethan Caldeira (2026), PhD
thesis, University of Buckingham; Chapter 7, *Towards an Explainable
Multimodal Pipeline for Trait Anxiety Classification*.

## License

Code is released under the GNU General Public License v3.0 or later (see
[`LICENSE`](LICENSE)).
