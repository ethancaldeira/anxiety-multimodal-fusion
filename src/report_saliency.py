"""
report_saliency.py — attribution report for one trained model.

Three modes:

  fMRI voxel (--connectivity voxel), the 3D CNN on voxel connectivity
      fingerprints: Integrated Gradients (all-zeros baseline, 50 steps) per
      test subject, absolute value averaged over the atlas channels and over
      subjects, rendered on the (31, 37, 31) fingerprint grid as a glass
      brain and an orthogonal stat map. With --atlas-nifti, the voxel map is
      also averaged within each atlas region (per-ROI scores), painted back
      onto the atlas, and, for Schaefer atlases, aggregated into Yeo 7- and
      17-network scores.

  fMRI ROI-ROI (--connectivity roi_roi), the MLP on ROI-ROI connectivity:
      attribution per edge, averaged per ROI and projected onto the atlas,
      plus the Haufe forward pattern (covariance of each input feature with
      the predicted probability; Haufe et al. 2014).

  sMRI (--modality smri), the 2D slice CNN: Grad-CAM on each test subject's
      median slice (ResNet50 preprocessing, as in training), averaged over
      the test set, plus one sample overlay.

Test subjects are taken from the run's predictions CSV. fMRI features are
read from the anxiety-fmri-classification feature cache
(<ANXIETY_FMRI_REPO>/data/cache/<dataset>/<voxel|roi_roi>/<atlas>/<subject>.npy,
override with --cache-dir).

Usage:

  # fMRI voxel 3D CNN (the Chapter 7 functional branch)
  python src/report_saliency.py \\
      --model           <fmri_run>/best.keras \\
      --predictions-csv <fmri_run>/predictions_subject.csv \\
      --dataset         abide \\
      --modality        fmri \\
      --connectivity    voxel \\
      --atlas-name      schaefer_200_7n_yeo \\
      --atlas-nifti     <ANXIETY_FMRI_REPO>/src/templates/atlases/schaefer_200_7n_yeo.nii.gz \\
      --out             results/ch7_v2/<cell>/fmri/saliency

  # sMRI 2D slice CNN (the Chapter 7 structural branch)
  python src/report_saliency.py \\
      --model           <smri_run>/best.keras \\
      --predictions-csv <smri_run>/predictions_slice.csv \\
      --modality        smri \\
      --slices-dir      <ANXIETY_SMRI_REPO>/data/GSP/slices/preprocessed/axial \\
      --out             results/ch7_v2/<cell>/smri/saliency

  # fMRI ROI-ROI MLP, from the cache or from a pre-built test matrix
  python src/report_saliency.py \\
      --model <fmri_run>/best.keras \\
      --predictions-csv <fmri_run>/predictions_subject.csv --dataset gsp \\
      --atlas-nifti <atlas>.nii.gz --atlas-name cc200_roi_atlas \\
      --out <out_dir>
  python src/report_saliency.py \\
      --model <fmri_run>/best.keras --features <test_X.npy> \\
      --atlas-nifti <atlas>.nii.gz --atlas-name cc200_roi_atlas \\
      --out <out_dir>

Outputs in --out (<m> = attribution method):
  fMRI voxel
    mean_attribution_<m>.npy              mean |attribution| on the voxel grid
    brain_volume_<m>.nii.gz               the same map as a NIfTI
    glass_brain_<m>.png, stat_map_<m>.png
    roi_importance_<m>.csv                per-ROI score (with --atlas-nifti)
    brain_volume_roi_<m>.nii.gz           per-ROI scores painted onto the atlas
    glass_brain_roi_<m>.png
    network_scores_{7n,17n}_<m>.csv       Yeo network scores (Schaefer only)
    network_bar_{7n,17n}_<m>.png
  fMRI ROI-ROI
    attribution_<m>.npy                   per-subject attribution
    roi_importance_<m>.csv, brain_volume_<m>.nii.gz,
    glass_brain_<m>.png, stat_map_<m>.png
    *_haufe.*                             Haufe forward pattern, same outputs
    network_scores_*.csv, network_bar_*.png (Schaefer only)
  sMRI
    mean_gradcam.npy, mean_gradcam.png    mean Grad-CAM over test subjects
    sample_gradcam.png                    one test subject's overlay
  summary.json                            model, inputs and run metadata
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from glob import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import tensorflow as tf
from nilearn import plotting
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess

# Make the explain sub-package importable when the script is run from
# the repo root or via `python -m src.report_saliency`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from explain.methods import attribute as run_attribution
from explain.network_attribution import roi_to_network
from explain.reporting import network_bar
from paths import CH5_CACHE_DIR


# ---------------------------------------------------------------------------
# Build the test-feature matrix from a predictions_subject.csv
# ---------------------------------------------------------------------------

# Per-subject feature cache of anxiety-fmri-classification (src/paths.py).
# Override with --cache-dir.
DEFAULT_CACHE = str(CH5_CACHE_DIR)


def _load_roiroi_feature(npy_path: str) -> np.ndarray:
    """Load a (n_rois, n_rois) Fisher-z matrix and return its upper triangle."""
    a = np.load(npy_path)
    # Tolerate accidental leading singleton dims (the cache sometimes
    # stores arrays as (1, n_rois, n_rois)).
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f'Expected square matrix, got shape {a.shape} from {npy_path}')
    n_rois = a.shape[0]
    iu = np.triu_indices(n_rois, k=1)
    return a[iu]


def _build_test_features(
    predictions_csv: str,
    dataset: str,
    atlas_name: str,
    cache_base: str,
) -> tuple[np.ndarray, list[str]]:
    """Read predictions_subject.csv → assemble test matrix from cached .npy files.

    Returns (X, subject_ids) where X has shape (n_test, n_features) with
    n_features = n_rois * (n_rois - 1) / 2 (upper-triangle of the
    Fisher-z connectivity matrix — the form the MLP was trained on).
    """
    df = pd.read_csv(predictions_csv)
    if 'subject_id' not in df.columns:
        raise ValueError(f"{predictions_csv} missing 'subject_id' column "
                         f"(got {list(df.columns)})")
    subjects = df['subject_id'].astype(str).tolist()
    atlas_cache = os.path.join(cache_base, dataset, 'roi_roi', atlas_name)
    if not os.path.isdir(atlas_cache):
        raise FileNotFoundError(
            f"No cached .npy files for {dataset}/{atlas_name} at {atlas_cache}. "
            f"Run the Ch5 cell once locally to populate the cache, or pass "
            f"--features with a pre-built test matrix instead."
        )

    rows = []
    missing = []
    for sid in subjects:
        path = os.path.join(atlas_cache, f'{sid}.npy')
        if not os.path.isfile(path):
            missing.append(sid)
            continue
        rows.append(_load_roiroi_feature(path))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(subjects)} test subjects' .npy files "
            f"missing in {atlas_cache} (first 5: {missing[:5]})"
        )
    X = np.stack(rows, axis=0).astype(np.float32)
    print(f'           assembled test matrix shape={X.shape} '
          f'(from {len(subjects)} subjects in {atlas_cache})')
    return X, subjects


# ---------------------------------------------------------------------------
# fMRI VOXEL: per-subject 4D feature (X,Y,Z,n_rois) → 3D attribution
# ---------------------------------------------------------------------------

# The Ch5 voxel fingerprint grid is (31, 37, 31) at ~6mm. We build a
# Nifti1Image with this voxel size + the same MNI152 origin as the
# Ch5 standard atlases — nilearn will resample to its own template for
# display, so being close-but-not-perfect is fine here.
SUPERVOXEL_AFFINE = np.array([
    [-6., 0.,  0.,  90.],
    [ 0., 6.,  0., -126.],
    [ 0., 0.,  6.,  -72.],
    [ 0., 0.,  0.,   1.],
], dtype=np.float64)


def _voxel_cache_path(cache_base: str, dataset: str, atlas_name: str,
                      subject_id: str) -> str:
    return os.path.join(cache_base, dataset, 'voxel', atlas_name,
                        f'{subject_id}.npy')


def _voxel_to_roi_importance(mean_heat: np.ndarray,
                              atlas_nifti_path: str) -> np.ndarray:
    """Aggregate a (31, 37, 31) voxel attribution into per-ROI scores.

    The atlas NIfTI lives in MNI152 standard space (e.g. 91×109×91); we
    resample it to the supervoxel grid via nearest-neighbour interpolation
    so each supervoxel inherits one integer label. Then for each ROI,
    score = mean |attribution| over the supervoxels carrying that label.

    The returned vector is COMPACT (length = number of distinct positive
    labels in the resampled atlas, ordered by sorted label value). This
    matches the convention used by ``_project_to_brain_volume`` and by
    ``explain.network_attribution.roi_to_network`` — Schaefer's labels are
    1..N so the compact and 1-based-dense layouts coincide there, but AAL
    uses sparse codes (2001, 2002, ..., 9170), so the compact form is
    what the downstream helpers actually expect.
    """
    from nilearn.image import resample_img
    atlas_native = nib.load(atlas_nifti_path)
    atlas_resamp = resample_img(
        atlas_native,
        target_affine=SUPERVOXEL_AFFINE,
        target_shape=mean_heat.shape,
        interpolation='nearest',
        copy_header=True,
        force_resample=True,
    )
    atlas_grid = atlas_resamp.get_fdata().astype(int)
    abs_heat   = np.abs(mean_heat)
    labels     = sorted(int(v) for v in np.unique(atlas_grid) if v > 0)
    roi_imp    = np.zeros(len(labels), dtype=np.float32)
    for i, lbl in enumerate(labels):
        mask = atlas_grid == lbl
        if mask.any():
            roi_imp[i] = float(abs_heat[mask].mean())
    return roi_imp


def _voxel_attribution_for_subject(model: tf.keras.Model,
                                    x: np.ndarray,
                                    method: str) -> np.ndarray:
    """Per-subject attribution → 3D heatmap.

    For Grad-CAM the output is already a 3D spatial map (1, X, Y, Z).
    For IG the output is 4D (1, X, Y, Z, n_rois) — collapse to (X, Y, Z)
    via mean |IG| over the channel dimension.
    """
    attr = np.asarray(run_attribution(method, model, x))
    if attr.ndim == 4:                      # (1, X, Y, Z)
        return attr[0].astype(np.float32)
    if attr.ndim == 5:                      # (1, X, Y, Z, C)
        return np.abs(attr[0]).mean(axis=-1).astype(np.float32)
    raise ValueError(f'Unexpected voxel-attribution shape {attr.shape}')


def report_fmri_voxel(args) -> int:
    """Per-subject voxel attribution → mean over test set → glass brain."""
    os.makedirs(args.out, exist_ok=True)
    print(f'[report] model:     {args.model}')
    print(f'[report] mode:      fmri / voxel')
    print(f'[report] output:    {args.out}')

    model = tf.keras.models.load_model(args.model)
    print(f'           model loaded ({model.count_params():,} params)')

    # Test subjects from predictions_subject.csv (built by Ch5's train.py).
    if not args.predictions_csv:
        sys.exit('voxel mode requires --predictions-csv to enumerate test subjects')
    df = pd.read_csv(args.predictions_csv)
    if 'subject_id' not in df.columns:
        sys.exit(f'{args.predictions_csv} missing subject_id column')
    subjects = df['subject_id'].astype(str).tolist()
    print(f'           {len(subjects)} test subjects')

    cache = args.cache_dir or DEFAULT_CACHE
    # Voxel mode uses Integrated Gradients by default: the attribution is
    # taken at the model input, on the (31, 37, 31) fingerprint grid.
    methods_to_run = args.method or ['integrated_gradients']
    print(f'           methods: {methods_to_run}')

    # Process one subject at a time (each voxel feature is ~26 MB) and
    # accumulate the mean attribution per method incrementally.
    accum = {m: None for m in methods_to_run}
    n_done = 0
    for sid in subjects:
        path = _voxel_cache_path(cache, args.dataset, args.atlas_name, sid)
        if not os.path.isfile(path):
            print(f'  [skip] {sid}: missing {path}')
            continue
        x = np.load(path)
        while x.ndim > 4 and x.shape[0] == 1:
            x = x[0]
        x = x.astype(np.float32)[None, ...]   # (1, X, Y, Z, C)
        for m in methods_to_run:
            heat = _voxel_attribution_for_subject(model, x, m)
            accum[m] = heat if accum[m] is None else accum[m] + heat
        n_done += 1
        if n_done % 20 == 0:
            print(f'           processed {n_done}/{len(subjects)} subjects ...')
    if n_done == 0:
        sys.exit('No voxel features found for any test subject — check --cache-dir.')

    print(f'           accumulated over {n_done} subjects; rendering ...')
    network_outputs = {}    # populated below if --atlas-nifti given
    for m in methods_to_run:
        mean_heat = accum[m] / n_done
        np.save(os.path.join(args.out, f'mean_attribution_{m}.npy'), mean_heat)
        brain_img = nib.Nifti1Image(mean_heat.astype(np.float32), SUPERVOXEL_AFFINE)
        nib.save(brain_img, os.path.join(args.out, f'brain_volume_{m}.nii.gz'))
        _render_glass_brain(
            brain_img,
            os.path.join(args.out, f'glass_brain_{m}.png'),
            title=f'{os.path.basename(args.model).replace(".keras", "")} — '
                  f'mean |{m}| (n={n_done})',
        )
        _render_stat_map(
            brain_img,
            os.path.join(args.out, f'stat_map_{m}.png'),
            title=f'{os.path.basename(args.model).replace(".keras", "")} — {m}',
        )

        # --- Voxel → per-ROI → Yeo network -------------------------------
        # With --atlas-nifti, average the voxel map within each atlas region
        # (atlas resampled to the (31, 37, 31) grid, nearest neighbour),
        # paint the per-ROI scores back onto the atlas, and aggregate them
        # into Yeo networks for Schaefer atlases.
        if not args.atlas_nifti:
            continue
        try:
            roi_imp = _voxel_to_roi_importance(mean_heat, args.atlas_nifti)
        except Exception as e:
            print(f'  [warn] voxel→ROI aggregation failed for {m}: {e}')
            continue
        pd.Series(roi_imp, name=f'mean_abs_{m}').to_csv(
            os.path.join(args.out, f'roi_importance_{m}.csv'),
            index_label='roi_index_1based',
        )
        # Per-ROI scores painted onto the atlas at its native resolution.
        try:
            atlas_brain = _project_to_brain_volume(roi_imp, args.atlas_nifti)
            nib.save(atlas_brain,
                     os.path.join(args.out, f'brain_volume_roi_{m}.nii.gz'))
            _render_glass_brain(
                atlas_brain,
                os.path.join(args.out, f'glass_brain_roi_{m}.png'),
                title=f'{os.path.basename(args.model).replace(".keras", "")}'
                      f' — {m} (atlas-projected)',
            )
        except Exception as e:
            print(f'  [warn] atlas projection failed for {m}: {e}')
        if 'schaefer' in (args.atlas_name or '').lower():
            for scheme, label in (('7networks', '7n'), ('17networks', '17n')):
                try:
                    net_scores = roi_to_network(
                        roi_imp, args.atlas_name, scheme=scheme,
                    )
                    network_bar(
                        net_scores,
                        os.path.join(args.out, f'network_bar_{label}_{m}.png'),
                        title=f'{args.atlas_name} {label}-network — {m}',
                    )
                    net_scores.to_csv(
                        os.path.join(args.out,
                                     f'network_scores_{label}_{m}.csv')
                    )
                    network_outputs[f'{label}_{m}'] = net_scores.to_dict()
                except Exception as e:
                    print(f'  [warn] {label} network attribution '
                          f'failed for {m}: {e}')

    json.dump({
        'model':           os.path.abspath(args.model),
        'predictions_csv': os.path.abspath(args.predictions_csv),
        'dataset':         args.dataset,
        'atlas_name':      args.atlas_name,
        'modality':        'fmri',
        'connectivity':    'voxel',
        'n_subjects':      n_done,
        'voxel_shape':     [31, 37, 31],
        'methods':         methods_to_run,
        'atlas_nifti':     (os.path.abspath(args.atlas_nifti)
                            if args.atlas_nifti else None),
        'network_scores':  network_outputs,
    }, open(os.path.join(args.out, 'summary.json'), 'w'), indent=2)
    print(f'[report] wrote summary.json — done.')
    return 0


# ---------------------------------------------------------------------------
# sMRI 2D slice CNN — Grad-CAM overlay
# ---------------------------------------------------------------------------

# Slice PNGs of anxiety-smri-classification live under
# <ANXIETY_SMRI_REPO>/data/<DATASET>/slices/<EXPERIMENT_TYPE>/<VIEW>/<subject>/.

def _load_slice_png(path: str, target_size: tuple[int, int]) -> np.ndarray:
    """Load a PNG slice as the structural-branch loader does: greyscale,
    LANCZOS resize, 0-255 float, replicated to RGB."""
    img = Image.open(path).convert('L').resize(target_size, Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32)
    return np.stack([arr, arr, arr], axis=-1)


def report_smri(args) -> int:
    """Per-subject middle-slice Grad-CAM overlay → mean Grad-CAM figure."""
    os.makedirs(args.out, exist_ok=True)
    print(f'[report] model:     {args.model}')
    print(f'[report] mode:      smri / 2D slice')
    print(f'[report] output:    {args.out}')

    model = tf.keras.models.load_model(args.model)
    print(f'           model loaded ({model.count_params():,} params)')

    # Pull (subject_id, slice_idx) pairs from predictions_slice.csv,
    # then pick each subject's median slice for the saliency view.
    pred_csv = args.predictions_csv
    if not pred_csv:
        sys.exit('sMRI mode requires --predictions-csv (typically predictions_slice.csv)')
    df = pd.read_csv(pred_csv)
    needed_cols = {'subject_id', 'slice_idx'}
    if not needed_cols.issubset(df.columns):
        sys.exit(f'{pred_csv} missing required columns {needed_cols} '
                 f'(got {list(df.columns)})')
    # One row per subject — the median slice index.
    chosen = (df.groupby('subject_id')['slice_idx']
                .median().astype(int).reset_index()
                .rename(columns={'slice_idx': 'mid_slice'}))
    print(f'           {len(chosen)} test subjects (median-slice view each)')

    if not args.slices_dir:
        sys.exit('sMRI mode requires --slices-dir (e.g. '
                 '<ANXIETY_SMRI_REPO>/data/GSP/slices/preprocessed/axial/)')

    target_size = (224, 224)
    accum_heat  = None
    n_done      = 0
    sample_overlay = None
    for _, row in chosen.iterrows():
        sid = row['subject_id']
        mid = int(row['mid_slice'])
        subj_dir = os.path.join(args.slices_dir, sid)
        if not os.path.isdir(subj_dir):
            continue
        # Pick the slice whose index (the number after '_slice_') is closest to `mid`.
        candidates = sorted(glob(os.path.join(subj_dir, '*_slice_*.png')))
        if not candidates:
            continue
        chosen_path = min(
            candidates,
            key=lambda p: abs(int(re.search(r'_slice_(\d+)', os.path.basename(p)).group(1)) - mid),
        )
        img = _load_slice_png(chosen_path, target_size)
        # Same backbone preprocessing the structural model was trained with
        # (ResNet50, ImageNet).
        x = resnet50_preprocess(img[None, ...].astype(np.float32).copy())
        try:
            heat = np.asarray(run_attribution('gradcam', model, x))
        except Exception as e:
            print(f'  [warn] gradcam failed for {sid}: {e}')
            continue
        heat = heat[0]                            # (h, w)
        # Resize to slice resolution for the overlay.
        heat_full = np.asarray(
            Image.fromarray((heat * 255).astype(np.uint8))
                 .resize(target_size, Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        accum_heat = heat_full if accum_heat is None else accum_heat + heat_full
        if sample_overlay is None:
            sample_overlay = (img[..., 0] / img.max(), heat_full)
        n_done += 1
    if n_done == 0:
        sys.exit('No sMRI slices loaded — check --slices-dir.')

    mean_heat = accum_heat / n_done
    np.save(os.path.join(args.out, 'mean_gradcam.npy'), mean_heat)

    # Mean Grad-CAM figure.
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.zeros(target_size), cmap='gray')
    ax.imshow(mean_heat, cmap='jet', alpha=0.7)
    ax.set_title(f'sMRI mean Grad-CAM (n={n_done})')
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'mean_gradcam.png'),
                dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  [plot] {os.path.join(args.out, "mean_gradcam.png")}')

    # Sample overlay (the first test subject processed).
    if sample_overlay is not None:
        img, heat_full = sample_overlay
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(img, cmap='gray')
        ax.imshow(heat_full, cmap='jet', alpha=0.55)
        ax.set_title('sample test subject — Grad-CAM')
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, 'sample_gradcam.png'),
                    dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(f'  [plot] {os.path.join(args.out, "sample_gradcam.png")}')

    json.dump({
        'model':           os.path.abspath(args.model),
        'predictions_csv': os.path.abspath(pred_csv),
        'modality':        'smri',
        'slices_dir':      os.path.abspath(args.slices_dir),
        'n_subjects':      n_done,
        'method':          'gradcam',
        'target_size':     list(target_size),
    }, open(os.path.join(args.out, 'summary.json'), 'w'), indent=2)
    print(f'[report] wrote summary.json — done.')
    return 0


# ---------------------------------------------------------------------------
# Haufe transform (Haufe et al. 2014, NeuroImage 87:96–110)
# ---------------------------------------------------------------------------
# IG (and any gradient-based attribution) gives a backward pattern: which
# inputs the model uses to discriminate. The Haufe transform gives the
# corresponding forward (activation) pattern: how each input co-varies with
# the model output.
#
# For a (possibly non-linear) classifier that produces probabilities
# `y_hat`, the per-feature forward pattern simplifies to:
#
#     A_forward[i] = Cov(X[:, i], y_hat)
#
# This is the per-feature covariance between the input and the model's
# predicted probability — exactly the activation pattern that the
# Haufe et al. derivation collapses to in the binary case. We then
# aggregate to per-ROI the same way IG is aggregated (mean over each
# ROI's incident edges) so the two variants are directly comparable.

def _haufe_forward_pattern(X: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Per-feature covariance between input X and predicted probability.

    X shape (n_subjects, n_features), probs shape (n_subjects,).
    Returns (n_features,) — the Haufe-equivalent forward pattern.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    pc = probs - probs.mean()
    return (Xc * pc[:, None]).mean(axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# fMRI ROI-ROI: map attribution → per-ROI importance → 3D brain volume
# ---------------------------------------------------------------------------

def _roi_importance_from_roiroi_attribution(attr: np.ndarray, n_rois: int) -> np.ndarray:
    """Collapse per-subject roi_roi attribution to a length-n_rois importance vector.

    attr can be either:
        shape (B, n_rois, n_rois)    full square matrix
        shape (B, n_edges)           flat upper-triangle
    where n_edges = n_rois * (n_rois - 1) / 2.

    Strategy:
        1. Mean over batch dimension → one mean attribution map.
        2. If 1D, unflatten back to symmetric n_rois × n_rois.
        3. Per-ROI importance = mean |attribution| over that row+column.
    """
    attr = np.abs(np.asarray(attr))
    mean_attr = attr.mean(axis=0)
    if mean_attr.ndim == 1:
        # flat upper-triangle of size n_rois*(n_rois-1)/2; reconstruct
        # the symmetric square.
        n_edges = mean_attr.shape[0]
        expected = n_rois * (n_rois - 1) // 2
        if n_edges != expected:
            raise ValueError(
                f"Attribution dim {n_edges} does not match upper-triangle of "
                f"n_rois={n_rois} (expected {expected})."
            )
        mat = np.zeros((n_rois, n_rois), dtype=np.float32)
        iu = np.triu_indices(n_rois, k=1)
        mat[iu] = mean_attr
        mat = mat + mat.T
        mean_attr = mat
    elif mean_attr.shape != (n_rois, n_rois):
        raise ValueError(
            f"Attribution shape {mean_attr.shape} does not match "
            f"(n_rois, n_rois)=({n_rois},{n_rois})."
        )
    # ROI importance = mean of absolute attribution over the ROI's edges.
    roi_imp = mean_attr.mean(axis=1)
    return roi_imp


def _project_to_brain_volume(roi_importance: np.ndarray,
                              atlas_nifti_path: str) -> nib.Nifti1Image:
    """Paint each ROI's importance value into its atlas mask, return a NIfTI."""
    atlas = nib.load(atlas_nifti_path)
    atlas_data = atlas.get_fdata()
    out = np.zeros_like(atlas_data, dtype=np.float32)
    labels = sorted(int(v) for v in np.unique(atlas_data) if v > 0)
    if len(labels) != len(roi_importance):
        raise ValueError(
            f"Atlas has {len(labels)} labelled ROIs but importance vector "
            f"has {len(roi_importance)} entries."
        )
    for i, lbl in enumerate(labels):
        out[atlas_data == lbl] = float(roi_importance[i])
    return nib.Nifti1Image(out, atlas.affine, atlas.header)


def _render_glass_brain(brain_img: nib.Nifti1Image, out_path: str, title: str) -> None:
    # One-sided 'Reds' colormap (data is mean |attribution|, strictly
    # non-negative). The default diverging RdBu_r would treat the range
    # as symmetric around 0 and paint the canvas dark blue.
    #
    # ``threshold`` masks out near-zero voxels: cortex-only atlases like
    # Schaefer leave a lot of the brain volume at 0 (white matter,
    # subcortex, ventricles, etc.) and ``Reds(0)`` is a pale cream, so
    # without a threshold those zero voxels render as a pale wash that
    # makes the figure look washed-out. ``vmax * 1e-3`` is well below any
    # real attribution value but well above floating-point noise.
    data   = brain_img.get_fdata()
    vmax   = float(np.max(np.abs(data))) if data.size else 0.0
    thresh = max(vmax * 1e-3, 1e-12)
    fig = plt.figure(figsize=(10, 4))
    plotting.plot_glass_brain(
        brain_img, figure=fig, title=title, colorbar=True, plot_abs=False,
        display_mode='lyrz', cmap='Reds', vmin=0, threshold=thresh,
    )
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  [plot] {out_path}')


def _render_stat_map(brain_img: nib.Nifti1Image, out_path: str, title: str) -> None:
    # Same rationale as _render_glass_brain — one-sided Reds, threshold
    # near-zero voxels so cortex-only atlases don't get a pale wash.
    data   = brain_img.get_fdata()
    vmax   = float(np.max(np.abs(data))) if data.size else 0.0
    thresh = max(vmax * 1e-3, 1e-12)
    plotting.plot_stat_map(
        brain_img, title=title, colorbar=True, display_mode='ortho',
        cmap='Reds', vmin=0, threshold=thresh, output_file=out_path,
    )
    print(f'  [plot] {out_path}')


# ---------------------------------------------------------------------------
# fMRI ROI-ROI report
# ---------------------------------------------------------------------------

def report_fmri_roi_roi(args) -> int:
    os.makedirs(args.out, exist_ok=True)

    print(f'[report] model:     {args.model}')
    print(f'[report] atlas:     {args.atlas_nifti} ({args.atlas_name})')
    print(f'[report] output:    {args.out}')

    model = tf.keras.models.load_model(args.model)
    print(f'           model loaded ({model.count_params():,} params)')

    if args.features:
        print(f'[report] features:  {args.features}')
        X = np.load(args.features)
        subjects = None
    else:
        print(f'[report] features:  building from {args.predictions_csv}')
        if not args.dataset:
            sys.exit('--predictions-csv mode requires --dataset {gsp,abide}')
        X, subjects = _build_test_features(
            args.predictions_csv, args.dataset, args.atlas_name,
            args.cache_dir or DEFAULT_CACHE,
        )
    print(f'           features shape={X.shape}')

    # Number of ROIs comes from the atlas NIfTI labels.
    atlas_data = nib.load(args.atlas_nifti).get_fdata()
    n_rois = int(np.unique(atlas_data[atlas_data > 0]).size)
    print(f'           atlas has {n_rois} ROIs')

    # Default attribution methods: Grad-CAM applies only to models with
    # conv layers; models without them (the ROI-ROI MLP) use Integrated
    # Gradients.
    has_conv = any(isinstance(L, (tf.keras.layers.Conv2D, tf.keras.layers.Conv3D))
                   for L in model.layers)
    if args.method:
        methods_to_run = args.method
    elif has_conv:
        methods_to_run = ['gradcam', 'integrated_gradients']
    else:
        methods_to_run = ['integrated_gradients']
    print(f'           model has conv layers: {has_conv}; '
          f'methods to run: {methods_to_run}')
    summary = {
        'model':           os.path.abspath(args.model),
        'features_source': (os.path.abspath(args.features) if args.features
                            else os.path.abspath(args.predictions_csv)),
        'features_shape':  list(X.shape),
        'atlas_nifti':     os.path.abspath(args.atlas_nifti),
        'atlas_name':      args.atlas_name,
        'n_rois':          n_rois,
        'methods':         methods_to_run + ['haufe'],
    }

    # Predicted probabilities on the same test set — needed for the
    # Haufe transform (covariance of input with predicted prob). We
    # call model.predict once per pipeline run, not once per method.
    model_probs = model.predict(X, verbose=0).flatten()

    # ---- Haufe forward pattern (independent of the attribution method)
    # The forward pattern depends only on (X, ŷ), so it is computed once,
    # with the same per-ROI aggregation and brain projection as the
    # gradient-based methods.
    print(f'\n[haufe] computing forward activation pattern ...')
    haufe_edge = _haufe_forward_pattern(X, model_probs)
    haufe_attr = haufe_edge[None, :]                      # add batch dim for the helper
    haufe_roi  = _roi_importance_from_roiroi_attribution(haufe_attr, n_rois)
    pd.Series(haufe_roi, name='haufe').to_csv(
        os.path.join(args.out, 'roi_importance_haufe.csv'),
        index_label='roi_index_1based',
    )
    haufe_brain = _project_to_brain_volume(haufe_roi, args.atlas_nifti)
    nib.save(haufe_brain, os.path.join(args.out, 'brain_volume_haufe.nii.gz'))
    _render_glass_brain(
        haufe_brain,
        os.path.join(args.out, 'glass_brain_haufe.png'),
        title=f'{os.path.basename(args.model).replace(".keras", "")} — '
              f'Haufe forward pattern',
    )
    _render_stat_map(
        haufe_brain,
        os.path.join(args.out, 'stat_map_haufe.png'),
        title=f'{os.path.basename(args.model).replace(".keras", "")} — Haufe',
    )
    if 'schaefer' in (args.atlas_name or '').lower():
        for scheme, label in (('7networks', '7n'), ('17networks', '17n')):
            try:
                net_h = roi_to_network(
                    haufe_roi, args.atlas_name, scheme=scheme,
                )
                network_bar(
                    net_h,
                    os.path.join(args.out, f'network_bar_{label}_haufe.png'),
                    title=f'{args.atlas_name} {label}-network — Haufe',
                )
                net_h.to_csv(os.path.join(args.out, f'network_scores_{label}_haufe.csv'))
            except Exception as e:
                print(f'           [warn] {label} Haufe network attribution failed: {e}')

    # ---- Per-method gradient-based attributions (Grad-CAM, IG, etc.)
    for m in methods_to_run:
        print(f'\n[{m}] computing attribution ...')
        attr = np.asarray(run_attribution(m, model, X))
        np.save(os.path.join(args.out, f'attribution_{m}.npy'), attr)

        # Collapse attribution → per-ROI importance vector.
        roi_imp = _roi_importance_from_roiroi_attribution(attr, n_rois)
        pd.Series(roi_imp, name=f'mean_abs_{m}').to_csv(
            os.path.join(args.out, f'roi_importance_{m}.csv'),
            index_label='roi_index_1based',
        )
        print(f'           roi_imp range=[{roi_imp.min():.4f}, {roi_imp.max():.4f}]')

        # Project to 3D brain volume and render the two views.
        brain_img = _project_to_brain_volume(roi_imp, args.atlas_nifti)
        nib.save(brain_img, os.path.join(args.out, f'brain_volume_{m}.nii.gz'))
        _render_glass_brain(
            brain_img,
            os.path.join(args.out, f'glass_brain_{m}.png'),
            title=f'{os.path.basename(args.model).replace(".keras", "")} — {m}',
        )
        _render_stat_map(
            brain_img,
            os.path.join(args.out, f'stat_map_{m}.png'),
            title=f'{os.path.basename(args.model).replace(".keras", "")} — {m}',
        )

        # Yeo network scores for Schaefer atlases: 7 and 17 networks from
        # the same per-ROI scores.
        if 'schaefer' in (args.atlas_name or '').lower():
            for scheme, label in (('7networks', '7n'), ('17networks', '17n')):
                try:
                    net_scores = roi_to_network(
                        roi_imp, args.atlas_name, scheme=scheme,
                    )
                    network_bar(
                        net_scores,
                        os.path.join(args.out, f'network_bar_{label}_{m}.png'),
                        title=f'{args.atlas_name} {label}-network — {m}',
                    )
                    net_scores.to_csv(
                        os.path.join(args.out, f'network_scores_{label}_{m}.csv')
                    )
                except Exception as e:
                    print(f'           [warn] {label} network attribution '
                          f'failed for {m}: {e}')

    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n[report] wrote summary.json — done.')
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--model',       required=True, help='Path to best.keras')
    # One of --features or --predictions-csv is required.
    feat = ap.add_mutually_exclusive_group(required=True)
    feat.add_argument('--features', help='Pre-built test_X.npy (shape (n_test, n_features))')
    feat.add_argument('--predictions-csv',
                      help="Run's predictions_subject.csv — script will assemble "
                           "the test matrix from the cached per-subject .npy files.")
    ap.add_argument('--dataset', choices=['gsp', 'abide'],
                    help='Required when --predictions-csv is used.')
    ap.add_argument('--cache-dir',
                    help=f'Per-subject .npy cache root (default: {DEFAULT_CACHE}).')
    # atlas-nifti is required for fMRI roi_roi only (we use it to project
    # ROI scores back to a 3D NIfTI). For voxel + sMRI it's optional.
    ap.add_argument('--atlas-nifti', help='Path to atlas .nii.gz '
                                          '(required for fMRI roi_roi mode).')
    ap.add_argument('--atlas-name',
                    help='Atlas name (e.g. aal_roi_atlas, schaefer_200_7n_yeo); '
                         'required for fMRI modes — used for the cache lookup '
                         'and the Yeo network aggregation.')
    ap.add_argument('--modality',    default='fmri', choices=['fmri', 'smri'])
    ap.add_argument('--connectivity', default='roi_roi',
                    choices=['roi_roi', 'voxel'],
                    help='fMRI only. Default = roi_roi (MLP). voxel = 3D CNN.')
    ap.add_argument('--slices-dir',
                    help='sMRI only. Slice PNG directory with one folder per '
                         'subject (e.g. <ANXIETY_SMRI_REPO>/data/GSP/slices/'
                         'preprocessed/axial/).')
    ap.add_argument('--method',      nargs='+',
                    choices=['gradcam', 'gradcam_plus_plus',
                             'smoothgrad', 'integrated_gradients'],
                    help='fMRI only. Attribution method(s). Default: '
                         'integrated_gradients for voxel mode and for models '
                         'without conv layers; gradcam + integrated_gradients '
                         'for conv models in roi_roi mode. sMRI mode always '
                         'uses gradcam.')
    ap.add_argument('--out',         required=True,
                    help='Output directory for figures + .npy + .csv')

    args = ap.parse_args()
    if args.modality == 'smri':
        sys.exit(report_smri(args))
    if args.connectivity == 'voxel':
        sys.exit(report_fmri_voxel(args))
    sys.exit(report_fmri_roi_roi(args))


if __name__ == '__main__':
    main()
