"""
run_pipeline.py — Chapter 7 orchestrator for one dataset × atlas run.

Steps:
  1. structural branch: trains the slice CNN in anxiety-smri-classification
     (src/train.py, GSP) and copies the run bundle to <run>/smri/;
  2. functional branch: trains the fMRI model in anxiety-fmri-classification
     (src/train.py) and copies the run bundle to <run>/fmri/;
  3. late fusion on the matched test subjects with the functions of
     recompute_fusion.py: each branch alone, the fixed mean (primary) and the
     leave-one-out logistic-regression stacker (secondary), written to
     <run>/fusion/;
  4. saliency for each branch (src/report_saliency.py): Integrated Gradients
     for the fMRI model, Grad-CAM for the sMRI model;
  5. optionally, the LLM report (src/llm_report.py).

A branch is not retrained when <run>/<smri|fmri>/best.keras already exists,
so a bundle trained elsewhere (for example the ABIDE structural arm, trained
in anxiety-smri-classification/abide) can be copied into place first.
`--force-retrain` retrains regardless. Source bundles in the upstream
repositories are copied, never moved or deleted. The training subprocesses
inherit the environment, so upstream settings (e.g. USE_GSP_SUBSET=0 and
SPLIT_FILE for the GSP structural branch) can be set on the command line.

The upstream repositories are located through ANXIETY_SMRI_REPO and
ANXIETY_FMRI_REPO (see src/paths.py). Runs are written to
<results-dir>/<run>/ (default results/ch7_v2).

Usage:
    python src/run_pipeline.py \\
        --dataset gsp \\
        --modalities smri fmri \\
        --atlas aal_roi_atlas \\
        --connectivity voxel \\
        --smri-cell exp1_preprocessed_axial_ResNet50 \\
        [--out <run_name>] [--force-retrain] [--llm-report]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import CH4_REPO, CH5_ATLAS_DIR, CH5_REPO, DEFAULT_RESULTS_DIR, REPO_ROOT  # noqa: E402
from recompute_fusion import (COLUMNS, fused_probabilities, fusion_rows,  # noqa: E402
                              label_disagreements, load_subject_predictions,
                              match_branches, stacker_weights)


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def _subprocess_train(label: str, argv: list[str], cwd: Path,
                       env: dict[str, str]) -> None:
    """Stream a training subprocess; raise on non-zero exit."""
    t0 = time.time()
    print(f'\n[{label}] launching: {" ".join(argv)}', flush=True)
    print(f'[{label}] cwd: {cwd}', flush=True)
    proc = subprocess.run(argv, cwd=str(cwd), env=env)
    dt = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f'[{label}] subprocess returned rc={proc.returncode} '
                           f'after {dt/60:.1f} min')
    print(f'[{label}] done in {dt/60:.1f} min', flush=True)


def _copy_bundle(src: Path, dst: Path) -> None:
    """Copy a results bundle into this run's directory (replacing `dst`).

    The source bundle is left in place.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f'           copied bundle: {src} → {dst}')


# ---------------------------------------------------------------------------
# sMRI: structural branch (anxiety-smri-classification)
# ---------------------------------------------------------------------------

def train_smri(dataset: str, smri_cell: str, out_dir: Path,
                force: bool) -> Path:
    """Train the structural branch for one cell and copy its bundle."""
    bundle_dst = out_dir / 'smri'
    if not force and (bundle_dst / 'best.keras').is_file():
        print(f'[smri] skip-if-exists: {bundle_dst}')
        return bundle_dst

    # Cell name convention: exp<N>_<EXPERIMENT_TYPE>_<VIEW_MODE>_<BASE_MODEL>
    parts = smri_cell.split('_')
    if len(parts) < 4 or not parts[0].startswith('exp'):
        raise ValueError(
            f'Unrecognised --smri-cell "{smri_cell}". Expected something like '
            f'exp1_preprocessed_axial_ResNet50.'
        )
    _, etype, view, base_model = parts[0], parts[1], parts[2], '_'.join(parts[3:])

    if dataset == 'gsp':
        train_script = 'src/train.py'
        train_argv   = [sys.executable, train_script]
        cwd          = CH4_REPO
        experiment_name = f'_ch7_pipeline_{smri_cell}'
        env = {
            **os.environ,
            'EXPERIMENT_TYPE': etype,
            'VIEW_MODE':       view,
            'BASE_MODEL':      base_model,
            'EXPERIMENT_NAME': experiment_name,
        }
        # anxiety-smri-classification writes to <repo>/results/<EXPERIMENT_NAME>/
        src_bundle = CH4_REPO / 'results' / experiment_name
    elif dataset == 'abide':
        raise NotImplementedError(
            'The ABIDE structural branch is trained in '
            'anxiety-smri-classification/abide (see its README). Copy the run '
            f'directory to {bundle_dst} and this step is skipped.'
        )
    else:
        raise ValueError(f'Unknown dataset {dataset}')

    _subprocess_train('smri', train_argv, cwd, env)

    if not (src_bundle / 'best.keras').is_file():
        raise FileNotFoundError(
            f'sMRI training reported success but no best.keras at {src_bundle}'
        )
    _copy_bundle(src_bundle, bundle_dst)
    return bundle_dst


# ---------------------------------------------------------------------------
# fMRI: functional branch (anxiety-fmri-classification)
# ---------------------------------------------------------------------------

def train_fmri(dataset: str, atlas: str, connectivity: str,
                out_dir: Path, force: bool) -> Path:
    """Train the functional branch for one cell and copy its bundle."""
    bundle_dst = out_dir / 'fmri'
    if not force and (bundle_dst / 'best.keras').is_file():
        print(f'[fmri] skip-if-exists: {bundle_dst}')
        return bundle_dst

    chapter         = '7_pipeline'
    experiment_name = f'{dataset}_classification_{connectivity}_{atlas}'
    env = {
        **os.environ,
        'CHAPTER':           chapter,
        'DATASET':           dataset,
        'TASK':              'classification',
        'CONNECTIVITY_TYPE': connectivity,
        'ATLAS':             atlas,
        'EXPERIMENT_NAME':   experiment_name,
        'KEEP_CACHE':        '1',
    }
    # anxiety-fmri-classification writes to <repo>/results/ch7_pipeline/<EXPERIMENT_NAME>/
    src_bundle = CH5_REPO / 'results' / f'ch{chapter}' / experiment_name

    _subprocess_train(
        'fmri',
        [sys.executable, 'src/train.py'],
        CH5_REPO, env,
    )

    if not (src_bundle / 'best.keras').is_file():
        raise FileNotFoundError(
            f'fMRI training reported success but no best.keras at {src_bundle}'
        )
    _copy_bundle(src_bundle, bundle_dst)
    return bundle_dst


# ---------------------------------------------------------------------------
# Late fusion (same functions as recompute_fusion.py)
# ---------------------------------------------------------------------------

def _bundle_predictions(bundle: Path, preferred: str) -> Path:
    """Predictions CSV of a bundle: `preferred` if present, else the other
    level (predictions_slice.csv / predictions_subject.csv)."""
    names = [preferred] + [n for n in ('predictions_slice.csv',
                                       'predictions_subject.csv') if n != preferred]
    for name in names:
        if (bundle / name).is_file():
            return bundle / name
    raise FileNotFoundError(
        f'Neither predictions_subject.csv nor predictions_slice.csv in {bundle}'
    )


def late_fusion(smri_bundle: Path | None,
                 fmri_bundle: Path | None,
                 out_dir: Path,
                 dataset: str,
                 atlas: str) -> dict | None:
    """Late fusion of the two branches on their matched test subjects.

    Subjects are matched on normalised ID (recompute_fusion.normalize_id);
    structural slice predictions are averaged per subject. On the matched
    subjects, each branch alone, the fixed mean (primary combiner) and the
    leave-one-out logistic-regression stacker (secondary combiner) are
    evaluated with recompute_fusion.metrics_row, so the numbers agree with
    fusion_recompute_matched.csv.

    Writes to <out_dir>/fusion/:
        fusion_metrics.csv       one row per model / combiner, same columns
                                 as fusion_recompute_matched.csv
        fused_predictions.csv    subject_id, true, prob_smri, prob_fmri,
                                 prob_fixed_mean, prob_loo_lr
        stacker_weights.json     stacker refitted on all matched subjects
                                 (intercept, w_smri, w_fmri)
    Returns a summary dict for summary.json, or None if a branch is missing.
    """
    if not (smri_bundle and fmri_bundle):
        print('[fusion] only one modality present — skipping fusion')
        return None

    dst = out_dir / 'fusion'
    dst.mkdir(parents=True, exist_ok=True)

    s = load_subject_predictions(_bundle_predictions(smri_bundle, 'predictions_slice.csv'))
    f = load_subject_predictions(_bundle_predictions(fmri_bundle, 'predictions_subject.csv'))
    m = match_branches(s, f)
    if m.empty:
        raise RuntimeError(
            f'No subject overlap between sMRI ({len(s)}) and fMRI ({len(f)}) '
            f'predictions; the two branches need the same test split.'
        )
    bad = label_disagreements(m)
    if len(bad):
        raise RuntimeError(
            f'{len(bad)} matched subject(s) have different labels in the two '
            f'branches (first: {list(bad["sid"][:5])}).'
        )

    p = fused_probabilities(m)
    rows = fusion_rows(dataset, atlas, m, include_structural=True, probs=p)
    pd.DataFrame(rows)[COLUMNS].to_csv(dst / 'fusion_metrics.csv', index=False)
    pd.DataFrame({
        'subject_id':      m['sid'],
        'true':            p['y'],
        'prob_smri':       p['structural'],
        'prob_fmri':       p['functional'],
        'prob_fixed_mean': p['fixed_mean'],
        'prob_loo_lr':     p['loo_lr'],
    }).to_csv(dst / 'fused_predictions.csv', index=False)
    weights = stacker_weights(np.column_stack([p['structural'], p['functional']]), p['y'])
    (dst / 'stacker_weights.json').write_text(json.dumps(weights, indent=2))

    keep = ('auc', 'accuracy', 'balanced_accuracy', 'precision', 'recall',
            'pred_pos_rate', 'base_rate', 'degenerate_flag')
    summary = {'n_smri': int(len(s)), 'n_fmri': int(len(f)), 'n_matched': int(len(m))}
    for r in rows:
        name = r['combiner'] if r['model'] == 'fusion' else r['model']
        summary[name] = {k: r[k] for k in keep}
    summary['stacker_weights'] = weights

    print(f'[fusion] n_matched={len(m)}  AUC: sMRI {summary["structural"]["auc"]:.3f}, '
          f'fMRI {summary["functional"]["auc"]:.3f}, '
          f'fixed mean {summary["fixed_mean"]["auc"]:.3f} (primary), '
          f'LOO LR stacker {summary["loo_lr"]["auc"]:.3f}  → {dst}')
    return summary


# ---------------------------------------------------------------------------
# Saliency dispatchers
# ---------------------------------------------------------------------------

def _run_saliency(label: str, argv: list[str]) -> None:
    print(f'[saliency:{label}] {" ".join(argv)}', flush=True)
    proc = subprocess.run(argv)
    if proc.returncode != 0:
        print(f'[saliency:{label}] WARN: rc={proc.returncode} — continuing')


def run_saliency_fmri(bundle: Path, atlas: str, connectivity: str,
                       dataset: str) -> None:
    """Run report_saliency.py on an fMRI bundle (roi_roi or voxel)."""
    out_dir = bundle / 'saliency'
    if (out_dir / 'summary.json').is_file():
        print(f'[saliency:fmri] skip-if-exists: {out_dir}')
        return
    atlas_nifti = CH5_ATLAS_DIR / f'{atlas}.nii.gz'
    argv = [
        sys.executable, str(REPO_ROOT / 'src' / 'report_saliency.py'),
        '--model',           str(bundle / 'best.keras'),
        '--predictions-csv', str(bundle / 'predictions_subject.csv'),
        '--dataset',         dataset,
        '--atlas-name',      atlas,
        '--modality',        'fmri',
        '--connectivity',    connectivity,
        '--out',             str(out_dir),
    ]
    if connectivity == 'roi_roi':
        if not atlas_nifti.is_file():
            print(f'[saliency:fmri] skip — atlas NIfTI not found: {atlas_nifti}')
            return
        argv += ['--atlas-nifti', str(atlas_nifti)]
    elif connectivity == 'voxel':
        # The atlas NIfTI enables the per-ROI scores and, for Schaefer
        # atlases, the Yeo network scores; without it only the voxel-grid
        # figures are written.
        if atlas_nifti.is_file():
            argv += ['--atlas-nifti', str(atlas_nifti)]
        else:
            print(f'[saliency:fmri] voxel: atlas NIfTI not found at '
                  f'{atlas_nifti} — voxel-grid figures only, no ROI/network '
                  f'aggregation.')
    _run_saliency(f'fmri/{connectivity}', argv)


def run_saliency_smri(bundle: Path, dataset: str, smri_cell: str) -> None:
    """Run report_saliency.py (Grad-CAM) on a structural 2D slice CNN bundle."""
    out_dir = bundle / 'saliency'
    if (out_dir / 'summary.json').is_file():
        print(f'[saliency:smri] skip-if-exists: {out_dir}')
        return
    # Input type + view from the cell name.
    parts = smri_cell.split('_')
    etype, view = parts[1], parts[2]
    dataset_dir = {'gsp': 'GSP', 'abide': 'ABIDE'}.get(dataset, dataset.upper())
    slices_dir  = CH4_REPO / 'data' / dataset_dir / 'slices' / etype / view
    pred_csv    = bundle / 'predictions_slice.csv'
    if not pred_csv.is_file():
        pred_csv = bundle / 'predictions_subject.csv'
    if not slices_dir.is_dir():
        print(f'[saliency:smri] skip — slices dir not found: {slices_dir}')
        return
    argv = [
        sys.executable, str(REPO_ROOT / 'src' / 'report_saliency.py'),
        '--model',           str(bundle / 'best.keras'),
        '--predictions-csv', str(pred_csv),
        '--modality',        'smri',
        '--slices-dir',      str(slices_dir),
        '--out',             str(out_dir),
    ]
    _run_saliency('smri', argv)


def run_llm_report(out_dir: Path, model: str | None, style: str) -> None:
    """Run llm_report.py on the run dir (dry run when OPENAI_API_KEY is unset)."""
    argv = [sys.executable, str(REPO_ROOT / 'src' / 'llm_report.py'),
            '--run', str(out_dir), '--style', style]
    if model:
        argv += ['--model', model]
    print(f'[llm] {" ".join(argv)}', flush=True)
    proc = subprocess.run(argv)
    if proc.returncode != 0:
        print(f'[llm] WARN: rc={proc.returncode} — continuing')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--dataset',       choices=['gsp', 'abide'], default='gsp')
    ap.add_argument('--modalities',    nargs='+',
                    choices=['smri', 'fmri'], default=['smri', 'fmri'])
    ap.add_argument('--atlas',         default='cc200_roi_atlas',
                    help='fMRI atlas (an atlas name of anxiety-fmri-classification).')
    ap.add_argument('--connectivity',  choices=['voxel', 'roi_roi'],
                    default='roi_roi')
    ap.add_argument('--smri-cell',     default='exp1_raw_sagittal_ResNet50',
                    help='Structural cell: exp<N>_<TYPE>_<VIEW>_<BACKBONE>.')
    ap.add_argument('--results-dir',   type=Path, default=DEFAULT_RESULTS_DIR,
                    help='Root for run directories (default: results/ch7_v2).')
    ap.add_argument('--out',           default=None,
                    help='Run directory name under --results-dir. '
                         'Default: auto-named.')
    ap.add_argument('--force-retrain', action='store_true',
                    help='Retrain even if a bundle with best.keras exists.')
    ap.add_argument('--llm-report',    action='store_true',
                    help='After saliency, write the LLM report '
                         '(src/llm_report.py) for the run.')
    ap.add_argument('--llm-style',     choices=['explainer', 'clinician'],
                    default='explainer',
                    help='LLM report style: explainer (<run>/llm/) or '
                         'clinician (<run>/llm_clinician/).')
    ap.add_argument('--llm-model',     default=None,
                    help='OpenAI model for --llm-report (default: '
                         '$OPENAI_MODEL or llm_narrative.DEFAULT_OPENAI_MODEL).')
    args = ap.parse_args()

    run_name = args.out or (
        f'{args.dataset}_{args.atlas}_{args.connectivity}'
        if 'fmri' in args.modalities else
        f'{args.dataset}_smri_{args.smri_cell}'
    )
    out_dir = args.results_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[pipeline] run_name = {run_name}')
    print(f'[pipeline] out_dir  = {out_dir}')

    bundles: dict[str, Path | None] = {'smri': None, 'fmri': None}

    if 'smri' in args.modalities:
        bundles['smri'] = train_smri(
            args.dataset, args.smri_cell, out_dir, args.force_retrain,
        )
    if 'fmri' in args.modalities:
        bundles['fmri'] = train_fmri(
            args.dataset, args.atlas, args.connectivity,
            out_dir, args.force_retrain,
        )

    fusion = late_fusion(bundles['smri'], bundles['fmri'], out_dir,
                         args.dataset, args.atlas)

    if bundles['fmri'] is not None:
        run_saliency_fmri(bundles['fmri'], args.atlas,
                          args.connectivity, args.dataset)
    if bundles['smri'] is not None:
        run_saliency_smri(bundles['smri'], args.dataset, args.smri_cell)

    summary = {
        'run_name':       run_name,
        'dataset':        args.dataset,
        'modalities':     args.modalities,
        'atlas':          args.atlas if 'fmri' in args.modalities else None,
        'connectivity':   args.connectivity if 'fmri' in args.modalities else None,
        'smri_cell':      args.smri_cell if 'smri' in args.modalities else None,
        'bundles': {
            k: str(v) if v else None for k, v in bundles.items()
        },
        'fusion':         fusion,
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    if args.llm_report:
        run_llm_report(out_dir, args.llm_model, args.llm_style)
    print(f'\n[pipeline] DONE — see {out_dir}/summary.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
