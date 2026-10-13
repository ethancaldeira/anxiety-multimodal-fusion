#!/usr/bin/env python3
"""
recompute_fusion.py — late-fusion metrics for the Chapter 7 cells, computed
from the saved per-subject test predictions of the two branches.

For each dataset (GSP, ABIDE) and atlas (AAL, Schaefer-200) cell under
--results-dir, the script:

  1. reads the structural predictions (<cell>/smri/predictions_slice.csv,
     averaged over each subject's slices) and the functional predictions
     (<cell>/fmri/predictions_subject.csv);
  2. matches the two branches on a normalised subject ID and keeps only the
     test subjects present in both. Ground-truth labels must agree for every
     matched subject; otherwise the disagreements are listed and no output
     is written;
  3. evaluates, on the matched subjects,
       - each branch alone,
       - the fixed mean (primary combiner): the equal-weight average of the
         two probabilities, thresholded at 0.5,
       - the logistic-regression stacker (secondary combiner): L2 logistic
         regression (C = 1.0, max_iter = 1000) on the two probabilities,
         fitted with leave-one-out over the matched subjects so that each
         subject's fused probability comes from a model that did not see it;
  4. reports ROC AUC, accuracy, balanced accuracy, precision and recall,
     with diagnostics for degenerate predictors: the predicted-positive
     rate, the base rate, the SD / min / max of the probabilities, and
     degenerate_flag, which is set when the probabilities are near-constant
     (SD < 0.01) or when AUC < 0.45 together with SD < 0.05. AUC is reported
     as computed (never flipped).

The structural branch does not depend on the atlas. When its predictions and
matched subjects are identical across the two atlases of a dataset, it is
reported once, with atlas = 'all'.

Usage:
    python src/recompute_fusion.py
    python src/recompute_fusion.py --results-dir results/ch7_v2 \\
        --out results/ch7_v2/fusion_recompute_matched.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    roc_auc_score, accuracy_score, balanced_accuracy_score, recall_score,
    precision_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / 'results' / 'ch7_v2'
OUT_NAME = 'fusion_recompute_matched.csv'

DATASETS = ['gsp', 'abide']
ATLASES = ['aal', 'schaefer']
# Cell directory per (dataset, atlas). The ABIDE cells use the split shared
# by both branches (train / val / test = 503 / 116 / 155); the GSP cells use
# the GSP split shared by both branches (595 / 106 / 176).
CELL = {
    ('gsp', 'aal'): 'gsp_aal_voxel',
    ('gsp', 'schaefer'): 'gsp_schaefer_voxel',
    ('abide', 'aal'): 'abide_aal_voxel_sharedsplit',
    ('abide', 'schaefer'): 'abide_schaefer_voxel_sharedsplit',
}

COLUMNS = [
    'dataset', 'atlas', 'model', 'combiner', 'n_matched', 'auc', 'accuracy',
    'balanced_accuracy', 'precision', 'recall', 'pred_pos_rate', 'base_rate',
    'prob_std', 'prob_min', 'prob_max', 'degenerate_flag',
]

# --- diagnostic thresholds (printed at runtime) ---
NEAR_CONST_STD = 0.01   # prob std below this => near-constant predictor
ANTI_AUC = 0.45         # AUC below this ...
ANTI_STD = 0.05         # ... combined with prob std below this => degenerate
SEED = 42


def normalize_id(s) -> str:
    """Normalise a subject ID so both branches use the same key.

    - lowercase and strip whitespace
    - strip the BIDS 'sub-' prefix
    - for a site-prefixed numeric ID such as 'site_00NNNNN', keep the digits
    - if the result is all digits, drop leading zeros
    GSP IDs (SubNNNN_S1) have a non-numeric tail and keep their shared form.
    """
    s = str(s).strip().lower()
    s = re.sub(r'^sub-', '', s)                      # BIDS prefix
    m = re.fullmatch(r'[a-z]+[_-](\d+)', s)          # site_12345 -> 12345 (numeric ids only)
    if m:
        s = m.group(1)
    if s.isdigit():
        s = str(int(s))                              # strip leading zeros
    return s


def load_subject_predictions(csv_path: str | Path) -> pd.DataFrame:
    """One row per subject with columns sid (normalised ID), true, prob.

    Slice-level files (a slice_idx column; the structural branch) are
    averaged over each subject's slices. Subject-level files (the functional
    branch) are used as they are.
    """
    df = pd.read_csv(csv_path)
    df['sid'] = df['subject_id'].map(normalize_id)
    prob_agg = 'mean' if 'slice_idx' in df.columns else 'first'
    return (df.groupby('sid')
              .agg(true=('true', 'first'), prob=('prob', prob_agg))
              .reset_index())


def match_branches(structural: pd.DataFrame, functional: pd.DataFrame) -> pd.DataFrame:
    """Inner join of the two subject tables on sid.

    Returns columns sid, true_s, prob_s, true_f, prob_f.
    """
    s = structural.rename(columns={'prob': 'prob_s', 'true': 'true_s'})
    f = functional.rename(columns={'prob': 'prob_f', 'true': 'true_f'})
    return s.merge(f, on='sid', how='inner')


def loo_lr_probs(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Leave-one-out logistic-regression stacker over the matched subjects."""
    fused = np.zeros(len(y), dtype=float)
    for tr, te in LeaveOneOut().split(X):
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
        clf.fit(X[tr], y[tr])
        fused[te[0]] = clf.predict_proba(X[te])[0, 1]
    return fused


def stacker_weights(X: np.ndarray, y: np.ndarray) -> dict:
    """Stacker refitted once on all matched subjects, to report its weights."""
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED).fit(X, y)
    w_s, w_f = (float(w) for w in clf.coef_[0])
    return {'intercept': float(clf.intercept_[0]), 'w_smri': w_s, 'w_fmri': w_f}


def metrics_row(dataset, atlas, model, combiner, y, prob, n_matched) -> dict:
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= 0.5).astype(int)
    auc = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else float('nan')
    std = float(prob.std())
    near_const = std < NEAR_CONST_STD
    anti_learn = (not np.isnan(auc)) and (auc < ANTI_AUC) and (std < ANTI_STD)
    return {
        'dataset': dataset, 'atlas': atlas, 'model': model, 'combiner': combiner,
        'n_matched': int(n_matched),
        'auc': round(auc, 4) if not np.isnan(auc) else np.nan,
        'accuracy': round(accuracy_score(y, pred), 4),
        'balanced_accuracy': round(balanced_accuracy_score(y, pred), 4),
        'precision': round(precision_score(y, pred, pos_label=1, zero_division=0), 4),
        'recall': round(recall_score(y, pred, pos_label=1, zero_division=0), 4),
        'pred_pos_rate': round(float(pred.mean()), 4),
        'base_rate': round(float(y.mean()), 4),
        'prob_std': round(std, 4),
        'prob_min': round(float(prob.min()), 4),
        'prob_max': round(float(prob.max()), 4),
        'degenerate_flag': bool(near_const or anti_learn),
    }


def fused_probabilities(m: pd.DataFrame) -> dict[str, np.ndarray]:
    """Labels and per-subject probabilities for one matched table: each
    branch alone, the fixed mean and the leave-one-out stacker."""
    y = m['true_s'].astype(int).values
    ps, pf = m['prob_s'].values, m['prob_f'].values
    return {
        'y': y,
        'structural': ps,
        'functional': pf,
        'fixed_mean': 0.5 * (ps + pf),
        'loo_lr': loo_lr_probs(np.column_stack([ps, pf]), y),
    }


def fusion_rows(dataset: str, atlas: str, m: pd.DataFrame,
                include_structural: bool = True,
                probs: dict[str, np.ndarray] | None = None) -> list[dict]:
    """Metric rows for one matched table: (structural,) functional, fixed
    mean and leave-one-out stacker. `probs` is the output of
    fused_probabilities(m), computed here if not given."""
    p = probs if probs is not None else fused_probabilities(m)
    y, n = p['y'], len(m)
    rows = []
    if include_structural:
        rows.append(metrics_row(dataset, atlas, 'structural', 'none', y, p['structural'], n))
    rows.append(metrics_row(dataset, atlas, 'functional', 'none', y, p['functional'], n))
    rows.append(metrics_row(dataset, atlas, 'fusion', 'fixed_mean', y, p['fixed_mean'], n))
    rows.append(metrics_row(dataset, atlas, 'fusion', 'loo_lr', y, p['loo_lr'], n))
    return rows


def label_disagreements(m: pd.DataFrame) -> pd.DataFrame:
    """Matched subjects whose ground-truth label differs between branches."""
    return m[m['true_s'] != m['true_f']]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--results-dir', type=Path, default=DEFAULT_RESULTS_DIR,
                    help='Directory holding the four cell directories '
                         '(default: results/ch7_v2 in this repository).')
    ap.add_argument('--out', type=Path, default=None,
                    help=f'Output CSV (default: <results-dir>/{OUT_NAME}).')
    args = ap.parse_args(argv)
    results_dir = args.results_dir
    out_csv = args.out or results_dir / OUT_NAME

    def cell_csvs(dataset: str, atlas: str) -> tuple[Path, Path]:
        cell = results_dir / CELL[(dataset, atlas)]
        return (cell / 'smri' / 'predictions_slice.csv',
                cell / 'fmri' / 'predictions_subject.csv')

    print('=' * 80)
    print(f'Ch7 late fusion on matched test subjects — {results_dir}')
    print('=' * 80)
    print(f'degenerate_flag rule: prob_std < {NEAR_CONST_STD}'
          f'  OR  (auc < {ANTI_AUC} AND prob_std < {ANTI_STD})')
    print('AUC reported as computed (no flip, no max(auc, 1-auc)).')

    # ---------------- PASS 1: matching + integrity validation ----------------
    print('\n' + '-' * 80)
    print('PASS 1 — matching & integrity')
    print('-' * 80)
    disagreements = []          # (dataset, atlas, sid, true_s, true_f)
    struct_identical = {}       # dataset -> bool
    matched = {}                # (dataset, atlas) -> merged df

    for dataset in DATASETS:
        s_by_atlas = {a: load_subject_predictions(cell_csvs(dataset, a)[0]) for a in ATLASES}
        a0, a1 = ATLASES
        s0 = s_by_atlas[a0].sort_values('sid').reset_index(drop=True)
        s1 = s_by_atlas[a1].sort_values('sid').reset_index(drop=True)
        same = (list(s0['sid']) == list(s1['sid'])
                and np.allclose(s0['prob'].values, s1['prob'].values, atol=1e-6)
                and list(s0['true']) == list(s1['true']))
        struct_identical[dataset] = same
        print(f'\n[{dataset}] structural predictions identical across atlases: '
              f'{"PASS" if same else "FAIL"}')

        for atlas in ATLASES:
            s_csv, f_csv = cell_csvs(dataset, atlas)
            s = load_subject_predictions(s_csv)
            f = load_subject_predictions(f_csv)
            m = match_branches(s, f)
            matched[(dataset, atlas)] = m
            n_s_only = len(s) - len(m)
            n_f_only = len(f) - len(m)
            print(f'  [{dataset}/{atlas}]  n_structural_only={n_s_only}  '
                  f'n_functional_only={n_f_only}  n_matched={len(m)}')
            for _, r in label_disagreements(m).iterrows():
                disagreements.append((dataset, atlas, r['sid'], int(r['true_s']), int(r['true_f'])))

    if disagreements:
        print('\n' + '!' * 80)
        print(f'STOP — y_true disagreements across branches on {len(disagreements)} subject(s):')
        for ds, at, sid, ts, tf in disagreements:
            print(f'   [{ds}/{at}] subject {sid}: structural true={ts}  functional true={tf}')
        print('No CSV written. Resolve subject-id matching before recomputing.')
        print('!' * 80)
        return 1

    # ---------------- PASS 2: metrics on matched sets ----------------
    print('\n' + '-' * 80)
    print('PASS 2 — metrics (matched subjects only)')
    print('-' * 80)
    rows = []
    for dataset in DATASETS:
        # Structural branch reported once per dataset when it is identical
        # across atlases and the matched subject sets are the same.
        msets = {a: set(matched[(dataset, a)]['sid']) for a in ATLASES}
        matched_equal = msets[ATLASES[0]] == msets[ATLASES[1]]
        report_struct_once = struct_identical[dataset] and matched_equal

        for atlas in ATLASES:
            rows += fusion_rows(dataset, atlas, matched[(dataset, atlas)],
                                include_structural=not report_struct_once)

        if report_struct_once:
            m = matched[(dataset, ATLASES[0])]
            rows.append(metrics_row(dataset, 'all', 'structural', 'none',
                                    m['true_s'].astype(int).values, m['prob_s'].values, len(m)))

    df = (pd.DataFrame(rows)[COLUMNS]
            .sort_values(['dataset', 'atlas', 'model', 'combiner'])
            .reset_index(drop=True))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # ---------------- printed summary ----------------
    print('\nFull table:')
    with pd.option_context('display.width', 200, 'display.max_columns', None):
        print(df.to_string(index=False))

    flagged = df[df['degenerate_flag']]
    print('\n' + '-' * 80)
    print(f'Rows with degenerate_flag=True: {len(flagged)}')
    print('-' * 80)
    if len(flagged):
        with pd.option_context('display.width', 200, 'display.max_columns', None):
            print(flagged[['dataset', 'atlas', 'model', 'combiner', 'auc',
                           'prob_std', 'pred_pos_rate', 'base_rate']].to_string(index=False))
    else:
        print('(none)')

    print(f'\ny_true disagreements: {len(disagreements)}')
    print(f'CSV written: {out_csv}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
