"""
explain.llm_narrative — turn a Ch7 run's saliency outputs into a
clinician-facing plain-English narrative via an LLM (OpenAI / ChatGPT).

Three stages, deliberately split so the LLM never decides what is
trustworthy:

    collect_evidence()    deterministic: reads the files report_saliency.py
                          and run_pipeline.py already wrote (ROI CSVs,
                          network scores, per-subject predictions) and
                          reduces them to a small JSON. Fusion is the
                          thesis's fixed mean (stacker as secondary).
    assess_reliability()  deterministic gate: bootstrap AUC CI per
                          modality, plus fusion checks (gain over fMRI,
                          class collapse, degenerate stacker). A modality
                          that fails the gate has
                          its attribution content REMOVED from the prompt,
                          so the LLM cannot narrate a map of noise.
    build_prompt() / call_openai()
                          the only LLM step — rephrases the gated evidence
                          for a clinician. Nothing it says is fed back.

Only aggregate, cohort-level numbers leave the machine: no subject IDs,
no images, no per-subject values.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from explain.atlas_glossary import describe_aal, describe_schaefer, network_name
from explain.network_attribution import schaefer_label_table

AAL_LABELS_CSV = Path(__file__).resolve().parent / 'labels' / 'aal_spm12.csv'

# Gate thresholds. Kept as module constants so the thesis can quote them.
N_BOOT            = 2000    # bootstrap resamples for the AUC 95% CI
CI_ALPHA          = 0.05
MIN_FUSION_GAIN   = 0.02    # fused AUC must beat fMRI-alone by this to count as a gain
SMALL_N_FUSED     = 50      # fusion on fewer matched subjects than this is noisy

DEFAULT_OPENAI_MODEL = 'gpt-5'


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _resolve(path_str: str | None, fallback: Path) -> Path | None:
    """Saliency summaries store absolute paths; fall back to the bundle copy."""
    if path_str and Path(path_str).is_file():
        return Path(path_str)
    return fallback if fallback.is_file() else None


def _auc_with_ci(y: np.ndarray, p: np.ndarray, seed: int = 0) -> dict:
    """Subject-level ROC AUC + percentile bootstrap 95% CI."""
    rng  = np.random.default_rng(seed)
    auc  = float(roc_auc_score(y, p))
    boot = []
    n = len(y)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        boot.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(boot, [100 * CI_ALPHA / 2, 100 * (1 - CI_ALPHA / 2)])
    return {'auc': round(auc, 3), 'ci95': [round(float(lo), 3), round(float(hi), 3)],
            'n_subjects': int(n), 'base_rate': round(float(y.mean()), 3)}


def _subject_predictions(pred_csv: Path) -> pd.DataFrame:
    """One row per subject; slice-level CSVs are averaged (Ch4 convention)."""
    df = pd.read_csv(pred_csv)
    if 'slice_idx' in df.columns:
        df = (df.groupby('subject_id')
                .agg(true=('true', 'first'), prob=('prob', 'mean'))
                .reset_index())
    return df


# ---------------------------------------------------------------------------
# ROI labelling
# ---------------------------------------------------------------------------

def _roi_labels(atlas: str, n_rois: int) -> pd.DataFrame | None:
    """Compact ROI index (0..n-1) → human-readable label table, or None.

    Compact order = sorted atlas label values (see
    report_saliency._voxel_to_roi_importance), so Schaefer index i is
    parcel i+1 and AAL index i is the i-th smallest AAL code.
    """
    if re.match(r'schaefer_\d+_(?:7|17)n', atlas or '', re.I):
        # The atlas volume is in 7-network label order; the 17-network
        # names are attached by parcel centroid (see
        # network_attribution.schaefer_label_table).
        try:
            t = schaefer_label_table(atlas)
        except ValueError:
            return None
        if len(t) != n_rois:
            return None
        n7, n17 = t['name_7'], t['name_17']
        return pd.DataFrame({
            'name':       n17.str.replace('17Networks_', '', regex=False),
            'hemisphere': n7.str.extract(r'_(LH|RH)_')[0].map({'LH': 'L', 'RH': 'R'}),
            'network_7':  n7.str.extract(r'7Networks_(?:LH|RH)_([A-Za-z]+)')[0],
            'network_17': n17.str.extract(r'17Networks_(?:LH|RH)_([A-Za-z]+)')[0],
            'mni_xyz':    t[['R', 'A', 'S']].astype(int).values.tolist(),
        })
    if (atlas or '').startswith('aal') and AAL_LABELS_CSV.is_file():
        aal = pd.read_csv(AAL_LABELS_CSV).sort_values('aal_code').reset_index(drop=True)
        if len(aal) != n_rois:
            return None
        return pd.DataFrame({'name': aal['region_name'], 'hemisphere': aal['hemisphere']})
    return None


# ---------------------------------------------------------------------------
# Evidence collection
# ---------------------------------------------------------------------------

def _fmri_evidence(sal_dir: Path, bundle: Path, top_k: int) -> dict | None:
    summ = _read_json(sal_dir / 'summary.json')
    if summ is None:
        return None
    ev: dict = {
        'model': f"{summ.get('connectivity', '?')} fMRI model",
        'atlas': summ.get('atlas_name'),
        'attribution_method': 'Integrated Gradients (mean |IG| across test subjects)',
        'n_subjects_explained': summ.get('n_subjects'),
    }
    pred = _resolve(summ.get('predictions_csv'), bundle / 'predictions_subject.csv')
    if pred is not None:
        df = _subject_predictions(pred)
        ev['performance'] = _auc_with_ci(df['true'].to_numpy(), df['prob'].to_numpy())

    method = (summ.get('methods') or ['integrated_gradients'])[0]
    roi_csv = sal_dir / f'roi_importance_{method}.csv'
    if roi_csv.is_file():
        scores = pd.read_csv(roi_csv).iloc[:, 1].to_numpy(dtype=float)
        labels = _roi_labels(ev['atlas'], len(scores))
        median = float(np.median(scores))
        order  = np.argsort(scores)[::-1][:top_k]
        rois = []
        for rank, i in enumerate(order, 1):
            row = {'rank': rank,
                   'relative_to_median_roi': round(float(scores[i] / median), 2)}
            if labels is not None:
                row.update(labels.iloc[i].dropna().to_dict())
                describe = describe_schaefer if 'network_7' in labels else describe_aal
                anatomy = describe(row['name'])
                if anatomy:
                    row['anatomy'] = anatomy
            else:
                row['name'] = f'ROI {i + 1} (unlabelled atlas)'
            rois.append(row)
        ev['roi_labels_available'] = labels is not None
        ev['top_rois'] = rois
        ev['roi_concentration'] = {
            'n_rois': int(len(scores)),
            f'top_{top_k}_share_of_total': round(float(scores[order].sum() / scores.sum()), 3),
            'uniform_share_would_be': round(top_k / len(scores), 3),
            'max_over_median': round(float(scores.max() / median), 2),
        }
        # Network scores recomputed here from the centroid-matched table
        # rather than read from summary.json (mean |IG| over member ROIs).
        if labels is not None and 'network_7' in labels:
            for scheme in ('7', '17'):
                s = pd.Series(scores).groupby(labels[f'network_{scheme}'].values).mean()
                ev[f'yeo_{scheme}n_networks'] = [
                    {'network': k, 'network_name': network_name(k),
                     'relative_to_network_mean': round(float(v / s.mean()), 2)}
                    for k, v in s.sort_values(ascending=False).items()
                ]
    return ev


def _smri_evidence(sal_dir: Path, bundle: Path) -> dict | None:
    summ = _read_json(sal_dir / 'summary.json')
    if summ is None:
        return None
    ev: dict = {
        'model': '2D slice CNN on structural MRI',
        'attribution_method': 'Grad-CAM on each subject\'s median slice, averaged',
        'n_subjects_explained': summ.get('n_subjects'),
        'view': Path(summ.get('slices_dir', '')).name or None,
    }
    pred = _resolve(summ.get('predictions_csv'), bundle / 'predictions_slice.csv')
    if pred is not None:
        df = _subject_predictions(pred)
        ev['performance'] = _auc_with_ci(df['true'].to_numpy(), df['prob'].to_numpy())

    return ev


def _smri_saliency_dir(run_dir: Path) -> tuple[Path, str] | None:
    """The run's structural Grad-CAM directory (<run>/smri/saliency), if present."""
    own = run_dir / 'smri' / 'saliency'
    if (own / 'summary.json').is_file():
        return own, 'own'
    return None


def _fusion_evidence(run_dir: Path) -> dict | None:
    """Fixed-mean fusion (primary) + LOO-LR stacker (secondary) on the
    matched subject set, recomputed from the run's saved predictions with
    the same code as recompute_fusion.py, so numbers match the thesis tables.
    """
    from recompute_fusion import loo_lr_probs, metrics_row, normalize_id
    s_csv = run_dir / 'smri' / 'predictions_slice.csv'
    f_csv = run_dir / 'fmri' / 'predictions_subject.csv'
    if not (s_csv.is_file() and f_csv.is_file()):
        return None
    s = _subject_predictions(s_csv)
    f = pd.read_csv(f_csv).groupby('subject_id').agg(true=('true', 'first'),
                                                       prob=('prob', 'first')).reset_index()
    for df in (s, f):
        df['sid'] = df['subject_id'].map(normalize_id)
    m = s.merge(f, on='sid', suffixes=('_s', '_f'))
    if (m['true_s'] != m['true_f']).any():
        raise ValueError(f'{run_dir}: y_true disagrees across arms for matched subjects')
    y, ps, pf = m['true_s'].astype(int).values, m['prob_s'].values, m['prob_f'].values

    def row(prob):
        r = metrics_row('', '', '', '', y, prob, len(y))
        return {k: r[k] for k in ('auc', 'accuracy', 'balanced_accuracy', 'recall',
                                  'pred_pos_rate', 'degenerate_flag')}

    X = np.column_stack([ps, pf])
    refit = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(X, y)
    w_s, w_f = (float(w) for w in refit.coef_[0])
    return {
        'n_subjects_with_both_modalities': int(len(y)),
        'base_rate': round(float(y.mean()), 3),
        'smri_alone':  row(ps),
        'fmri_alone':  row(pf),
        'primary_fixed_mean': {
            'method': 'equal-weight average of the sMRI and fMRI probabilities, threshold 0.5',
            **row(0.5 * (ps + pf)),
        },
        'secondary_lr_stacker': {
            'method': 'logistic regression on the two probabilities, leave-one-out',
            **row(loo_lr_probs(X, y)),
            'refit_weight_smri': round(w_s, 2),
            'refit_weight_fmri': round(w_f, 2),
        },
    }


def collect_evidence(run_dir: str | Path, top_k: int = 15) -> dict:
    """Read one results/ch7_v2/<cell>/ directory into a compact evidence dict."""
    run_dir = Path(run_dir)
    fmri = _fmri_evidence(run_dir / 'fmri' / 'saliency', run_dir / 'fmri', top_k)
    smri, found = None, _smri_saliency_dir(run_dir)
    if found is not None:
        smri = _smri_evidence(found[0], run_dir / 'smri')
        if smri is not None:
            smri['saliency_source'] = found[1]
    fmri_summ = _read_json(run_dir / 'fmri' / 'saliency' / 'summary.json') or {}
    run_summ  = _read_json(run_dir / 'summary.json') or {}
    dataset = run_summ.get('dataset') or fmri_summ.get('dataset')
    return {
        'run': run_dir.name,
        'dataset': dataset,
        'target': {'abide': 'autism diagnosis (ASD vs typically developing control)',
                   'gsp':   'high vs low trait anxiety (STAI-T)'}.get(dataset, dataset),
        'scope': 'cohort-level (averaged over the held-out test set), not an individual patient',
        'fmri': fmri,
        'smri': smri,
        'fusion': _fusion_evidence(run_dir),
    }


# ---------------------------------------------------------------------------
# Deterministic reliability gate
# ---------------------------------------------------------------------------

def assess_reliability(ev: dict) -> dict:
    """Decide, without the LLM, which attribution maps are worth explaining.

    A modality is `interpretable` only if its AUC 95% CI excludes 0.5.
    """
    out: dict = {}
    for mod in ('fmri', 'smri'):
        m = ev.get(mod)
        if m is None:
            continue
        reasons, ok = [], True
        perf = m.get('performance')
        if perf is None:
            ok = False
            reasons.append('no predictions found, so performance is unknown')
        elif perf['ci95'][0] <= 0.5:
            ok = False
            reasons.append(f"AUC {perf['auc']} (95% CI {perf['ci95'][0]}-{perf['ci95'][1]}) "
                           f"is not distinguishable from chance")
        else:
            reasons.append(f"AUC {perf['auc']} (95% CI {perf['ci95'][0]}-{perf['ci95'][1]}) "
                           f"is above chance")
        if mod == 'fmri' and m.get('top_rois') and not m.get('roi_labels_available'):
            reasons.append('atlas has no region-name table; ROIs are reported by index only')
        out[mod] = {'interpretable': ok, 'reasons': reasons}

    fus = ev.get('fusion')
    if fus:
        notes = []
        fm, fa, st = fus['primary_fixed_mean'], fus['fmri_alone'], fus['secondary_lr_stacker']
        gain = fm['auc'] - fa['auc']
        if gain < MIN_FUSION_GAIN:
            notes.append(f"fixed-mean fusion AUC {fm['auc']:.2f} vs fMRI alone {fa['auc']:.2f}: "
                         f"fusion does not improve discrimination")
        if 'smri' in out and not out['smri']['interpretable']:
            if abs(fm['recall'] - fa['recall']) >= 0.05:
                notes.append(f"the equal-weight average includes a chance-level sMRI arm, which shifts "
                             f"the decision threshold (recall {fa['recall']:.2f} -> {fm['recall']:.2f}) "
                             f"without adding discrimination")
            else:
                notes.append("the equal-weight average includes a chance-level sMRI arm, which adds "
                             "no discrimination and leaves the fMRI predictions largely unchanged")
        if fm['degenerate_flag'] or not (0.1 <= fm['pred_pos_rate'] <= 0.9):
            notes.append(f"fused predictions collapse towards one class "
                         f"(predicted-positive rate {fm['pred_pos_rate']:.2f})")
        if st['degenerate_flag']:
            notes.append(f"secondary stacker is degenerate (AUC {st['auc']:.2f}); this is an artefact "
                         f"of fitting a combiner with no signal, not anti-discrimination")
        else:
            notes.append(f"secondary stacker AUC {st['auc']:.2f}; refit weights sMRI "
                         f"{st['refit_weight_smri']:+.2f}, fMRI {st['refit_weight_fmri']:+.2f}")
        n = fus['n_subjects_with_both_modalities']
        if n < SMALL_N_FUSED:
            notes.append(f'fusion evaluated on only {n} subjects; estimates are unstable')
        out['fusion'] = {'notes': notes}
    return out


def redact_for_llm(ev: dict, rel: dict) -> dict:
    """Drop region/network content for modalities that failed the gate."""
    ev = json.loads(json.dumps(ev))
    for mod in ('fmri', 'smri'):
        if ev.get(mod) and not rel.get(mod, {}).get('interpretable', False):
            for k in ('top_rois', 'roi_concentration', 'yeo_7n_networks',
                      'yeo_17n_networks'):
                ev[mod].pop(k, None)
            ev[mod]['attribution_withheld'] = (
                'Attribution content withheld because the model failed the '
                'reliability gate; do not describe brain regions for this modality.')
    return ev


# ---------------------------------------------------------------------------
# Prompt + OpenAI call
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You explain machine-learning attribution maps from brain MRI to clinicians \
(psychiatrists, neurologists, radiologists) who are not ML specialists.

You receive a JSON object with two parts: `evidence` (model performance, the \
brain regions and Yeo networks the attribution method weighted most, fusion \
results) and `reliability` (a deterministic gate already applied by code).
Fusion: the primary result is the fixed mean (equal-weight average of the two \
probabilities); the logistic-regression stacker is a secondary comparison.

Rules:
- Use ONLY regions, networks and numbers present in `evidence`. Never add \
regions, effect sizes, or citations that are not given.
- If a modality's reliability says interpretable=false, say plainly that its \
map should not be interpreted and give the reason; do not speculate about \
what it might show.
- Attribution shows what the model relied on, not what causes the condition. \
Never imply causation, diagnosis, or treatment guidance.
- This is a cohort-level summary, not an individual patient report.
- Relative scores are unitless (1.0 = median ROI / mean network). Describe \
them in words ("about 1.6x the typical region"), and say when differences \
are small or attribution is spread broadly.
- Name regions using their `anatomy` field and networks using their \
`network_name` field, verbatim; these were written by code from the atlas \
documentation. Never expand atlas codes (e.g. pCun, StriCal, Cingulum_Post) \
yourself. If a region has no `anatomy` field, refer to it by its code only \
and do not guess its anatomy.
- You may say in one sentence where a highlighted network is commonly \
discussed in the clinical literature, phrased as context, not as confirmation.
- Plain language, no jargon without a gloss, under 400 words.

Output Markdown with exactly these sections:
## Bottom line
## How reliable is this model?
## What the functional MRI map emphasises
## What the structural MRI map emphasises
## How the two scans were combined
## Caveats
Omit a section only if that modality/fusion is absent from the evidence."""


CLINICIAN_PROMPT = """\
You are a consultant neuroradiologist writing a short research report on \
which brain regions a machine-learning model relied on when classifying a \
cohort from MRI. Write it as you would a clinical imaging report, for \
psychiatrists and neurologists.

You receive a JSON object describing one cohort: the clinical question, and \
for each MRI modality either the brain regions the model weighted most \
(ordered, with laterality, MNI coordinates and a relative weighting where \
1.0 = a typical region) or a note that the regional findings were withheld \
because the model did not perform better than chance.

Rules:
- Report ONLY the regions given. Name each region exactly as in its \
`anatomy` field. Do not add regions, effect sizes, numbers or citations.
- Describe the weighting in words (e.g. "most strongly", "prominently", \
"to a lesser degree"); do not quote the numeric weightings.
- You may group the listed regions by lobe or anatomical territory and \
note laterality (bilateral, left- or right-predominant) and midline structures.
- If a modality's findings were withheld, state in one sentence that no \
regional findings are reported because the model did not perform better than \
chance, and say nothing else about that modality.
- These are regions the model relied on across the cohort, not lesions or \
abnormalities, and not findings in any individual patient. Never imply \
causation, diagnosis or treatment.

Output Markdown with exactly these sections:
## Clinical question
## Technique
## Findings
### Functional MRI
### Structural MRI
## Impression
Technique: one or two sentences (cohort-level attribution maps averaged over \
the held-out test subjects). Impression: two to four numbered points. Under \
250 words in total."""


def redact_for_clinician(ev: dict, rel: dict) -> dict:
    """Region-only payload for the clinician-style report.

    Performance numbers, confidence intervals, network scores and fusion are
    left out; the gate decision is passed as a plain statement.
    """
    out = {'clinical_question': ev.get('target'), 'scope': ev.get('scope')}
    names = {'fmri': 'functional MRI (resting-state connectivity)',
             'smri': 'structural MRI (axial slices)'}
    for mod in ('fmri', 'smri'):
        if not ev.get(mod):
            continue
        m = {'modality': names[mod]}
        if rel.get(mod, {}).get('interpretable', False) and ev[mod].get('top_rois'):
            m['regions'] = [{'rank': r['rank'],
                             'anatomy': re.sub(r'\s*\([^()]*network[^()]*\)$', '',
                                               r.get('anatomy') or r.get('name')),
                             'hemisphere': r.get('hemisphere'),
                             'mni_xyz': r.get('mni_xyz'),
                             'relative_weighting': r['relative_to_median_roi']}
                            for r in ev[mod]['top_rois']]
        else:
            m['regional_findings'] = ('withheld: the model did not perform better '
                                      'than chance')
        out[mod] = m
    return out


def build_prompt(ev: dict, rel: dict, style: str = 'explainer') -> dict:
    if style == 'clinician':
        return {'system': CLINICIAN_PROMPT,
                'user': json.dumps(redact_for_clinician(ev, rel), indent=2)}
    payload = {'evidence': redact_for_llm(ev, rel), 'reliability': rel}
    return {'system': SYSTEM_PROMPT,
            'user': json.dumps(payload, indent=2)}


def call_openai(prompt: dict, model: str = DEFAULT_OPENAI_MODEL) -> dict:
    """Single Responses API call. Needs OPENAI_API_KEY in the environment."""
    from openai import OpenAI
    client = OpenAI()
    resp = client.responses.create(
        model=model,
        instructions=prompt['system'],
        input=prompt['user'],
    )
    usage = getattr(resp, 'usage', None)
    return {
        'text':  resp.output_text,
        'id':    resp.id,
        'model': resp.model,
        'usage': usage.model_dump() if usage is not None else None,
    }


def reliability_table_md(rel: dict) -> str:
    """Code-written gate summary, shown above the LLM text in the report."""
    lines = ['| Modality | Interpretable | Reason |', '|---|---|---|']
    for mod in ('fmri', 'smri'):
        if mod in rel:
            r = rel[mod]
            lines.append(f"| {mod.upper()} | {'yes' if r['interpretable'] else '**no**'} "
                         f"| {'; '.join(r['reasons'])} |")
    for note in rel.get('fusion', {}).get('notes', []):
        lines.append(f'| Fusion | — | {note} |')
    return '\n'.join(lines)
