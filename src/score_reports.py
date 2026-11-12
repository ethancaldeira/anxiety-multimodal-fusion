"""
score_reports.py — automatic scoring of the six paper reports against their own evidence.

Reads (never modifies) {model}_{atlas}.md, payload_{atlas}.json and prompt_used.txt
from results/ch7/paper_reports_2026-11/ (written by paper_reports.py).
Writes, in results/ch7/paper_reports_2026-11/scoring/:
    prompt_diff.txt                    prompt_used.txt vs the paper's Figure 4 text
    vocab_aal.txt, vocab_schaefer.txt  full atlas anatomy vocabularies used for out-of-evidence search
    matches.csv                        every evidence / out-of-evidence / group-term match with its span
    errors.csv                         rule-based errors with severity
    table4_accuracy.csv                region precision / recall / F1, Kendall tau-b, error counts
    reference_texts.txt                BERTScore references
    table5_bertscore_readability.csv   BERTScore (Bio_ClinicalBERT, roberta-large raw and rescaled) + readability

Usage (needs bert_score, textstat, torch, scipy):
    python src/score_reports.py                 # all tables
    python src/score_reports.py --no-bertscore  # skip the BERTScore / readability table

Rule definitions are documented next to each check below.
"""
from __future__ import annotations

import difflib
import json
import re
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau

REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / 'results/ch7/paper_reports_2026-11'
HERE = REPORTS / 'scoring'                     # outputs
sys.path.insert(0, str(REPO / 'src'))
from explain.atlas_glossary import describe_aal, describe_schaefer  # noqa: E402
from explain.network_attribution import schaefer_label_table      # noqa: E402

MODELS = ['claude-fable-5-2', 'gpt-6-astra', 'gpt-6.1-sol']
ATLASES = ['aal', 'schaefer']
EXPECTED_SECTIONS = ['## Clinical question', '## Technique', '## Findings',
                     '### Functional MRI', '### Structural MRI', '## Impression']

FIG4 = """You are a consultant neuroradiologist writing a short research report on which brain regions a machine-learning model relied on when classifying a cohort from MRI. Write it as you would a clinical imaging report, for psychiatrists and neurologists.
You receive a JSON object describing one cohort: the clinical question, and for each MRI modality either the brain regions the model weighted most (ordered, with laterality, MNI coordinates and a relative weighting where 1.0 = a typical region) or a note that the regional findings were withheld because the model did not perform better than chance.
Rules:
- Report ONLY the regions given. Name each region exactly as in its `anatomy` field. Do not add regions, effect sizes, numbers or citations.
- Describe the weighting in words (e.g. "most strongly", "prominently", "to a lesser degree"); do not quote the numeric weightings.
- You may group the listed regions by lobe or anatomical territory and note laterality (bilateral, left- or right-predominant) and midline structures.
- If a modality's findings were withheld, state in one sentence that no regional findings are reported because the model did not perform better than chance, and say nothing else about that modality.
- These are regions the model relied on across the cohort, not lesions or abnormalities, and not findings in any individual patient. Never imply causation, diagnosis or treatment.
Output Markdown with exactly these sections:
## Clinical question
## Technique
## Findings
### Functional MRI
### Structural MRI
## Impression
Technique: one or two sentences (cohort-level attribution maps averaged over the held-out test subjects). Impression: two to four numbered points. Under 250 words in total."""

# Group terms the prompt allows (lobe / territory / laterality groupings). Searched only
# in text not already consumed by a named region; longest first.
GROUP_TERMS = [
    'medial occipital visual cortex', 'medial occipital cortex', 'posteromedial parietal cortex',
    'posteromedial parietal', 'posteromedial regions', 'posteromedial contributions', 'posteromedial',
    'posterior midline contributions', 'posterior midline', 'posterior cortical regions',
    'visual territories', 'visual-region', 'visual cortex', 'cerebellar regions', 'cerebellum',
    'cerebellar', 'occipital', 'parietal', 'frontal', 'temporal', 'lobe',
    'precuneus / posterior cingulate cortex', 'precuneus', 'cingulate', 'orbitofrontal', 'limbic',
    'extrastriate', 'striate', 'midline',
]

# Weighting descriptors -> strength tier (1 strongest). Used for the top-region and ordering rules.
TIER_CUES = [
    (1, r'most strongly|strongest|greatest|chiefly'),
    (2, r'prominent(?:ly)?|followed by'),
    (3, r'moderately|additional|somewhat less|slightly less'),
    (4, r'lesser|less\b'),
]
CLAUSE_SPLIT = re.compile(r';|\bfollowed by\b|,?\s+with\s+|,\s+(?:and\s+)?(?=the\s)', re.I)
# Territory a laterality claim refers to when it modifies a group term rather than named regions:
# group-term regex -> regex selecting the evidence regions in that territory.
TERRITORY = [
    (r'cerebell', r'cerebell'),
    (r'occipital|visual|striate|calcarine', r'(?<!pre)cuneus|calcarine|lingual|striate|visual|occipital'),
    (r'posteromedial|posterior midline|precuneus|cingulate|parietal', r'precuneus|cingulate|parietal'),
    (r'orbitofrontal|frontal', r'frontal'),
    (r'limbic', r'amygdala|hippocamp|limbic'),
]
LATERALITY = re.compile(r'\b(bilateral(?:ly)?|(?:broadly\s+)?symmetric(?:al)?|'
                        r'(?:left|right)[- ](?:predominant|predominance|sided)|'
                        r'(?:left|right)\s+predominance|on the (?:left|right))\b', re.I)
PEAK_QUAL = re.compile(r'\b(peak|strongest|most strongly)\b', re.I)
TOPK_QUAL = re.compile(r'\bhighest[- ]weighted\b', re.I)
CAUSAL = re.compile(r'\b(caus\w*|diagnos\w*|abnormal\w*|lesions?|treat\w*|biomarker\w*)', re.I)
NEGATION = re.compile(r"\b(not|no|never|nor|without|cannot)\b|n't\b", re.I)
CITATION = re.compile(r'et al\.?|\[\d+(?:[,–-]\d+)*\]|\([A-Z][A-Za-z-]+,? (?:19|20)\d\d\)|\bdoi\b', re.I)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def name_pattern(name: str) -> re.Pattern:
    """Case-insensitive pattern tolerant to whitespace, spacing around '/', hyphen/dash variants."""
    n = re.sub(r'\s*([/\-‐‑‒–—])\s*', r'\1', re.sub(r'\s+', ' ', name.strip()))
    out = []
    for ch in n:
        if ch == ' ':
            out.append(r'\s+')
        elif ch == '/':
            out.append(r'\s*/\s*')
        elif ch in '-‐‑‒–—':
            out.append(r'\s*[-‐‑‒–—]\s*')
        else:
            out.append(re.escape(ch))
    return re.compile(r'(?<![\w])' + ''.join(out) + r'(?![\w])', re.I)


def sections(md: str) -> tuple[list[str], dict[str, tuple[int, int]]]:
    """Headings in order, and char spans of each heading's body (to next heading of same/higher level)."""
    heads = [(m.start(), m.end(), m.group(0).strip()) for m in re.finditer(r'(?m)^#{1,6} .*$', md)]
    spans = {}
    for i, (s, e, h) in enumerate(heads):
        level = len(h) - len(h.lstrip('#'))
        end = len(md)
        for s2, _, h2 in heads[i + 1:]:
            if len(h2) - len(h2.lstrip('#')) <= level:
                end = s2
                break
        spans[h] = (e, end)
    return [h for _, _, h in heads], spans


def strip_md(text: str, keep_headings: bool = False) -> str:
    lines = []
    for ln in text.splitlines():
        if re.match(r'^\s*#{1,6}\s', ln):
            if keep_headings:
                lines.append(re.sub(r'^\s*#{1,6}\s+', '', ln))
            continue
        ln = re.sub(r'^\s*[-*+]\s+', '', ln)
        ln = re.sub(r'^\s*\d+\.\s+', '', ln)
        ln = ln.replace('**', '').replace('__', '').replace('`', '')
        lines.append(ln)
    return re.sub(r'\n{2,}', '\n', '\n'.join(lines)).strip()


def sentence_spans(md: str, start: int, end: int) -> list[tuple[int, int]]:
    """Sentences inside md[start:end]; lines (bullets, list items) are always boundaries."""
    out = []
    for lm in re.finditer(r'[^\n]+', md[start:end]):
        ls = start + lm.start()
        line = lm.group(0)
        pos = 0
        for sm in re.finditer(r'(?<=[.!?])\s+(?=[A-Z0-9*])', line):
            out.append((ls + pos, ls + sm.start()))
            pos = sm.end()
        out.append((ls + pos, ls + len(line)))
    return [(s, e) for s, e in out if md[s:e].strip() and not md[s:e].lstrip().startswith('#')]


def clause_spans(md: str, s: int, e: int) -> list[tuple[int, int]]:
    cuts = [s] + [s + m.start() for m in CLAUSE_SPLIT.finditer(md[s:e])] + [e]
    return [(a, b) for a, b in zip(cuts, cuts[1:]) if b > a]


def tier_of(text: str) -> int | None:
    for tier, pat in TIER_CUES:
        if re.search(pat, text, re.I):
            return tier
    return None


def shares(regions: list[dict]) -> dict:
    tot = {'L': 0.0, 'R': 0.0, 'M': 0.0}
    for r in regions:
        side = {'L': 'L', 'R': 'R'}.get(r['hemisphere'], 'M')
        tot[side] += r['relative_weighting']
    s = sum(tot.values()) or 1.0
    return {k: round(v / s, 3) for k, v in tot.items()}


# ---------------------------------------------------------------------------
# Vocabulary and truth sets
# ---------------------------------------------------------------------------

def strip_network(anat: str) -> str:
    # same regex as explain.llm_narrative.redact_for_clinician
    return re.sub(r'\s*\([^()]*network[^()]*\)$', '', anat)


def build_vocab() -> dict[str, list[str]]:
    aal = pd.read_csv(REPO / 'src/explain/labels/aal_spm12.csv')
    v_aal = [describe_aal(n) for n in aal['region_name']]
    t = schaefer_label_table('schaefer_200_7n_yeo')
    v_sch = [strip_network(describe_schaefer(n.replace('17Networks_', ''))) for n in t['name_17']]
    assert len(v_aal) == 116 and len(v_sch) == 200 and None not in v_aal + v_sch
    return {'aal': sorted(set(v_aal)), 'schaefer': sorted(set(v_sch))}


def truth_set(atlas: str) -> tuple[list[dict], list[dict]]:
    payload = json.loads((REPORTS / f'payload_{atlas}.json').read_text())
    regions = payload['fmri']['regions']
    best: dict[str, dict] = {}
    for r in regions:
        if r['anatomy'] not in best or r['rank'] < best[r['anatomy']]['rank']:
            best[r['anatomy']] = r
    return sorted(best.values(), key=lambda r: r['rank']), regions


# ---------------------------------------------------------------------------
# Scoring one report
# ---------------------------------------------------------------------------

def score_report(model: str, atlas: str, vocab: dict) -> tuple[dict, list, list]:
    rid = f'{model}_{atlas}'
    md = (REPORTS / f'{rid}.md').read_text()
    heads, spans = sections(md)
    truth, all_regions = truth_set(atlas)
    truth_by_name = {r['anatomy'].lower(): r for r in truth}
    other_vocab = {n.lower(): src for src in ('aal', 'schaefer') for n in vocab[src]}

    f_s, f_e = spans.get('## Findings', (0, 0))
    i_s, i_e = spans.get('## Impression', (0, 0))
    search = [('Findings', f_s, f_e), ('Impression', i_s, i_e)]
    smri = spans.get('### Structural MRI', (0, 0))

    def where(pos: int) -> str:
        if smri[0] <= pos < smri[1]:
            return 'Findings/Structural MRI'
        for name, a, b in search:
            if a <= pos < b:
                return name if name == 'Impression' else 'Findings/Functional MRI'
        return '?'

    # --- region matching: evidence + whole vocabulary, longest first, consuming spans
    names = {n.lower(): n for n in [r['anatomy'] for r in truth] + vocab['aal'] + vocab['schaefer']}
    order = sorted(names, key=lambda n: -len(re.sub(r'\s+', ' ', n)))
    consumed: list[tuple[int, int]] = []
    matches = []

    def free(a, b):
        return all(b <= s or a >= e for s, e in consumed)

    for key in order:
        pat = name_pattern(names[key])
        for _, a, b in search:
            for m in pat.finditer(md, a, b):
                if free(m.start(), m.end()):
                    consumed.append((m.start(), m.end()))
                    is_ev = key in truth_by_name
                    matches.append({
                        'report': rid, 'model': model, 'atlas': atlas,
                        'kind': 'evidence' if is_ev else 'out_of_evidence',
                        'region': names[key],
                        'evidence_rank': truth_by_name[key]['rank'] if is_ev else None,
                        'vocab_source': atlas if is_ev else other_vocab.get(key),
                        'section': where(m.start()), 'start': m.start(), 'end': m.end(),
                        'matched_text': md[m.start():m.end()],
                    })
    for term in GROUP_TERMS:
        for _, a, b in search:
            for m in name_pattern(term).finditer(md, a, b):
                if free(m.start(), m.end()):
                    consumed.append((m.start(), m.end()))
                    matches.append({'report': rid, 'model': model, 'atlas': atlas, 'kind': 'group_term',
                                    'region': term, 'evidence_rank': None, 'vocab_source': None,
                                    'section': where(m.start()), 'start': m.start(), 'end': m.end(),
                                    'matched_text': md[m.start():m.end()]})
    matches.sort(key=lambda r: r['start'])
    ev_m = [m for m in matches if m['kind'] == 'evidence']
    oe_m = [m for m in matches if m['kind'] == 'out_of_evidence']
    ev_names = {m['region'].lower() for m in ev_m}
    oe_names = {m['region'].lower() for m in oe_m}

    # --- accuracy
    tp, n_oe = len(ev_names), len(oe_names)
    precision = tp / (tp + n_oe) if tp + n_oe else float('nan')
    recall = tp / len(truth)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    # Kendall tau-b: order of first mention in Findings vs evidence rank
    first = {}
    for m in ev_m:
        if m['section'].startswith('Findings') and m['region'].lower() not in first:
            first[m['region'].lower()] = (m['start'], m['evidence_rank'])
    if len(first) >= 2:
        pos, rk = zip(*first.values())
        tau = kendalltau(pos, rk, variant='b').statistic
    else:
        tau = float('nan')

    errors = []

    def err(rule, sev, a, b, ctx=''):
        errors.append({'report': rid, 'rule': rule, 'severity': sev,
                       'quote': md[a:b].strip(), 'evidence_context': ctx})

    sents = [(s, e) for _, a, b in search for s, e in sentence_spans(md, a, b)]

    def regions_in(a, b):
        return [m for m in ev_m if a <= m['start'] < b]

    # Out-of-evidence region (major); laterality (major) if its opposite-side twin is in the evidence
    for m in oe_m:
        flip = re.sub(r'^(left|right)\b', lambda x: 'right' if x.group(1) == 'left' else 'left', m['region'].lower())
        if flip != m['region'].lower() and flip in truth_by_name:
            err('laterality: wrong side', 'major', m['start'], m['end'],
                f"opposite side '{truth_by_name[flip]['anatomy']}' is evidence rank {truth_by_name[flip]['rank']}")
        else:
            err('out-of-evidence region', 'major', m['start'], m['end'],
                f"not in payload; found in {m['vocab_source']} vocabulary")

    # Summary laterality claims, with left/right/midline shares of summed relative weighting
    sh_truth = shares(truth)
    sh_all15 = shares(all_regions)
    top5 = shares(truth[:5])
    rank1_side = {'L': 'L', 'R': 'R'}.get(truth[0]['hemisphere'], 'M')
    def claim_scope(s, e, m):
        """Regions a laterality claim refers to.
        - claim inside a bold bullet header ('**X (bilateral):**'): the territory of the header's group term
        - otherwise the phrase after the claim up to the next , ; . or claim: the first named evidence
          region(s) there, else the territory of the first group term there
        - else the whole truth set."""
        line_s = md.rfind('\n', 0, m.start()) + 1
        hdr = re.match(r'\s*[-*]\s+\*\*(.+?)\*\*', md[line_s:e])
        if hdr and line_s + hdr.start(1) <= m.start() < line_s + hdr.end(1):
            phrase = hdr.group(1)
            named = []
        else:
            nxt = re.search(r'[,;.]|' + LATERALITY.pattern, md[m.end():e], re.I)
            p_end = m.end() + (nxt.start() if nxt else e - m.end())
            phrase = md[m.end():p_end]
            named = regions_in(m.end(), p_end)
        if named:
            regs = [truth_by_name[x['region'].lower()] for x in named]
            return regs, f"named regions after the claim: {', '.join(r['anatomy'] for r in regs)}"
        for g_pat, r_pat in TERRITORY:
            g = re.search(g_pat, phrase, re.I)
            if g:
                regs = [r for r in truth if re.search(r_pat, r['anatomy'], re.I)]
                if regs:
                    return regs, f"territory of '{phrase.strip()}' ({len(regs)} evidence regions)"
        return truth, 'whole truth set'

    for s, e in sents:
        for m in LATERALITY.finditer(md, s, e):
            claim = m.group(1).lower()
            scope_regs, scope = claim_scope(s, e, m)
            sh = shares(scope_regs)
            clause = next(md[a:b] for a, b in clause_spans(md, s, e) if a <= m.start() < b)
            side = 'L' if 'left' in claim else 'R' if 'right' in claim else None
            if side and PEAK_QUAL.search(clause):
                ok, basis = rank1_side == side, f'peak qualifier: rank-1 region side = {rank1_side}'
            elif side and TOPK_QUAL.search(md[s:e]):
                ok, basis = top5[side] > top5['R' if side == 'L' else 'L'], f'top-5 shares {top5}'
            elif side:
                ok, basis = sh[side] > sh['R' if side == 'L' else 'L'], f'{scope} shares {sh}'
            elif 'symmetric' in claim:
                lo, hi = sorted([sh_truth['L'], sh_truth['R']])
                ok, basis = (hi > 0 and lo / hi >= 0.8), f'whole truth set L/R ratio {lo / hi:.2f} (>=0.80 counts as symmetric)'
            else:  # bilateral
                ok, basis = sh['L'] > 0 and sh['R'] > 0, f'{scope} shares {sh}'
            errors.append({'report': rid, 'rule': f"laterality claim '{m.group(1)}'",
                           'severity': 'none' if ok else 'minor', 'quote': md[s:e].strip(),
                           'evidence_context': f"{basis}; truth-set shares L/R/midline = {sh_truth['L']}/{sh_truth['R']}/{sh_truth['M']}"
                                               f"; all 15 payload entries = {sh_all15['L']}/{sh_all15['R']}/{sh_all15['M']}"})

    # Top-region and ordering rules use the weighting descriptor of each region's clause (Findings only)
    tiers = {}
    for s, e in sents:
        for a, b in clause_spans(md, s, e):
            t = tier_of(md[a:b])
            for m in regions_in(a, b):
                if t == 1 and m['evidence_rank'] != 1:
                    err('top region not rank 1', 'major', a, b,
                        f"'{m['region']}' is evidence rank {m['evidence_rank']}; rank 1 is '{truth[0]['anatomy']}'")
                if t is not None and m['section'].startswith('Findings') and m['region'].lower() not in tiers:
                    tiers[m['region'].lower()] = (t, m['evidence_rank'], md[a:b].strip())
    for x, (tx, rx, cx) in tiers.items():
        for y, (ty, ry, _) in tiers.items():
            if rx < ry and tx > ty:   # x ranked higher but described more weakly than y
                idx = md.find(cx)
                err('ordering', 'minor', idx, idx + len(cx),
                    f"'{truth_by_name[x]['anatomy']}' (rank {rx}, tier {tx}) described more weakly than "
                    f"'{truth_by_name[y]['anatomy']}' (rank {ry}, tier {ty})")

    # Numbers (minor): digits in Findings/Impression except list markers
    for _, a, b in search:
        for m in re.finditer(r'\d+(?:\.\d+)?', md[a:b]):
            p = a + m.start()
            ls = md.rfind('\n', 0, p) + 1
            if re.fullmatch(r'\s*', md[ls:p]) and md[p + len(m.group(0)):p + len(m.group(0)) + 1] == '.':
                continue
            err('number', 'minor', p, p + len(m.group(0)))

    # Citations (major) and causal/diagnostic wording (major unless negated in the same sentence).
    # Technique, Findings and Impression are checked; Clinical question restates the payload target.
    t_s, t_e = spans.get('## Technique', (0, 0))
    for s, e in sentence_spans(md, t_s, t_e) + sents:
        for m in CITATION.finditer(md, s, e):
            err('citation', 'major', s, e)
        for m in CAUSAL.finditer(md, s, e):
            if not NEGATION.search(md[s:e]):
                err('causal/diagnostic wording', 'major', s, e, f"term '{m.group(0)}'")

    # Structural MRI (major): more than one sentence or names any region
    s_sents = sentence_spans(md, *smri)
    if len(s_sents) > 1:
        err('structural MRI: more than one sentence', 'major', *smri, f'{len(s_sents)} sentences')
    for m in matches:
        if m['section'] == 'Findings/Structural MRI' and m['kind'] != 'group_term':
            err('structural MRI: names a region', 'major', m['start'], m['end'])

    # Format (minor)
    if heads != EXPECTED_SECTIONS:
        err('format: sections', 'minor', 0, 0, f'got {heads}')
    n_points = len(re.findall(r'(?m)^\s*\d+\.\s', md[i_s:i_e]))
    if not 2 <= n_points <= 4:
        err('format: impression points', 'minor', i_s, i_e, f'{n_points} numbered points')
    import textstat
    words_total = textstat.lexicon_count(strip_md(md, keep_headings=True))
    if words_total > 250:
        err('format: over 250 words', 'minor', 0, 0, f'{words_total} words')

    row = {'model': model, 'atlas': atlas, 'n_truth': len(truth),
           'n_named': tp + n_oe, 'n_matched': tp, 'n_out_of_evidence': n_oe,
           'precision': round(precision, 3), 'recall': round(recall, 3), 'f1': round(f1, 3),
           'kendall_tau_b': round(tau, 3) if tau == tau else None,
           'n_major_errors': sum(e['severity'] == 'major' for e in errors),
           'n_minor_errors': sum(e['severity'] == 'minor' for e in errors),
           'missed_regions': '; '.join(r['anatomy'] for r in truth if r['anatomy'].lower() not in ev_names)}
    return row, matches, errors


# ---------------------------------------------------------------------------
# BERTScore + readability
# ---------------------------------------------------------------------------

def reference_text(atlas: str) -> str:
    truth, _ = truth_set(atlas)
    return ('Functional MRI: the model weighted the following regions, in order: '
            + ', '.join(r['anatomy'] for r in truth) + '. '
            'Structural MRI: no regional findings are reported because the model did not perform better than chance.')


def candidate_text(md: str) -> str:
    _, spans = sections(md)
    f, i = spans['## Findings'], spans['## Impression']
    return strip_md(md[f[0]:f[1]] + '\n' + md[i[0]:i[1]]).replace('\n', ' ')


def bertscore_and_readability() -> pd.DataFrame:
    import textstat
    from bert_score import BERTScorer
    ids, cands, refs = [], [], []
    for model in MODELS:
        for atlas in ATLASES:
            ids.append((model, atlas))
            cands.append(candidate_text((REPORTS / f'{model}_{atlas}.md').read_text()))
            refs.append(reference_text(atlas))
    runs = {
        'clinicalbert': dict(model_type='emilyalsentzer/Bio_ClinicalBERT', num_layers=9, idf=False,
                             rescale_with_baseline=False),
        'roberta_raw': dict(lang='en', idf=False, rescale_with_baseline=False),
        'roberta_rescaled': dict(lang='en', idf=False, rescale_with_baseline=True),
    }
    out = pd.DataFrame(ids, columns=['model', 'atlas'])
    for tag, kw in runs.items():
        scorer = BERTScorer(**kw)
        # Bio_ClinicalBERT's tokenizer config has no model_max_length (sentinel 1e30 overflows the
        # fast tokenizer). Use BERT's 512-token limit; all texts here are far shorter, so nothing is truncated.
        if scorer._tokenizer.model_max_length > 100_000:
            scorer._tokenizer.model_max_length = 512
        n_tok = max(len(scorer._tokenizer(t)['input_ids']) for t in cands + refs)
        assert n_tok <= scorer._tokenizer.model_max_length, n_tok
        P, R, F = scorer.score(cands, refs)
        out[f'{tag}_P'], out[f'{tag}_R'], out[f'{tag}_F1'] = P.numpy().round(4), R.numpy().round(4), F.numpy().round(4)
    rd = []
    for model, atlas in ids:
        md = (REPORTS / f'{model}_{atlas}.md').read_text()
        body = strip_md(md)
        rd.append({'words_body': textstat.lexicon_count(body),
                   'words_total': textstat.lexicon_count(strip_md(md, keep_headings=True)),
                   'flesch_reading_ease': textstat.flesch_reading_ease(body),
                   'flesch_kincaid_grade': textstat.flesch_kincaid_grade(body)})
    return pd.concat([out, pd.DataFrame(rd)], axis=1)


# ---------------------------------------------------------------------------

def prompt_diff() -> str:
    used = (REPORTS / 'prompt_used.txt').read_text()
    collapsed = re.sub(r'\n\n+', '\n', used)
    lines = ['prompt_used.txt vs paper Figure 4', f'identical: {used == FIG4}',
             f'identical after collapsing blank lines: {collapsed == FIG4}', '']
    lines += list(difflib.unified_diff(FIG4.splitlines(), used.splitlines(),
                                       'figure4', 'prompt_used.txt', lineterm='', n=0))
    return '\n'.join(lines) + '\n'


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / 'prompt_diff.txt').write_text(prompt_diff())
    vocab = build_vocab()
    for a in ATLASES:
        (HERE / f'vocab_{a}.txt').write_text('\n'.join(vocab[a]) + '\n')
    rows, matches, errors = [], [], []
    for model in MODELS:
        for atlas in ATLASES:
            r, m, e = score_report(model, atlas, vocab)
            rows.append(r); matches += m; errors += e
    pd.DataFrame(rows).to_csv(HERE / 'table4_accuracy.csv', index=False)
    pd.DataFrame(matches).to_csv(HERE / 'matches.csv', index=False)
    pd.DataFrame(errors, columns=['report', 'rule', 'quote', 'severity', 'evidence_context']) \
        .to_csv(HERE / 'errors.csv', index=False)
    (HERE / 'reference_texts.txt').write_text(''.join(f'[{a}]\n{reference_text(a)}\n\n' for a in ATLASES))
    if '--no-bertscore' not in sys.argv:
        bertscore_and_readability().to_csv(HERE / 'table5_bertscore_readability.csv', index=False)
    print('[score] done:', HERE)


if __name__ == '__main__':
    main()
