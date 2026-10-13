"""
llm_report.py — optional LLM-written summary of a Chapter 7 cell's
attribution maps.

Reads the saliency and prediction files already under results/ch7_v2/<cell>/,
applies a deterministic reliability gate (explain/llm_narrative.py), and asks
an OpenAI model to describe what passes the gate. Only cohort-level
aggregates are sent: no subject IDs, images or per-subject values. Run this
after report_saliency.py.

Two styles:
    explainer   plain-language summary for clinicians, with performance and
                fusion results (writes <cell>/llm/)
    clinician   region-only report in the style of an imaging report
                (writes <cell>/llm_clinician/)

The API key is read from the OPENAI_API_KEY environment variable; without it
the script runs as --dry-run. The model is --model, else $OPENAI_MODEL, else
gpt-5.

Usage:
    export OPENAI_API_KEY=...
    python src/llm_report.py --run results/ch7_v2/abide_schaefer_voxel_sharedsplit

    # The four Chapter 7 cells:
    python src/llm_report.py --run results/ch7_v2/{gsp_aal_voxel,gsp_schaefer_voxel,\\
abide_aal_voxel_sharedsplit,abide_schaefer_voxel_sharedsplit}

    # Region-only, imaging-report style (writes <cell>/llm_clinician/):
    python src/llm_report.py --run results/ch7_v2/abide_schaefer_voxel_sharedsplit --style clinician

    # No API call: write the evidence, gate and exact prompt only.
    python src/llm_report.py --run results/ch7_v2/gsp_aal_voxel --dry-run

Outputs in <cell>/llm/ (or <cell>/llm_clinician/):
    evidence.json      Full deterministic evidence (before redaction)
    reliability.json   Gate decision per modality + fusion notes
    prompt.json        Exact system + user message sent to the model
    response.json      Model id, response id, token usage (live runs only)
    report.md          Gate table (written by code) + LLM text
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from explain.llm_narrative import (DEFAULT_OPENAI_MODEL, assess_reliability,
                                   build_prompt, call_openai, collect_evidence,
                                   reliability_table_md)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--run', required=True, nargs='+', help='results/ch7_v2/<cell> directories')
    ap.add_argument('--model', default=os.environ.get('OPENAI_MODEL', DEFAULT_OPENAI_MODEL),
                    help=f'OpenAI model (default $OPENAI_MODEL or {DEFAULT_OPENAI_MODEL})')
    ap.add_argument('--top-k', type=int, default=15, help='Top ROIs passed to the LLM')
    ap.add_argument('--style', choices=['explainer', 'clinician'], default='explainer',
                    help='explainer: plain-language summary (writes <run>/llm/); '
                         'clinician: region-only imaging-report style (writes <run>/llm_clinician/)')
    ap.add_argument('--dry-run', action='store_true',
                    help='Skip the API call; write evidence, gate and prompt only')
    args = ap.parse_args()

    dry = args.dry_run
    if not dry and not os.environ.get('OPENAI_API_KEY'):
        print('[llm] OPENAI_API_KEY not set — falling back to --dry-run')
        dry = True
    for run in args.run:
        report_run(Path(run), args.model, args.top_k, dry, args.style)
    return 0


def report_run(run_dir: Path, model: str, top_k: int, dry: bool,
               style: str = 'explainer') -> None:
    if not run_dir.is_dir():
        sys.exit(f'run dir not found: {run_dir}')
    out = run_dir / ('llm' if style == 'explainer' else f'llm_{style}')
    out.mkdir(parents=True, exist_ok=True)

    ev  = collect_evidence(run_dir, top_k=top_k)
    rel = assess_reliability(ev)
    prompt = build_prompt(ev, rel, style)
    (out / 'evidence.json').write_text(json.dumps(ev, indent=2))
    (out / 'reliability.json').write_text(json.dumps(rel, indent=2))
    (out / 'prompt.json').write_text(json.dumps(prompt, indent=2))
    print(f'[llm] run:    {run_dir}')
    for mod, r in rel.items():
        if 'interpretable' in r:
            print(f"[llm] {mod}: interpretable={r['interpretable']} — {'; '.join(r['reasons'])}")

    if dry:
        narrative = ('_Dry run: no LLM call was made. The exact prompt is in '
                     '`prompt.json`._')
        provenance = 'dry run (no model called)'
    else:
        resp = call_openai(prompt, model=model)
        (out / 'response.json').write_text(json.dumps(
            {k: v for k, v in resp.items() if k != 'text'}, indent=2))
        narrative  = resp['text']
        provenance = f"{resp['model']} (response {resp['id']})"

    report = '\n'.join([
        f"# Attribution summary — {ev['run']}",
        '',
        f"> AI-generated explanation of model attribution maps for research use. "
        f"Not a diagnostic report. Generated {dt.date.today().isoformat()} by {provenance}.",
        '',
        f"**Task:** {ev['target']}  ",
        f"**Scope:** {ev['scope']}",
        '',
        '### Reliability gate (computed by code, not the LLM)',
        '',
        reliability_table_md(rel),
        '',
        '---',
        '',
        narrative,
        '',
    ])
    (out / 'report.md').write_text(report)
    print(f'[llm] wrote {out / "report.md"}')


if __name__ == '__main__':
    sys.exit(main())
