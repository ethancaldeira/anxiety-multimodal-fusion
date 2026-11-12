"""
paper_reports.py — clinician-style reports for the "Explaining the Brain" paper.

Extends llm_report.py --style clinician to three models from two providers
without changing it. The system prompt (CLINICIAN_PROMPT) and the user
message (redact_for_clinician payload) come from explain/llm_narrative.py,
built from the saved Chapter 7 evidence.json files; nothing is recomputed.

One call per model per atlas, default sampling, no extra instructions.
Retries are left to the SDKs (connection errors, 408/409/429, 5xx only).
Any other API error stops the run. Existing outputs are never overwritten.

Usage:
    source ~/.bashrc
    python src/paper_reports.py --dry-run   # writes prompt, payloads, evidence copies only
    python src/paper_reports.py             # makes the six API calls
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from explain.llm_narrative import assess_reliability, build_prompt

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = {
    'aal':      REPO / 'results/ch7_v2/abide_aal_voxel_sharedsplit/llm/evidence.json',
    'schaefer': REPO / 'results/ch7_v2/abide_schaefer_voxel_sharedsplit/llm/evidence.json',
}
OUT = REPO / 'results/ch7/paper_reports_2026-11'

# (slug, provider, requested model id, effort)
MODELS = [
    ('claude-fable-5-2', 'anthropic', 'claude-fable-5-2', 'xhigh'),
    ('gpt-6-astra',      'openai',    'gpt-6-astra',      'max'),
    ('gpt-6.1-sol',      'openai',    'gpt-6.1-sol',      'max'),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_prompt(atlas: str) -> dict:
    """Same prompt llm_report.py --style clinician builds; checked against the saved pilot."""
    ev_path = EVIDENCE[atlas]
    ev = json.loads(ev_path.read_text())
    prompt = build_prompt(ev, assess_reliability(ev), 'clinician')
    pilot = ev_path.parent.parent / 'llm_clinician' / 'prompt.json'
    if pilot.exists():
        saved = json.loads(pilot.read_text())
        if saved != prompt:
            sys.exit(f'prompt for {atlas} differs from {pilot}; refusing to run')
    return prompt


def call_anthropic(model: str, effort: str, prompt: dict) -> dict:
    import anthropic
    client = anthropic.Anthropic(max_retries=5)
    # Thinking is always on for Fable 5.2; depth is set by effort only.
    with client.messages.stream(
        model=model,
        max_tokens=64000,
        system=prompt['system'],
        messages=[{'role': 'user', 'content': prompt['user']}],
        output_config={'effort': effort},
    ) as stream:
        msg = stream.get_final_message()
    return {
        'text': ''.join(b.text for b in msg.content if b.type == 'text'),
        'response_id': msg.id,
        'model_returned': msg.model,
        'stop_reason': msg.stop_reason,
        'stop_details': msg.stop_details.model_dump() if getattr(msg, 'stop_details', None) else None,
        'usage': msg.usage.model_dump(),
        'input_tokens': msg.usage.input_tokens,
        'output_tokens': msg.usage.output_tokens,
    }


def call_openai(model: str, effort: str, prompt: dict) -> dict:
    from openai import OpenAI
    client = OpenAI(max_retries=5, timeout=3600)
    # Same Responses API call as explain/llm_narrative.call_openai, plus reasoning effort.
    resp = client.responses.create(
        model=model,
        instructions=prompt['system'],
        input=prompt['user'],
        reasoning={'effort': effort},
    )
    inc = getattr(resp, 'incomplete_details', None)
    return {
        'text': resp.output_text,
        'response_id': resp.id,
        'model_returned': resp.model,
        'stop_reason': resp.status if inc is None else f'{resp.status}: {inc.reason}',
        'usage': resp.usage.model_dump() if resp.usage else None,
        'input_tokens': resp.usage.input_tokens if resp.usage else None,
        'output_tokens': resp.usage.output_tokens if resp.usage else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dry-run', action='store_true', help='write prompt, payloads and evidence copies; no API calls')
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    prompts = {a: make_prompt(a) for a in EVIDENCE}
    system = {p['system'] for p in prompts.values()}
    assert len(system) == 1
    system = system.pop()

    prompt_file = OUT / 'prompt_used.txt'
    if not prompt_file.exists():
        prompt_file.write_text(system)
    elif prompt_file.read_text() != system:
        sys.exit(f'{prompt_file} exists with different content')
    for atlas, prompt in prompts.items():
        payload = OUT / f'payload_{atlas}.json'          # exact user message sent
        if not payload.exists():
            payload.write_text(prompt['user'])
        ev_copy = OUT / f'evidence_{atlas}.json'
        if not ev_copy.exists():
            shutil.copy2(EVIDENCE[atlas], ev_copy)
        assert sha256(ev_copy) == sha256(EVIDENCE[atlas])
    print(f'[paper] wrote prompt, payloads and evidence copies to {OUT}')
    if args.dry_run:
        return 0

    for slug, provider, model, effort in MODELS:
        for atlas, prompt in prompts.items():
            md, meta = OUT / f'{slug}_{atlas}.md', OUT / f'{slug}_{atlas}.json'
            if md.exists() or meta.exists():
                print(f'[paper] {slug} {atlas}: exists, skipping')
                continue
            print(f'[paper] {slug} {atlas}: calling {model} (effort {effort}) ...', flush=True)
            started = dt.datetime.now(dt.timezone.utc).isoformat()
            fn = call_anthropic if provider == 'anthropic' else call_openai
            res = fn(model, effort, prompt)            # any non-retryable error propagates and stops the run
            md.write_text(res.pop('text'))
            meta.write_text(json.dumps({
                'provider': provider,
                'model_requested': model,
                **res,
                'effort': effort,
                'timestamp_utc_request_start': started,
                'timestamp_utc_saved': dt.datetime.now(dt.timezone.utc).isoformat(),
                'evidence_path': str(EVIDENCE[atlas].relative_to(REPO)),
                'evidence_sha256': sha256(EVIDENCE[atlas]),
                'payload_file': f'payload_{atlas}.json',
                'prompt_file': 'prompt_used.txt',
            }, indent=2))
            print(f"[paper]   -> {res['model_returned']}  stop={res['stop_reason']}  "
                  f"in={res['input_tokens']} out={res['output_tokens']}", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
