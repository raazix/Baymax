"""Compare LLM configurations for the inspection assistant on real stored inspections.

Measures, per (model, evidence format): valid grounded answers, corrective retries, latency and prompt size. Creates
three inspections through the running API (flagged casting on drifting M-02, normal casting on M-01, steel image on
M-03), then calls analytics.assistant.explain in-process with the cache disabled. Results: data/assistant_eval.json.
"""
import asyncio
import glob
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import config  # noqa: F401  (loads .env)
from analytics import assistant
from analytics.briefing import evidence_packet

API = 'http://127.0.0.1:8000'
ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [('nvidia/nemotron-3-ultra-550b-a55b', False), ('nvidia/nemotron-3-ultra-550b-a55b', True),
           ('nvidia/nemotron-3-super-120b-a12b', True), ('nvidia/nemotron-3.5-lightning-30b-a3b', True)]
QUESTIONS = ['Summarize the image-derived findings, severity rule, recommended action, and any important model or measurement limits. Do not decide or approve the action.',
             'Why was this part flagged, and how severe is it?',
             'What is the risk for the next lot and what is driving it?']


def upload(path, query):
    data = Path(path).read_bytes()
    request = urllib.request.Request(f'{API}/api/inspections/upload?{query}', data=data, method='POST',
                                     headers={'Content-Type': 'image/png' if path.endswith('png') else 'image/jpeg'})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


async def main():
    demo = ROOT / 'demo_images'
    inspections = {
        'flagged casting / M-02': upload(sorted(glob.glob(str(demo / '2_casting_anomaly_PatchCore/defect_*')))[-1], 'model=casting&patchcore_model=default&machine_id=M-02&part_diameter_mm=120'),
        'normal casting / M-01': upload(sorted(glob.glob(str(demo / '2_casting_anomaly_PatchCore/normal_*')))[0], 'model=casting&patchcore_model=default&machine_id=M-01'),
        'steel image / M-03': upload(sorted(glob.glob(str(demo / '1_steel_defects_YOLO/crazing_*')))[0], 'model=neu&machine_id=M-03'),
    }
    results = []
    for model, compact in CONFIGS:
        for name, inspection in inspections.items():
            packet = assistant.compact_packet(inspection) if compact else evidence_packet(inspection)
            for question in QUESTIONS:
                started = time.perf_counter()
                try:
                    answer = await assistant.explain(inspection, question, model=model, compact=compact, use_cache=False)
                    outcome = {'ok': True, 'attempts': answer['attempts'], 'answer': answer['answer'],
                               'mentions_limits': any(w in answer['answer'].lower() for w in ('proxy', 'synthetic', 'simulated', 'not validated', 'hypothesis'))}
                except assistant.AssistantUnavailable as error:
                    outcome = {'ok': False, 'error': str(error)}
                outcome |= {'model': model, 'evidence': 'compact' if compact else 'full', 'inspection': name, 'question': question[:40],
                            'latency_s': round(time.perf_counter() - started, 2), 'prompt_chars': len(json.dumps(packet, separators=(',', ':')))}
                results.append(outcome)
                print(f"{model.split('/')[-1]:32} {outcome['evidence']:7} {name:24} ok={outcome['ok']!s:5} "
                      f"tries={outcome.get('attempts', '-')} {outcome['latency_s']:5.1f}s", flush=True)
    summary = []
    for model, compact in CONFIGS:
        rows = [r for r in results if r['model'] == model and r['evidence'] == ('compact' if compact else 'full')]
        ok = [r for r in rows if r['ok']]
        summary.append({'model': model, 'evidence': 'compact' if compact else 'full', 'answers': len(rows),
                        'grounded_ok': len(ok), 'needed_retry': sum(r['attempts'] > 1 for r in ok),
                        'mentions_limits': sum(r['mentions_limits'] for r in ok),
                        'median_latency_s': round(statistics.median(r['latency_s'] for r in rows), 2),
                        'mean_prompt_chars': round(statistics.mean(r['prompt_chars'] for r in rows))})
    out = ROOT / 'data' / 'assistant_eval.json'
    out.write_text(json.dumps({'summary': summary, 'results': results}, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    asyncio.run(main())
