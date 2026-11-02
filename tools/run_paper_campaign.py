#!/usr/bin/env python3
"""Frozen, serial Gazebo campaign. Keep failures and expose live progress."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'src/rangermini_dynamic_semantic/scripts'
sys.path.insert(0, str(SCRIPTS))
from run_research_motion_matrix import INSTRUCTIONS, audit_research_run

SEEDS = [41, 37, 53, 61, 67, 71, 73, 79, 83, 89]
REFERENCE = ROOT / 'artifacts/grounded_delivery_revision_20261016/selector_comparison_pose_complete_20261018'


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--cost-only', action='store_true', help='Continue API-independent trials; retain all completed visual attempts')
    parser.add_argument('--fresh-source', action='store_true', help='Freeze current sources and run all 110 trials without reusing the earlier revision')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((REFERENCE / 'comparison_spec.json').read_text())['source_sha256']
    if args.fresh_source:
        frozen = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in frozen}
    # Include scene generators as well as the immutable controller inventory.
    for path in SCRIPTS.glob('*scene.py'):
        frozen[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name, digest in frozen.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise SystemExit('Changed source; do not pool with reference: ' + name)
    trials = []
    for scenario in ['headon', 'heading_passing', 'encounter_entry']:
        for i, seed in enumerate(SEEDS):
            for selector in (['cost', 'vlm'] if i % 2 == 0 else ['vlm', 'cost']):
                name = f'{scenario}_{selector}_s{seed}'
                trials.append(dict(block='selector', scenario=scenario, seed=seed,
                    selector=selector, variant='full', flags=[],
                    output=str(REFERENCE / name if seed in SEEDS[:3] and not args.fresh_source else out / name),
                    reused=seed in SEEDS[:3] and not args.fresh_source))
    # Same cost selector and complete execution chain: one changed rollout ingredient.
    variants = [('full', []), ('instant_steering', ['--ideal-steering-rollout']),
                ('point_guides', ['--unprojected-behavior-guides'])]
    for scenario in ['offset_portal', 'encounter_entry']:
        for i, seed in enumerate(SEEDS):
            for variant, flags in variants[i % 3:] + variants[:i % 3]:
                if scenario == 'encounter_entry' and variant == 'full':
                    continue  # Reuse the ten selector-block cost controls; never double-count.
                trials.append(dict(block='mechanism', scenario=scenario, seed=seed,
                    selector='cost', variant=variant, flags=flags,
                    output=str(out / f'{scenario}_{variant}_s{seed}'), reused=False))
    reuse_count = sum(t['reused'] for t in trials)
    spec = dict(planned=110, reused=reuse_count, new=110-reuse_count, seeds=SEEDS, source_sha256=frozen,
        questions={
            'selector': 'Under the same admissible candidates and execution, what does visual task selection change?',
            'mechanism': 'Do steering-state rollout and footprint-supported guides improve executable motion?',
        },
        interpretation='60 selector episodes plus 60 mechanism comparisons sharing ten cost controls = 110 unique episodes; not external algorithm baselines.',
        shared_controls=[t['output'] for t in trials if t['block']=='selector' and t['scenario']=='encounter_entry' and t['selector']=='cost'],
        metrics=['verified arrival', 'signed proxy clearance', 'path and elapsed simulation time',
                 'heading during encounter', 'moving mode switches', 'alignment duration',
                 'same-query selection difference and acceptance', 'planning cost and API usage'],
        trials=trials)
    manifest = out / 'campaign_spec.json'
    if manifest.exists() and json.loads(manifest.read_text()) != spec:
        raise SystemExit('Refusing changed campaign resume')
    save(manifest, spec)
    print(f'Prepared {len(trials)} unique episodes, {reuse_count} reused and {110-reuse_count} new.', flush=True)
    if not args.execute:
        return
    if not args.cost_only and not os.environ.get('DEEPSEEK_API_KEY'):
        raise SystemExit('API credential missing; no experiment started')
    results = []
    def progress(state, active=None):
        save(out / 'progress.json', dict(state=state, planned=110, recorded=len(results),
             new_recorded=sum(not r['reused'] for r in results), active=active,
             updated_unix=time.time(), pid=os.getpid()))
        save(out / 'campaign_results.json', results)
    for trial in trials:
        if any(hashlib.sha256((ROOT / n).read_bytes()).hexdigest() != h for n, h in frozen.items()):
            progress('stopped_source_change', trial)
            raise SystemExit('Source changed; preserved completed episodes')
        work = Path(trial['output'])
        resultfile = work / 'episode/result.json'
        if args.cost_only and trial['selector'] == 'vlm' and not resultfile.exists():
            continue
        progress('running', trial)
        if not resultfile.exists():
            if work.exists():
                progress('stopped_incomplete_attempt', trial)
                raise SystemExit('Incomplete attempt retained: ' + str(work))
            command = [sys.executable, str(SCRIPTS / 'run_grounded_delivery.py'),
                '--output', str(work), '--scenario', trial['scenario'], '--policy', 'grounded',
                '--seed', str(trial['seed']), '--proposal-selector', trial['selector'],
                '--freeze-object-labels', '--stable-modes', '--verified-recovery', '--map-recovery',
                '--max-sim-time', '110', '--max-wall-time', '420', '--ros-domain-id', '187',
                '--ign-partition', 'ranger_matched_selectors', *trial['flags']]
            if trial['scenario'] in INSTRUCTIONS:
                command += ['--instruction', INSTRUCTIONS[trial['scenario']]]
            with (out / (work.name + '.launcher.log')).open('w') as log:
                completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode or not resultfile.exists():
                progress('stopped_run_error', trial)
                raise SystemExit('Run error; raw attempt retained: ' + str(work))
        result = json.loads(resultfile.read_text())
        audit = audit_research_run(work) if result['status'] != 'VLM_REQUEST_FAILED' else None
        results.append(dict(**trial, result=result, audit=audit))
        progress('running')
        print(json.dumps(dict(recorded=len(results), run=work.name, status=result['status'])), flush=True)
        # A failed visual startup is an observed failed attempt, not a reason
        # to discard it or prevent the remaining independent trials from running.
    progress('complete' if len(results) == 110 else 'cost_trials_complete_visual_pending')


if __name__ == '__main__':
    main()
