#!/usr/bin/env python3
"""Matched grounded-delivery matrix; preserves failures and independent audits."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from audit_grounded_delivery import audit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--scenarios', default='plain,headon,door_crossing')
    parser.add_argument('--policies', default='geometric,predictive,grounded')
    parser.add_argument('--seeds', default='17,23')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stable-modes', action='store_true')
    parser.add_argument('--verified-recovery', action='store_true')
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    trials = [(s, p, int(seed)) for s in args.scenarios.split(',') for p in args.policies.split(',')
              for seed in args.seeds.split(',')]
    if any(p in ('grounded', 'visual') for _, p, _ in trials) and not os.environ.get('DEEPSEEK_API_KEY'):
        raise SystemExit('Grounded matrix requires DEEPSEEK_API_KEY')
    completed = {}
    for terminal in sorted(root.glob('*/episode/result.json')):
        work = terminal.parent.parent
        manifest = json.loads((work/'episode/manifest.json').read_text())
        if bool(manifest.get('stable_motion_family_execution')) != args.stable_modes:
            raise SystemExit(f'Mismatched stable-motion configuration in {work}')
        revised = manifest.get('execution_profile', 'frozen_v7') != 'frozen_v7'
        if revised != args.verified_recovery:
            raise SystemExit(f'Mismatched execution revision in {work}')
        source = json.loads((work/'scene_manifest.json').read_text())['source_sha256']
        for name, expected in source.items():
            current = HERE/name
            if not current.exists():
                current = HERE.parent/'rangermini_dynamic_semantic'/name
            if hashlib.sha256(current.read_bytes()).hexdigest() != expected:
                raise SystemExit(f'Source revision changed since {work}; use a new matrix directory')
        row = audit(work)
        completed[(row['scenario'], row['policy'], row['seed'])] = row
    all_trials = sorted(set(trials) | set(completed))
    (root/'matrix_spec.json').write_text(json.dumps({'trials': all_trials, 'requested_trials': trials, 'workers': args.workers,
        'stable_motion_family_execution': args.stable_modes,
        'verified_recovery': args.verified_recovery,
        'primary_success': 'verified navigation AND no moving person/cart proxy overlap',
        'shared_conditions': 'world, task, kinematics, candidate families, sampling settings and seed',
        'policies': {'geometric': 'same candidate bank, frozen-current object occupancy',
            'predictive': 'same bank and tracked constant-velocity occupancy; minimum objective selection',
            'grounded': 'same predictive bank; real visual proposal, binding, held behavior and event feedback'}}, indent=2))
    def run(index, scenario, policy, seed):
        work = root/f'{scenario}_{policy}_s{seed}'
        if args.resume and (work/'episode/result.json').exists():
            return audit(work)
        if args.resume and work.exists():
            archive = root/'incomplete_attempts'
            archive.mkdir(exist_ok=True)
            preserved = archive/f'{work.name}_{time.time_ns()}'
            work.rename(preserved)
            (preserved/'interruption.json').write_text(json.dumps({
                'status': 'INCOMPLETE_ATTEMPT_PRESERVED',
                'reason': 'resume found existing raw data without a terminal result',
                'replacement_run_path': str(work)}, indent=2))
        start = time.monotonic()
        command = [sys.executable, str(HERE/'run_grounded_delivery.py'), '--output', str(work),
            '--scenario', scenario, '--policy', policy, '--seed', str(seed),
            '--ros-domain-id', str(120+index), '--ign-partition', f'grounded_matrix_{index}',
            '--max-sim-time', '110', '--max-wall-time', '440']
        if args.stable_modes:
            command.append('--stable-modes')
        if args.verified_recovery:
            command.append('--verified-recovery')
        with (root/f'{scenario}_{policy}_s{seed}.launcher.log').open('w') as log:
            try:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    timeout=460, env=os.environ.copy())
                exit_code = result.returncode
            except subprocess.TimeoutExpired:
                exit_code = 'TIMEOUT'
        if (work/'episode/result.json').exists():
            row = audit(work)
        else:
            row = {'scenario': scenario, 'policy': policy, 'seed': seed, 'status': 'NO_TERMINAL_RESULT',
                'interaction_success': False, 'navigation_success': False, 'run_path': str(work)}
        row.update(launcher_exit=exit_code, wall_s=time.monotonic()-start)
        return row
    rows = list(completed.values())
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, index, *trial) for index, trial in enumerate(trials)]
        for future in as_completed(futures):
            row = future.result()
            key = (row['scenario'], row['policy'], row['seed'])
            completed[key] = row
            rows = list(completed.values())
            rows.sort(key=lambda r: (r['scenario'], r['policy'], r['seed']))
            (root/'results.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
            fields = ['scenario', 'policy', 'seed', 'status', 'navigation_success', 'interaction_success',
                'elapsed_sim_s', 'path_length_m', 'person_signed_min_gap_m', 'trolley_signed_SAT_min_gap_m',
                'proxy_overlap_trace_count', 'accepted_proposals', 'rejected_proposals', 'semantic_queries',
                'mode_switches', 'wait_time_s']
            with (root/'results.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows({k: r.get(k) for k in fields} for r in rows)
            print(json.dumps({k: row.get(k) for k in ('scenario', 'policy', 'seed', 'status',
                'navigation_success', 'interaction_success', 'elapsed_sim_s', 'proxy_overlap_trace_count')},
                ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
