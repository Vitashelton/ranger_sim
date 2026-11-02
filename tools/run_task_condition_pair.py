"""Freeze and run one matched semantic-condition mechanism pair in Gazebo."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=41)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    files = list((ROOT/'src/rangermini_dynamic_semantic/rangermini_dynamic_semantic').glob('*.py'))
    files += [ROOT/'src/rangermini_dynamic_semantic/scripts/run_grounded_delivery.py', Path(__file__),
              ROOT/'integration/mppi_native_core/build/libranger_mppi_native.so']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    instruction = '把资料送到619室门外等候点并停稳。与推车会车时尽量保持车头朝向走廊前方，平稳避让，通过后继续配送。'
    reference = ROOT/'artifacts/hybrid_sequence_20261022/headon_local_vlm_pilot_v2/common_static_reference_s37.json'
    protocol = dict(kind='paired task-condition mechanism check; not an algorithm ranking',
        seed=args.seed, instruction=instruction, goal=[12.8, -2.1, 0],
        methods=['vlm', 'no_condition'], source_sha256=hashes,
        reference=str(reference.relative_to(ROOT)),
        reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        condition_weight_per_s=1., all_families_enabled=True, max_rollouts=101,
        common_execution='one current-state bank per planning cycle; no legacy behavior gate',
        common_sampling='route reference plus symmetric track-derived local seeds; same rollout cap',
        geometry_labels_frozen=True)
    (out/'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+'\n')
    for index, method in enumerate(protocol['methods']):
        for name, expected in hashes.items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
                raise RuntimeError('Source changed during pair: '+name)
        run = out/method
        command = [sys.executable, str(ROOT/'src/rangermini_dynamic_semantic/scripts/run_grounded_delivery.py'),
            '--output', str(run), '--scenario', 'heading_passing', '--seed', str(args.seed),
            '--planner', 'ranger-mppi', '--hybrid-sequences', '--stable-modes', '--obstacle-seeded-sampling',
            '--verified-recovery', '--map-recovery', '--samples', '8',
            '--freeze-object-labels',
            '--max-sim-time', '80', '--max-wall-time', '360', '--real-time-factor', '.5',
            '--local-pose-goal', '12.8', '-2.1', '0', '--frozen-global-reference', str(reference),
            '--instruction', instruction, '--ros-domain-id', str(166+index),
            '--ign-partition', out.name+'_'+method]
        command += (['--policy', 'grounded', '--proposal-selector', 'vlm', '--proposal-interface', 'task_condition']
                    if method == 'vlm' else ['--policy', 'predictive', '--proposal-selector', 'cost'])
        print('START', method, flush=True)
        with (out/(method+'.log')).open('w') as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        result = run/'episode/result.json'
        if not result.exists():
            raise RuntimeError(f'Attempt retained without terminal result: {method}, exit {completed.returncode}')
        record = json.loads(result.read_text())
        print('DONE', method, record['status'], record['elapsed_sim_s'], flush=True)
    (out/'complete.json').write_text(json.dumps(dict(complete=True, methods=protocol['methods']))+'\n')


if __name__ == '__main__':
    main()
