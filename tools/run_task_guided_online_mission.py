"""One fresh language parse, online spatial VLM conditions and hybrid MPPI run."""
import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
INSTRUCTION = '帮我去617拿一下资料，交到614，拿到回执后再送去619。'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=37)
    p.add_argument('--scenario', default='multiroom_service')
    p.add_argument('--domain-id', type=int, default=224)
    p.add_argument('--instruction', default=INSTRUCTION)
    p.add_argument('--ask-key', action='store_true')
    a = p.parse_args()
    if a.ask_key:
        os.environ['DEEPSEEK_API_KEY'] = getpass.getpass('API credential (not saved): ')
    if not os.environ.get('DEEPSEEK_API_KEY'):
        raise SystemExit('DEEPSEEK_API_KEY required')
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    run = out / 'run'
    command = [sys.executable, str(ROOT/'src/rangermini_dynamic_semantic/scripts/run_grounded_delivery.py'),
        '--output', str(run), '--scenario', a.scenario, '--seed', str(a.seed),
        '--planner', 'ranger-mppi', '--hybrid-sequences', '--stable-modes',
        '--obstacle-seeded-sampling', '--verified-recovery', '--map-recovery',
        '--static-global-map', '--freeze-object-labels',
        '--unrestricted-task-motion',
        '--motion-executor','cpp',
        '--long-task', '--service-confirmation', 'simulation',
        '--policy', 'grounded', '--proposal-selector', 'vlm',
        '--proposal-interface', 'task_condition', '--spatial-conditions-only',
        '--task-heading-scale-deg','10','--task-clearance-scale-m','.25',
        '--samples', '6', '--instruction', a.instruction,
        '--max-sim-time', '240', '--max-wall-time', '650', '--real-time-factor', '.7',
        '--ros-domain-id', str(a.domain_id), '--ign-partition', out.name+'_online_task_20261023']
    (out/'protocol.json').write_text(json.dumps(dict(
        purpose='One integrated Gazebo episode: fresh language-plan parse, actual visual queries and hybrid MPPI execution.',
        instruction=a.instruction, seed=a.seed, command=command,
        task_plan_replay=False, motion_mode_output_allowed=False,
        instruction_style='Everyday service request; no user-specified wheel mode, heading or clearance.',
        grounding_evidence='Record actual model outputs and observed context; do not impute unstated preferences.',
        global_reference='Shared static-map global planner; no dynamic tracks or actor schedules in global guidance',
        motion_modes_assigned_by='Hybrid MPPI only',
        service_confirmation='Simulated acknowledgements, not physical material handling.',
        evidence_rule='All storyboard frames, model responses and executed trajectories must come from this run.',
        launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        ensure_ascii=False, indent=2)+'\n')
    print('START integrated online mission', flush=True)
    with (out/'runtime.log').open('w') as log:
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    result_path = run/'episode/result.json'
    if result_path.exists():
        subprocess.run([sys.executable, str(ROOT/'tools/summarize_long_service_task.py'), str(run)],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        result = json.loads(result_path.read_text())
        print('DONE', result['status'], 'visual requests', result.get('vlm_api_requests'), flush=True)
    else:
        print('Attempt retained without terminal navigation result; see runtime.log', flush=True)
    raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
