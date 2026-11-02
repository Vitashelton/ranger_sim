#!/usr/bin/env python3
"""Prepare or execute a matched matrix for the redesigned motion-planning study.

Preparation is read-only with respect to Gazebo and makes no API requests.
Use a new output directory after source changes; partial results are retained.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
METHODS={
    'predictive_4ws': ('predictive', False),
    'predictive_point_guides': ('predictive', False),
    'predictive_instant_steering': ('predictive', False),
    'visual_single_stage': ('grounded', False),
    'visual_two_stage': ('grounded', True),
    'direct_visual': ('visual', False),
}
INSTRUCTIONS={
    'heading_passing': '把资料送到 Office A（619 室）门内并停稳。迎面遇到推车时，保持前置相机朝向走廊来车方向，为对方留出通行空间，通行恢复后继续。',
    'offset_portal': '把资料送到 Office A（619 室）门内并停稳。门侧停放着器材，保持留距并选择合适的进入动作。',
    'encounter_entry': '把资料送到 Office A（619 室）门内并停稳。门前有来往推车和出入人员，让出必要通行空间，随后继续进入目的办公室。',
}

def audit_research_run(work):
    from audit_grounded_delivery import audit, rect_gap
    answer=audit(work)
    scene=json.loads((work/'scene_manifest.json').read_text())
    station=scene.get('parked_equipment_pose')
    gap=None
    if station is not None:
        rows=[json.loads(s) for s in (work/'episode/trace.jsonl').read_text().splitlines()]
        gap=min(rect_gap(r['pose'],[*station,0.]) for r in rows)
    answer['parked_station_proxy_min_gap_m']=gap
    answer['research_completion']=bool(answer['interaction_success'] and (gap is None or gap>0))
    (work/'research_audit.json').write_text(json.dumps(answer,ensure_ascii=False,indent=2))
    return answer

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',required=True)
    ap.add_argument('--scenarios',default='heading_passing,offset_portal,encounter_entry')
    ap.add_argument('--methods',default='predictive_4ws,predictive_instant_steering,direct_visual,visual_single_stage')
    ap.add_argument('--seeds',default='17,23,31,37,41')
    ap.add_argument('--execute',action='store_true')
    args=ap.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__).resolve(),HERE/'run_grounded_delivery.py',HERE/'motion_planning_research_scene.py',
             HERE/'continuous_delivery_scene.py',HERE/'grounded_delivery_scene.py']
    sources+=sorted((HERE.parent/'rangermini_dynamic_semantic').glob('*.py'))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    manifest=out/'research_matrix_spec.json'
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old['source_sha256']!=hashes:
            raise SystemExit('Source changed: preserve this batch and choose a new output directory.')
    trials=[]
    for scenario in args.scenarios.split(','):
        if scenario not in ('heading_passing','offset_portal','encounter_entry'):
            raise SystemExit('Unknown research scenario: '+scenario)
        for method in args.methods.split(','):
            if method not in METHODS:raise SystemExit('Unknown method: '+method)
            policy,lookahead=METHODS[method]
            for seed in map(int,args.seeds.split(',')):
                work=out/f'{scenario}_{method}_s{seed}'
                command=['/usr/bin/python3',str(HERE/'run_grounded_delivery.py'),
                    '--output',str(work),'--scenario',scenario,'--policy',policy,
                    '--instruction',INSTRUCTIONS[scenario],
                    '--seed',str(seed),'--stable-modes','--verified-recovery','--map-recovery',
                    '--max-sim-time','110','--max-wall-time','440',
                    '--ros-domain-id','184','--ign-partition','research_motion_matrix']
                if lookahead:command.append('--continuation-lookahead')
                if method == 'predictive_instant_steering':command.append('--ideal-steering-rollout')
                if method == 'predictive_point_guides':command.append('--unprojected-behavior-guides')
                trials.append(dict(scenario=scenario,method=method,seed=seed,
                    output=str(work),command=command))
    spec=dict(protocol='behavior_conditioned_multimode_mppi_v2',source_sha256=hashes,
        status='planned; outcomes must come from executed episodes',trials=trials,
        shared=['sensors','4WS execution adapter','scene/proximity triggers','seed',
                'first-stage samples','map guide','recovery profile'],
        additional_compute='two-stage method adds bounded continuation rollouts; record wall cost',
        excludes=['published Nav2 MPPI / MPPI-H: not implemented baselines in this runner',
                  'old matrix_v6: different source version'])
    manifest.write_text(json.dumps(spec,indent=2))
    print(f'Prepared {len(trials)} matched episodes: {manifest}')
    if not args.execute:return
    if any(METHODS[t['method']][0] in ('grounded','visual') for t in trials) and not os.environ.get('DEEPSEEK_API_KEY'):
        raise SystemExit('Vision credential is absent; plan saved without making API requests.')
    for trial in trials:
        work=Path(trial['output'])
        if (work/'episode/result.json').exists():
            audit_research_run(work);continue
        if work.exists():
            raise SystemExit(f'Incomplete attempt retained at {work}; choose a fresh batch for replacement.')
        with (out/(work.name+'.launcher.log')).open('w') as log:
            subprocess.run(trial['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
        if (work/'episode/result.json').exists():audit_research_run(work)

if __name__=='__main__':main()
