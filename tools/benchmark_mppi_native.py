"""Numerical equivalence and timed recorded-input experiment for C++ migration."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/rangermini_dynamic_semantic'))
from rangermini_dynamic_semantic.ranger_4ws_kinematics import Ranger4WSKinematics
from rangermini_dynamic_semantic.interaction_mppi import InteractionMPPI
from rangermini_dynamic_semantic.hybrid_sequence_mppi import HybridSequenceMPPI
from rangermini_dynamic_semantic.command_semantics_mppi import MPPIConfig
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph, Track
from rangermini_dynamic_semantic.native_kernels import prepare_points, library

def nested(x):
    return tuple(map(nested,x)) if isinstance(x,list) else x

def stats(samples):
    return dict(median_ms=float(np.median(samples)),p95_ms=float(np.percentile(samples,95)))

def timed(f, repeats=21):
    f()
    values=[]
    for _ in range(repeats):
        start=time.perf_counter();f();values.append((time.perf_counter()-start)*1000)
    return stats(values)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,
        default=ROOT/'artifacts/mppi_native_20261022');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    library()
    rng=random.Random(29)
    commands=[(rng.uniform(-.6,.6),rng.choice([0.,rng.uniform(-.6,.6)]),
               rng.uniform(-2,2),rng.choice([0.,.02,.12,-.01])) for _ in range(3000)]
    # Repeated modes and zero commands exercise aligned motion as well as holds.
    commands += [u for u in [(0.,0.,0.,.12),(.3,0.,.4,.12),(0.,.3,0.,.12),
                             (0.,0.,.8,.12),(-.3,0.,0.,.12)] for _ in range(40)]
    py=Ranger4WSKinematics(max_speed=.35,use_native=False)
    cpp=Ranger4WSKinematics(max_speed=.35,use_native=True)
    max_error=0.
    for u in commands:
        a=py.step(*u);b=cpp.step(*u)
        assert a[1]==b[1] and a[3]==b[3] and py.transition_pending==cpp.transition_pending
        error=max(abs(x-y) for x,y in zip(a[0]+a[2],b[0]+b[2]))
        max_error=max(max_error,error)
        assert error<1e-10,(u,a,b)
    points=prepare_points([(rng.uniform(-3,3),rng.uniform(-3,3)) for _ in range(180)])
    model=InteractionMPPI(MPPIConfig(allow_reverse=False))
    queries=[(rng.uniform(-2,2),rng.uniform(-2,2),rng.uniform(-math.pi,math.pi)) for _ in range(1000)]
    clearance_error=0.
    for q in queries:
        a=model._clearance_cost_python(*q,points);b=model._clearance_cost(*q,points)
        clearance_error=max(clearance_error,max(abs(x-y) for x,y in zip(a,b)))
    assert clearance_error<1e-10
    assert model._clearance_cost(0,0,0,[])==(0.,20.)
    def wheels(native):
        model=Ranger4WSKinematics(max_speed=.35,use_native=native)
        for u in commands:model.step(*u)
    def clearance(native):
        fun=model._clearance_cost if native else model._clearance_cost_python
        for q in queries:fun(*q,points)
    result={'kind':'numeric kernel migration, not navigation success evaluation',
        'wheel_steps':len(commands),'wheel_max_abs_error':max_error,
        'clearance_queries':len(queries),'clearance_max_abs_error':clearance_error,
        'wheel_workload':{'python':timed(lambda:wheels(False)), 'cpp':timed(lambda:wheels(True))},
        'clearance_workload':{'python':timed(lambda:clearance(False)), 'cpp':timed(lambda:clearance(True))}}
    run=ROOT/'artifacts/nav2_matched_cases_20261021/frozen_routefix_v3/offset_portal_multimode_s17'
    source=run/'episode/planner_observations.jsonl.gz'
    with gzip.open(source,'rt') as stream:observations=[json.loads(line) for line in stream]
    guides=[json.loads(s) for s in (run/'episode/route_guidance.jsonl').read_text().splitlines()]
    graph=SpatialGraph(run/'world.sdf')
    cfg=MPPIConfig(**json.loads((run/'episode/manifest.json').read_text())['mppi_config'])
    checks=[]
    for index in [0,10,20]:
        d=observations[index]
        def plan(native):
            planner=HybridSequenceMPPI(cfg,static_boxes=graph.static_boxes)
            planner.use_native_clearance=planner.use_native_kinematics=native
            planner.rng.setstate(nested(d['rng_state']))
            g=min(guides,key=lambda g:abs(g['sim_time']-d['sim_time']))
            planner.route_goal=g['guide'];planner.route_yaw=g['guide_yaw'];planner.reference_path=g['path']
            scan=(np.asarray([float('inf') if x is None else x for x in d['scan']['ranges']]),
                  d['scan']['angle_min'],d['scan']['angle_increment'])
            act=d['actuation']
            return planner.behavior_plan(d['pose'],d['goal'][0],scan,
                [Track(**x) for x in d['tracks']],d['sim_time'],'CONTINUE',goal_yaw=d['goal'][1],
                current_mode=act['mode'],current_steering_angles=act['wheel_steering_rad'],
                current_steering_transition=act['steering_transition_stop'])['plan']
        a=plan(False);b=plan(True)
        assert a['selected_behavior']==b['selected_behavior']
        assert a.get('selected_schedule')==b.get('selected_schedule')
        error=abs(a['predicted_score']-b['predicted_score'])
        assert error<1e-8
        trajectory_error=float(np.max(np.abs(np.asarray(a['selected_trajectory'])-b['selected_trajectory'])))
        assert trajectory_error<1e-8
        checks.append(dict(index=index,selected=a['selected_behavior'],score_error=error,
            trajectory_max_abs_error=trajectory_error,
            python=timed(lambda:plan(False),7),cpp=timed(lambda:plan(True),7)))
        print('Recorded input',index,'checked',flush=True)
    result['recorded_plans']=checks
    paths=[Path(__file__),source,ROOT/'integration/mppi_native_core/core.cpp',
           ROOT/'integration/mppi_native_core/build/libranger_mppi_native.so']
    paths+=list((ROOT/'src/rangermini_dynamic_semantic/rangermini_dynamic_semantic').glob('*.py'))
    result['sha256']={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    (args.output/'benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='sha256'},indent=2))

if __name__=='__main__':main()
