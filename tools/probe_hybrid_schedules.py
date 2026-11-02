"""Recorded-observation planning experiment; no closed-loop success is inferred."""
import argparse,gzip,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src/rangermini_dynamic_semantic'))
from rangermini_dynamic_semantic.hybrid_sequence_mppi import HybridSequenceMPPI
from rangermini_dynamic_semantic.command_semantics_mppi import MPPIConfig
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph,Track

def nested(x):return tuple(map(nested,x)) if isinstance(x,list) else x
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'artifacts/hybrid_sequence_20261022/recorded_probe.json');p.add_argument('--indices',type=int,nargs='+',default=[0,10,20]);a=p.parse_args()
run=ROOT/'artifacts/nav2_matched_cases_20261021/frozen_routefix_v3/offset_portal_multimode_s17'
source=run/'episode/planner_observations.jsonl.gz'
with gzip.open(source,'rt') as f:observations=[json.loads(s) for s in f]
guides=[json.loads(s) for s in (run/'episode/route_guidance.jsonl').read_text().splitlines()]
graph=SpatialGraph(run/'world.sdf');cfg=MPPIConfig(**json.loads((run/'episode/manifest.json').read_text())['mppi_config'])
results=[]
for index in a.indices:
 d=observations[index];planner=HybridSequenceMPPI(cfg,static_boxes=graph.static_boxes)
 planner.rng.setstate(nested(d['rng_state']))
 g=min(guides,key=lambda g:abs(g['sim_time']-d['sim_time'])) if guides else None
 if g:planner.route_goal=g['guide'];planner.route_yaw=g['guide_yaw'];planner.reference_path=g['path']
 scan=(np.asarray([float('inf') if x is None else x for x in d['scan']['ranges']]),d['scan']['angle_min'],d['scan']['angle_increment'])
 act=d['actuation'];tracks=[Track(**x) for x in d['tracks']]
 c=planner.behavior_plan(d['pose'],d['goal'][0],scan,tracks,d['sim_time'],'CONTINUE',goal_yaw=d['goal'][1],current_mode=act['mode'],current_steering_angles=act['wheel_steering_rad'],current_steering_transition=act['steering_transition_stop'])
 results.append(dict(index=index,pose=d['pose'],sim_time=d['sim_time'],actuation=act,goal=c['goal'],plan=c['plan']))
 print(index,c['plan']['selected_behavior'],c['plan'].get('selected_schedule'),len(c['plan'].get('hybrid_schedule_candidates',[])),flush=True)
sources=[source,run/'world.sdf',run/'episode/route_guidance.jsonl',run/'episode/manifest.json',ROOT/'src/rangermini_dynamic_semantic/rangermini_dynamic_semantic/hybrid_sequence_mppi.py',ROOT/'src/rangermini_dynamic_semantic/rangermini_dynamic_semantic/observation_mppi.py',Path(__file__)]
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(kind='same-observation predictive rollout; not executed Gazebo trajectories',results=results,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),indent=2)+'\n')
