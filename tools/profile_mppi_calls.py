import sys,json,gzip,functools,dataclasses,cProfile,pstats
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src/rangermini_dynamic_semantic'))
from rangermini_dynamic_semantic.command_semantics_mppi import MPPIConfig
from rangermini_dynamic_semantic.stable_interaction_mppi import StableInteractionMPPI
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph,Track
run=root/'artifacts/executor_audit_20261021/process_verified';out=root/'post/group_meeting_20261021/mppi_runtime'
ep=run/'episode';m=json.loads((ep/'manifest.json').read_text())
with gzip.open(ep/'planner_observations.jsonl.gz','rt') as f:d=json.loads(next(f))
r=json.loads((ep/'route_guidance.jsonl').read_text().splitlines()[0]);g=SpatialGraph(run/'world.sdf')
p=StableInteractionMPPI(MPPIConfig(**m['mppi_config']),prediction=True,static_boxes=g.static_boxes)
p.guide_support=functools.partial(g.supports_pose,margin=.10);p.obs=dataclasses.replace(p.obs,arrival_radius=.24)
p.wait_action_penalty=p.obs.stop_action_penalty;p.route_goal=r['guide'];p.route_yaw=r['guide_yaw'];p.reference_path=r['path']
def nested(x):return tuple(nested(v) for v in x) if isinstance(x,list) else x
p.rng.setstate(nested(d['rng_state']));a=d['actuation']
prof=cProfile.Profile();prof.enable()
p.bank(d['pose'],d['goal'][0],([float('inf') if x is None else x for x in d['scan']['ranges']],d['scan']['angle_min'],d['scan']['angle_increment']),[Track(**x) for x in d['tracks']],d['sim_time'],current_mode=a['mode'],goal_yaw=d['goal'][1],current_steering_angles=a['wheel_steering_rad'],current_steering_transition=a['steering_transition_stop'])
prof.disable();prof.dump_stats(str(out/'hotspot_calls.prof'))
rows=[]
for (file,line,fn),(cc,nc,tt,ct,callers) in pstats.Stats(prof).stats.items():
 rows.append(dict(file=file,line=line,function=fn,primitive_calls=cc,total_calls=nc,self_ms=tt*1000,cumulative_ms=ct*1000))
rows.sort(key=lambda r:r['self_ms'],reverse=True)
(out/'hotspot_calls.json').write_text(json.dumps(dict(snapshot_index=0,qualification='cProfile call counts and function hot spots; tracing times not used in phase bar chart',functions=rows),indent=2))
print(json.dumps(rows[:10],indent=2))
