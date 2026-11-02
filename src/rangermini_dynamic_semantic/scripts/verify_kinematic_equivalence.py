#!/usr/bin/env python3
"""Paired numerical regression and recorded-observation planner benchmark."""
import argparse,gzip,importlib.util,json,random,statistics,sys,time
from pathlib import Path
import numpy as np
from rangermini_dynamic_semantic.ranger_4ws_kinematics import Ranger4WSKinematics
from rangermini_dynamic_semantic import observation_mppi
from rangermini_dynamic_semantic.command_semantics_mppi import MPPIConfig
from rangermini_dynamic_semantic.interaction_mppi import InteractionMPPI
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--reference',required=True);ap.add_argument('--episode',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
 spec=importlib.util.spec_from_file_location('reference_kinematics',args.reference);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 rng=random.Random(20261017);old=m.Ranger4WSKinematics(max_speed=.35);new=Ranger4WSKinematics(max_speed=.35);error=0.
 for i in range(10000):
  angles=[rng.uniform(-1.57,1.57) for _ in range(4)];speeds=[rng.uniform(-.35,.35) for _ in range(4)]
  old.angles=angles[:];new.angles=angles[:]
  a=old._twist_from_wheels(speeds);b=new._twist_from_wheels(speeds);error=max(error,max(abs(x-y) for x,y in zip(a,b)))
 assert error < 1e-12,error
 old=m.Ranger4WSKinematics(max_speed=.35);new=Ranger4WSKinematics(max_speed=.35);step_error=0.
 for i in range(3000):
  # Hold requests across several steps so alignment and actual motion are exercised.
  if i%15==0:cmd=rng.choice([(.3,0,.2),(.2,.2,0),(0,.3,0),(0,0,.4),(0,0,0),(-.2,0,-.2)])
  a=old.step(*cmd,.12);b=new.step(*cmd,.12)
  assert a[1:]==b[1:]
  step_error=max(step_error,max(abs(x-y) for x,y in zip(a[0],b[0])))
 assert step_error < 1e-12
 root=Path(args.episode);d=json.loads(gzip.open(root/'episode/planner_observations.jsonl.gz','rt').readline())
 route=json.loads((root/'episode/route_guidance.jsonl').read_text().splitlines()[0]);cfg=MPPIConfig(**json.loads((root/'episode/manifest.json').read_text())['mppi_config']);graph=SpatialGraph(root/'world.sdf')
 def nested(v):return tuple(nested(x) for x in v) if isinstance(v,list) else v
 scan=(np.array([float('inf') if x is None else x for x in d['scan']['ranges']]),d['scan']['angle_min'],d['scan']['angle_increment'])
 assert not d['tracks'],'Choose an observation without tracks for this fixed replay'
 timings={'general_solve':[],'closed_form':[]};bank_error=0.;reference=None
 for repeat in range(6):
  order=[('general_solve',m.Ranger4WSKinematics),('closed_form',Ranger4WSKinematics)]
  if repeat%2:order.reverse()
  for name,cls in order:
   observation_mppi.Ranger4WSKinematics=cls
   p=InteractionMPPI(cfg,prediction=True,static_boxes=graph.static_boxes);p.route_goal=route['guide'];p.route_yaw=route['guide_yaw'];p.wait_action_penalty=4;p.rng.setstate(nested(d['rng_state']))
   t=time.perf_counter();bank=p.bank(d['pose'],d['goal'][0],scan,[],d['sim_time'],current_mode=d['actuation']['mode'],goal_yaw=d['goal'][1],current_steering_angles=d['actuation']['wheel_steering_rad'],current_steering_transition=d['actuation']['steering_transition_stop']);elapsed=time.perf_counter()-t
   if repeat:timings[name].append(elapsed*1000)
   signature=[(c['behavior_id'],bool(c['feasible']),c['plan']['selected_behavior']) for c in bank]
   scores=np.array([c['plan']['predicted_score'] for c in bank]);actions=np.array([c['plan']['action'] for c in bank])
   if reference is None:reference=(signature,scores,actions)
   else:
    assert signature==reference[0]
    np.testing.assert_allclose(scores,reference[1],rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(actions,reference[2],rtol=1e-10,atol=1e-10)
    finite=np.isfinite(scores)&np.isfinite(reference[1]);bank_error=max(bank_error,float(np.max(np.abs(scores[finite]-reference[1][finite]))))
 result={'kind':'offline equivalence and paired planner replay; not control-frequency measurement','wheel_fit_cases':10000,'sequential_steps':3000,'max_fit_error':error,'max_step_error':step_error,'max_bank_score_error':bank_error,'timing_ms':timings,'median_ms':{k:statistics.median(v) for k,v in timings.items()},'reference':args.reference,'observation':args.episode}
 Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
