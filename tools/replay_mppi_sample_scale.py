"""Compare rollout budgets at three recorded Gazebo states; predictions only."""
from pathlib import Path
import argparse,csv,gzip,hashlib,json,math,statistics,sys,time
from dataclasses import replace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src/rangermini_dynamic_semantic'))
from rangermini_dynamic_semantic.hybrid_sequence_mppi import HybridSequenceMPPI
from rangermini_dynamic_semantic.command_semantics_mppi import MPPIConfig
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph,Track

class CountedPlanner(HybridSequenceMPPI):
 def __init__(self,*a,**kw):super().__init__(*a,**kw);self.proposed=[]
 def _scheduled_evaluation(self,schedule,args,kwargs,sequence=None,nominal=False,reference_goal=None):
  x=super()._scheduled_evaluation(schedule,args,kwargs,sequence=sequence,nominal=nominal,reference_goal=reference_goal)
  self.proposed.append(dict(structure=list(schedule[0]),boundaries=list(schedule[1]),phase=('weighted_recheck' if sequence is not None else 'nominal' if nominal else 'sample'),feasible=x is not None,cost=None if x is None else x['cost']))
  return x

def readgz(p):
 with gzip.open(p,'rt') as f:return [json.loads(s) for s in f]
def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 cases=[('trolley_encounter','online_v1',135),('office_614_departure','static_online_v1',531),('office_619_entry','static_online_v1',713)]
 budgets=[49,101,201];seeds=[11,29,47];base=ROOT/'artifacts/conversational_service_20261027';records=[];caseinfo=[];source=set()
 for name,runname,index in cases:
  run=base/runname/'run';ep=run/'episode';op=ep/'planner_observations.jsonl.gz';rp=ep/'hybrid_rollouts.jsonl.gz';manifest=ep/'manifest.json';route=ep/'route_guidance.jsonl';world=run/'world.sdf';source.update([op,rp,manifest,route,world])
  obs=readgz(op)[index];plans=readgz(rp);logged=min(plans,key=lambda x:abs(x['request_sim_time']-obs['sim_time']));assert abs(logged['request_sim_time']-obs['sim_time'])<.01
  condition=logged['plan'].get('applied_motion_condition');g=SpatialGraph(world);cfg=MPPIConfig(**json.loads(manifest.read_text())['mppi_config']);routes=[x for x in read(route) if x['sim_time']<=obs['sim_time'] and math.dist(x['task_goal'],obs['goal'][0])<.01];latest=routes[-1] if routes and obs['task_stage']=='APPROACH' else None
  scan=(np.asarray([float('inf') if x is None else x for x in obs['scan']['ranges']]),obs['scan']['angle_min'],obs['scan']['angle_increment']);tracks=[Track(**x) for x in obs['tracks']];act=obs['actuation'];stage=obs['task_stage']
  caseinfo.append(dict(name=name,run=str(run.relative_to(ROOT)),observation_index=index,sim_time=obs['sim_time'],pose=obs['pose'],goal=obs['goal'],task_stage=stage,active_condition=condition,condition_source='same-timestamp logged planner return; null denotes none recorded',visual_identity_note='Labels are model outputs tied to logged track IDs, not independently verified recognition.',route_record=latest))
  for seed in seeds:
   for budget in budgets:
    planner=CountedPlanner(cfg,prediction=True,static_boxes=g.static_boxes);planner.rng.seed(seed);planner.rollout_budget=budget;planner.active_motion_condition=condition;planner.obstacle_seeded_sampling=True;planner.require_navigation_progress=True
    planner.service_pose_sampling=stage.startswith('EXIT_') or stage=='SERVICE_CONFIRM';planner.capture_spin_only=False;planner.capture_heading_tolerance=None
    planner.obs=replace(planner.obs,arrival_radius=.16 if planner.service_pose_sampling else .04 if stage in ('ALIGN','CROSS') else .24,arrival_heading_tolerance=math.radians(8) if planner.service_pose_sampling else planner.obs.arrival_heading_tolerance)
    if latest:planner.route_goal=latest['guide'];planner.route_yaw=latest['guide_yaw'];planner.reference_path=latest['path']
    start=time.perf_counter();c=planner.behavior_plan(obs['pose'],obs['goal'][0],scan,tracks,obs['sim_time'],'CONTINUE',goal_yaw=obs['goal'][1],current_mode=act['mode'],current_steering_angles=act['wheel_steering_rad'],current_steering_transition=act['steering_transition_stop']);wall=(time.perf_counter()-start)*1000
    plan=c['plan'];assert plan['rollout_evaluations']<=budget
    feasible=[v for v in planner.proposed if v['feasible']];unique={tuple(v['structure'])+tuple(v['boundaries']) for v in feasible};trajectory=plan.get('selected_trajectory') or [];terminal=trajectory[-1] if trajectory else None
    result=dict(case=name,budget=budget,seed=seed,rollouts=plan['rollout_evaluations'],scheduled_attempts=len(planner.proposed),feasible_scheduled_rollouts=len(feasible),feasible_unique_schedules=len(unique),feasible_rate=(len(feasible)/len(planner.proposed) if planner.proposed else None),selected_behavior=plan.get('selected_behavior'),selected_schedule=plan.get('selected_schedule'),selected_cost=plan.get('predicted_score'),terminal_goal_distance_m=math.dist(terminal[:2],obs['goal'][0]) if terminal else None,terminal_heading_error_deg=abs(math.degrees(math.atan2(math.sin(terminal[2]-obs['goal'][1]),math.cos(terminal[2]-obs['goal'][1])))) if terminal and obs['goal'][1] is not None else None,planning_wall_ms=wall,selected_trajectory=trajectory,scheduled_records=planner.proposed)
    records.append(result);print(name,budget,seed,'rollouts',result['rollouts'],'feasible',len(feasible),'unique',len(unique),'cost',result['selected_cost'],flush=True)
 summary=[]
 for case,_,_ in cases:
  for budget in budgets:
   r=[x for x in records if x['case']==case and x['budget']==budget]
   summary.append(dict(case=case,budget=budget,n=len(r),rollouts_mean=statistics.mean(x['rollouts'] for x in r),feasible_scheduled_mean=statistics.mean(x['feasible_scheduled_rollouts'] for x in r),unique_feasible_mean=statistics.mean(x['feasible_unique_schedules'] for x in r),feasible_rate_mean=statistics.mean(x['feasible_rate'] for x in r if x['feasible_rate'] is not None) if any(x['feasible_rate'] is not None for x in r) else None,selected_cost_mean=statistics.mean(x['selected_cost'] for x in r if x['selected_cost'] is not None) if any(x['selected_cost'] is not None for x in r) else None,median_wall_ms=statistics.median(x['planning_wall_ms'] for x in r)))
 protocol=dict(kind='Recorded-state open-loop planning replay; not closed-loop Gazebo performance or semantic-attention evaluation',cases=caseinfo,budgets=budgets,seeds=seeds,rollout_definition='All calls to _evaluate, including primitive population and weighted rechecks. Same controller configuration and task condition within each case.',metric_definition='Feasible scheduled rollouts include nominal, noisy and weighted rechecks, not primitive calls. Unique feasible schedules count distinct structure plus transition indices.',condition_note='At 619 no active visual condition was recorded; goal and terminal heading are taken from the task stage. No semantic-attention policy implemented in this experiment.',random_note='Identical initial PRNG seed per case and budget; budget changes may alter subsequent consumption.',source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source)})
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 with gzip.open(out/'rollouts.jsonl.gz','wt') as f:
  for r in records:f.write(json.dumps(r)+'\n')
 with (out/'summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 print('DONE',len(records),'replays',flush=True)
if __name__=='__main__':main()
