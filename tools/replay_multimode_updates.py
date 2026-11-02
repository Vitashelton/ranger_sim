#!/usr/bin/env python3
"""Same-observation MPPI update analysis; no new closed-loop episode is implied."""
import argparse,dataclasses,gzip,hashlib,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/rangermini_dynamic_semantic'))
from rangermini_dynamic_semantic.command_semantics_mppi import MPPIConfig,ranger_command
from rangermini_dynamic_semantic.interaction_mppi import InteractionMPPI
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph,Track
from rangermini_dynamic_semantic.ranger_4ws_kinematics import Ranger4WSKinematics

def nested(v):return tuple(nested(x) for x in v) if isinstance(v,list) else v

def rollout(pose,controls,act,cfg):
    chassis=Ranger4WSKinematics(max_speed=cfg.max_speed)
    chassis.mode=act['mode'];chassis.angles=list(act['wheel_steering_rad'])
    chassis.transition_pending=act['steering_transition_stop'];pose=list(pose);out=[pose[:]]
    for u in controls:
        cmd=ranger_command(*u,min_turn_radius=cfg.min_turn_radius,wheelbase=cfg.wheelbase)
        (vx,vy,w),_,_,_=chassis.step(*cmd[:3],cfg.dt)
        yaw=pose[2]+.5*w*cfg.dt;c,s=math.cos(yaw),math.sin(yaw)
        pose=[pose[0]+(c*vx-s*vy)*cfg.dt,pose[1]+(s*vx+c*vy)*cfg.dt,pose[2]+w*cfg.dt]
        out.append(pose[:])
    return out

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/translation_family_merge_20261021/update_analysis.json')
    args=parser.parse_args()
    all_rows=[];example=None;sources={}
    for name in ['collection_return_v5_visual_s17','collection_delivery_v5_visual_s17']:
        root=ROOT/'artifacts/public_mppi_benchmark_20261020'/name
        path=root/'episode/planner_observations.jsonl.gz'
        sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        cfg=MPPIConfig(**json.loads((root/'episode/manifest.json').read_text())['mppi_config'])
        graph=SpatialGraph(root/'world.sdf')
        with gzip.open(path,'rt') as f: observations=[json.loads(l) for l in f]
        for index,d in enumerate(observations):
            if index%10 or d['task_stage'] not in ('APPROACH','SERVICE_COLLECTION','SERVICE_CORRIDOR'):continue
            p=InteractionMPPI(cfg,prediction=True,static_boxes=graph.static_boxes)
            p.service_pose_sampling=d['task_stage'].startswith('SERVICE_')
            p.require_navigation_progress=False # Both updates share this setting.
            if p.service_pose_sampling:p.obs=dataclasses.replace(p.obs,arrival_radius=.16,arrival_heading_tolerance=math.radians(8))
            p.rng.setstate(nested(d['rng_state']));samples=[];evaluate=p._evaluate
            def capture(*args,**kwargs):
                value=evaluate(*args,**kwargs)
                if value is not None and kwargs.get('sequence') is None:
                    samples.append((args[4],value))
                return value
            p._evaluate=capture
            scan=(np.array([float('inf') if x is None else x for x in d['scan']['ranges']]),d['scan']['angle_min'],d['scan']['angle_increment'])
            tracks=[Track(**x) for x in d['tracks']]
            candidate=p.behavior_plan(d['pose'],d['goal'][0],scan,tracks,d['sim_time'],'CONTINUE',
                current_mode=d['actuation']['mode'],goal_yaw=d['goal'][1],
                current_steering_angles=d['actuation']['wheel_steering_rad'],
                current_steering_transition=d['actuation']['steering_transition_stop'])
            if not samples:continue
            costs=np.array([s['cost'] for _,s in samples]);weights=np.exp(-np.minimum(60,(costs-costs.min())/cfg.temperature));weights/=weights.sum()
            controls=np.sum(np.array([s['controls'] for _,s in samples])*weights[:,None,None],axis=0)
            points=p._scan_points_world(d['pose'],scan)
            if tracks:points=[x for x in points if all(math.dist(x,t.position)>t.radius+.18 for t in tracks)]
            pooled=evaluate(d['pose'],d['goal'][0],points,(),'POOLED',d['actuation']['mode'],d['goal'][1],sequence=controls.tolist(),
                current_steering_angles=d['actuation']['wheel_steering_rad'],current_steering_transition=d['actuation']['steering_transition_stop'])
            mapped=ranger_command(*controls[0]);residual=abs(float(controls[0,2])-mapped[2])
            groups=candidate['plan'].get('candidate_summary',[])
            record=dict(run=name,observation_index=index,sim_time=d['sim_time'],stage=d['task_stage'],
                sample_count=len(samples),feasible_families=sorted(set(b for b,_ in samples)),
                family_selection=candidate['plan']['selected_behavior'],family_cost=candidate['plan']['predicted_score'],
                pooled_feasible=pooled is not None,pooled_cost=pooled['cost'] if pooled else None,
                pooled_requested_first=controls[0].tolist(),pooled_native_first=list(mapped[:3]),yaw_mapping_residual=residual)
            all_rows.append(record)
            if (example is None and len(record['feasible_families'])>=2 and candidate['plan']['selected_behavior']!='STOP'
                    and residual>.02):
                example=dict(**record,pose=d['pose'],goal=d['goal'],actuation=d['actuation'],tracks=d['tracks'],
                    world=str(root/'world.sdf'),native_trajectory=candidate['plan']['selected_trajectory'],
                    native_first=candidate['plan']['action'],family_projected_first=list(ranger_command(*candidate['plan']['action'])[:3]),pooled_trajectory=rollout(d['pose'],controls,d['actuation'],cfg),
                    samples=[dict(family=b,cost=s['cost'],trajectory=s['trajectory']) for b,s in samples],
                    family_representatives=groups)
    result=dict(sampling_profile='merged_translation_v1',planner_source_sha256={name:hashlib.sha256((ROOT/'src/rangermini_dynamic_semantic/rangermini_dynamic_semantic'/name).read_bytes()).hexdigest() for name in ['observation_mppi.py','interaction_mppi.py','ranger_4ws_kinematics.py']},kind='fixed-observation counterfactual; same samples, costs and 4WS rollout; commitment and task-progress gating disabled in both',
        observation_rule='Every tenth observation in APPROACH, SERVICE_COLLECTION or SERVICE_CORRIDOR, return case before ordinary case',
        example_rule='First chronological analyzed observation with at least two feasible families, a moving family selection and pooled yaw mapping residual > 0.02 rad/s',
        source_sha256=sources,renderer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=all_rows,example=example)
    out=args.output;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(observations=len(all_rows),example={k:example[k] for k in ['run','sim_time','family_selection','family_cost','pooled_feasible','pooled_cost','pooled_requested_first','pooled_native_first']} if example else None),indent=2))
if __name__=='__main__':main()
