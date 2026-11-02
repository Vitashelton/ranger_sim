"""Task satisfaction and actual motion outcomes; no pooled optimizer costs."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src/rangermini_dynamic_semantic/scripts'),
              str(ROOT/'src/rangermini_dynamic_semantic')]
from audit_grounded_delivery import audit,rect_gap
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph


def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def main(batch):
    batch=batch.resolve();protocol=json.loads((batch/'protocol.json').read_text())
    out=batch/'analysis';out.mkdir(exist_ok=True)
    metrics=[];sources={}
    for t in protocol['trials']:
        run=batch/t['name']
        if not (run/'episode/result.json').exists():continue
        basic=audit(run);trace=rows(run/'episode/trace.jsonl')
        result=json.loads((run/'episode/result.json').read_text())
        graph=SpatialGraph(run/'world.sdf')
        static=[min(rect_gap(r['pose'],box[:3],bl=2*box[3],bw=2*box[4])
                    for box in graph.static_boxes) for r in trace]
        clearance=json.loads((run/'clearance_trace.json').read_text())
        if t['context']=='heading_passing':
            gaps=[r['trolley_signed_SAT_gap_m'] for r in clearance]
            centers=[r['trolley_pose'] for r in clearance]
        else:
            centers=[[9.,-2.1,0.]]*len(trace)
            gaps=[rect_gap(r['pose'],p) for r,p in zip(trace,centers)]
        dt=np.diff([r['timestamp'] for r in trace],append=trace[-1]['timestamp'])
        encounter=np.array([math.dist(r['pose'][:2],p[:2])<2.5 for r,p in zip(trace,centers)])
        yaw=np.array([math.atan2(math.sin(r['pose'][2]),math.cos(r['pose'][2])) for r in trace])
        weights=dt*encounter
        active=[];holds=0.;solver_errors=0;projection_deltas=[]
        mode_time={name:0. for name in ('DUAL_ACKERMANN','PARALLEL','SPINNING')}
        for r,d in zip(trace,dt):
            act=r.get('actuation',{});v=act.get('executed',{})
            if math.hypot(v.get('vx',0),v.get('vy',0))>.02 or abs(v.get('wz',0))>.02:
                active.append(act.get('mode'))
                if act.get('mode') in mode_time:mode_time[act['mode']]+=float(d)
            holds+=float(d)*bool(act.get('steering_transition_stop'))
            plan=r.get('plan',{});solver_errors+=bool(plan.get('public_solver_error'))
            if plan.get('public_raw_command') and plan.get('public_projected_command'):
                raw=plan['public_raw_command'];mapped=plan['public_projected_command']
                projection_deltas.append(math.sqrt((raw[0]-mapped[0])**2+(raw[1]-mapped[1])**2+(.247*(raw[2]-mapped[2]))**2))
        proposals=[]
        for p in sorted((run/'episode').glob('proposal_*_response.json')):
            response=json.loads(p.read_text());proposals.append(response['proposal'])
            sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
        relevant=[p['preferred_clearance_m'] for p in proposals if p['relevant_object_ids']]
        requested_clearance=statistics.mean(relevant) if relevant else None
        clearance_attainment=None
        clearance_shortfall=None
        if requested_clearance is not None and np.any(encounter) and float(weights.sum())>0:
            gap_array=np.asarray(gaps,dtype=float)
            clearance_attainment=float(np.average(
                (gap_array>=requested_clearance).astype(float)[encounter],
                weights=weights[encounter]))
            clearance_shortfall=float(np.average(
                np.maximum(requested_clearance-gap_array[encounter],0),
                weights=weights[encounter]))
        error_free=solver_errors==0
        metric=dict(context=t['context'],condition=t['condition'],method=t['method'],seed=t['seed'],
            status=result['status'],arrived=result['task_success'],
            collision_free_arrival=bool(result['task_success'] and basic['fixture_audit_valid'] and not basic['proxy_overlap_trace_count'] and min(static)>0),
            path_m=result['path_length_m'],elapsed_s=result['elapsed_sim_s'],interaction_gap_m=min(gaps),
            static_gap_m=min(static),actor_overlap_samples=basic['proxy_overlap_trace_count'],
            static_overlap_samples=sum(g<=0 for g in static),
            interaction_heading_rms_deg=math.degrees(math.sqrt(float(weights@(yaw*yaw))/max(float(weights.sum()),1e-9))),
            interaction_heading_max_deg=math.degrees(float(np.max(np.abs(yaw[encounter])))) if encounter.any() else None,
            nonzero_mode_changes=sum(a!=b for a,b in zip(active,active[1:])),alignment_s=holds,
            visual_responses=len(proposals),accepted_conditions=result['accepted_proposals'],
            requested_clearance_m=requested_clearance,
            clearance_attainment_fraction=clearance_attainment,
            clearance_shortfall_mean_m=clearance_shortfall,
            heading_retention_requested=any(p['heading_policy']=='preserve_current_heading' for p in proposals),
            raw_family_preference_used=any(p['motion_preference']!='auto' for p in proposals),
            public_solver_error_records=solver_errors,
            public_native_projection_mean=statistics.mean(projection_deltas) if projection_deltas else None,
            public_solver_error_free=error_free)
        moving_s=sum(mode_time.values())
        for mode,value in mode_time.items():metric[mode.lower()+'_moving_fraction']=value/max(moving_s,1e-9)
        metrics.append(metric)
        for p in [run/'episode/trace.jsonl',run/'episode/result.json',run/'world.sdf',run/'scene_manifest.json']:
            sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    if metrics:
        with (out/'episode_metrics.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    report=dict(kind='Matched task-conditioned navigation; incomplete cells remain unfilled.',
        planned_episodes=protocol['planned_episodes'],completed_episodes=len(metrics),metrics=metrics,
        heading_measure='Body yaw relative to initial corridor heading, evaluated within 2.5 m of the relevant obstacle center; a continuous soft-preference metric, not a hard completion gate.',
        interaction_measure='Moving-trolley signed rectangle gap for heading_passing; registered equipment proxy gap for static_detour.',
        public_baseline_scope='Official Omni optimizer/critics, task costmap/reference adapter and common Ranger execution. Solver health, command projection and geometry reviewed separately from arrival.',
        source_sha256=sources)
    (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    for r in metrics:print(r['context'],r['condition'],r['method'],r['seed'],r['status'],
                           'collision_free_arrival',r['collision_free_arrival'],
                           'object_gap',round(r['interaction_gap_m'],3),
                           'heading_rms_deg',round(r['interaction_heading_rms_deg'],2),
                           'solver_errors',r['public_solver_error_records'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('batch',type=Path)
    main(p.parse_args().batch)
