"""Audit completed matched offset-pose runs without pooling historical versions."""
from pathlib import Path
import json,sys,math,csv
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src/rangermini_dynamic_semantic/scripts'),str(ROOT/'src/rangermini_dynamic_semantic')]
from audit_grounded_delivery import audit,rect_gap
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph
base=ROOT/'artifacts/offset_pose_public_pair_20261029/steering_ablation_v1'
results=[]
for seed in (53,37,41):
 for method in ('ideal',):
  run=base/f'{method}_s{seed}';p=run/'episode/result.json'
  if not p.exists():continue
  result=json.loads(p.read_text());summary=audit(run);trace=[json.loads(x) for x in (run/'episode/trace.jsonl').read_text().splitlines() if x.strip()];graph=SpatialGraph(run/'world.sdf')
  gaps=[min(rect_gap(row['pose'],b[:3],bl=2*b[3],bw=2*b[4]) for b in graph.static_boxes) for row in trace]
  active=[];align=0;spin=0
  for a,b in zip(trace,trace[1:]):
   dt=max(0,b['timestamp']-a['timestamp']);v=a['actuation']['executed'];mode=a['actuation']['mode'];align+=dt*bool(a['actuation'].get('steering_transition_stop'));spin+=dt*(mode=='SPINNING')
   if math.hypot(v['vx'],v['vy'])>.02 or abs(v['wz'])>.02:active.append(mode)
  results.append(dict(seed=seed,planner=method,status=result['status'],elapsed_s=result['elapsed_sim_s'],path_m=result['path_length_m'],heading_error_deg=result['heading_error_deg'],position_error_m=result['position_error_m'],static_min_gap_m=min(gaps),static_overlap_samples=sum(g<=0 for g in gaps),actor_overlap_samples=summary['proxy_overlap_trace_count'],fixture_audit_valid=summary['fixture_audit_valid'],spinning_mode_s=spin,steering_alignment_s=align,active_mode_switches=sum(a!=b for a,b in zip(active,active[1:])),solver_error_samples=sum(bool(a.get('plan',{}).get('public_solver_error')) for a in trace)))
if results:
 out=base/'metrics.csv'
 with out.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
 print(json.dumps(results,indent=2))
