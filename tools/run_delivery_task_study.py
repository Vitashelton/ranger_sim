#!/usr/bin/env python3
"""One route, crossed aisle-width/instruction factors, matched task-level outcomes.

Default writes a protocol, never asserts planned trials were executed. Pilot
runs are excluded from the frozen comparison to avoid tuning on reported seeds.
"""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SCRIPTS=ROOT/'src/rangermini_dynamic_semantic/scripts'
from audit_delivery_task import audit_task
INSTRUCTIONS={
 'ordinary':'把资料送到 Office A（619室）门内并停稳。经过器材通道时与推车保持安全距离，随后继续到达办公室。',
 'priority':'把资料送到 Office A（619室）门内并停稳。先在器材通道外让推车完全驶出通道，再进入通道，随后继续送达办公室。',
}
def save(path,value):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(path)
def scene_preflight():
 sys.path[:0]=[str(SCRIPTS),str(ROOT/'src/rangermini_dynamic_semantic')]
 import tempfile
 from motion_planning_research_scene import create_world,SPECS
 from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph
 from rangermini_dynamic_semantic.footprint_route_guide import FootprintRouteGuide
 from audit_grounded_delivery import rect_gap
 checks=[]
 with tempfile.TemporaryDirectory(prefix='ranger_task_scene_') as tmp:
  for name in ['delivery_aisle_open','delivery_aisle_narrow']:
   world=Path(tmp)/(name+'.sdf');meta=create_world(world,name);g=SpatialGraph(world);spec=SPECS[name];q=spec['start']
   route=FootprintRouteGuide(g).guide(q,(12.8,-1.15))[1]
   static=min(rect_gap(q,[x,y,a],bl=2*hx,bw=2*hy) for x,y,a,hx,hy in g.static_boxes)
   waiting=rect_gap(q,[q[0],spec['cart'][1],0])
   inside=rect_gap(q,[(sum(spec['aisle_x'])/2),-2.1,0],bl=spec['aisle_x'][1]-spec['aisle_x'][0],bw=spec['aisle_width'])<0
   # Straight, heading-retaining passage of the two registered chassis proxies.
   side_available=spec['aisle_width']/2-.5/2-.10
   side_required=.42/2+.5/2+.12
   item=dict(scene=name,route_found=bool(route),initial_static_gap_m=static,waiting_cart_gap_m=waiting,start_outside_aisle=not inside,parallel_passing_margin_m=side_available-side_required)
   if not(route and static>=.10 and waiting>=.12 and not inside):raise SystemExit('Invalid task scene: '+json.dumps(item))
   checks.append(item)
 return checks

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--execute',action='store_true');ap.add_argument('--seeds',default='41,53,61');ap.add_argument('--selectors',default='cost,vlm');args=ap.parse_args()
 out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
 preflight=scene_preflight()
 paths=[Path(__file__).resolve(),ROOT/'tools/audit_delivery_task.py']+list(SCRIPTS.glob('*scene.py'))+[SCRIPTS/'run_grounded_delivery.py']+list((ROOT/'src/rangermini_dynamic_semantic/rangermini_dynamic_semantic').glob('*.py'))
 hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
 trials=[]
 for scene in ['delivery_aisle_open','delivery_aisle_narrow']:
  for condition,instruction in INSTRUCTIONS.items():
   for seed in map(int,args.seeds.split(',')):
    selectors=args.selectors.split(',');selectors=selectors if seed%2 else selectors[::-1]
    for selector in selectors:
     if selector not in ['cost','vlm']:raise SystemExit('Unknown selector')
     work=out/f'{scene}_{condition}_{selector}_s{seed}'
     cmd=['/usr/bin/python3',str(SCRIPTS/'run_grounded_delivery.py'),'--output',str(work),'--scenario',scene,'--policy','grounded','--proposal-selector',selector,'--instruction',instruction,'--seed',str(seed),'--freeze-object-labels','--stable-modes','--verified-recovery','--map-recovery','--max-sim-time','85','--max-wall-time','240','--ros-domain-id','193','--ign-partition','ranger_delivery_task_20261020']
     trials.append(dict(scene=scene,instruction_condition=condition,selector=selector,seed=seed,output=str(work),command=cmd))
 spec=dict(protocol='delivery_aisle_v3',status='planned, not measured results',route='same document delivery to Office 619',
  scene_preflight=preflight,
  factors=dict(aisle_width_m=[1.9,1.5],instruction_conditions=list(INSTRUCTIONS)),
  instructions=INSTRUCTIONS,primary_metrics=['verified arrival without actor/static proxy overlap and instruction violation','priority compliance when requested'],
  secondary_metrics=['completion time for successful trials','terminal time and failure category for all trials'],
  diagnostic_only=['minimum clearance','wheel angles','behavior labels','planner costs'],
  evaluation_rule='priority: no positive robot-footprint occupancy of aisle before acknowledged trolley rear clears the entrance; registered at protocol creation',
  shared=['world and actor proximity triggers per scene','MPPI and 4WS execution code','sampling budget','random seed','object labels','route anchors','recovery'],
  changed='selector only within each matched instruction/geometry/seed pair; no external MPPI-H replication claimed',
  pilot_excluded=['seed37; scene calibration only'],planned_episodes=len(trials),source_sha256=hashes,trials=trials)
 manifest=out/'task_study_spec.json'
 if manifest.exists() and json.loads(manifest.read_text())!=spec:raise SystemExit('Changed frozen study; use a fresh output')
 save(manifest,spec);print(f'Prepared {len(trials)} episodes; zero completed outcomes implied.',flush=True)
 if not args.execute:return
 if any(t['selector']=='vlm' for t in trials) and not os.environ.get('DEEPSEEK_API_KEY'):raise SystemExit('Visual credential missing; protocol retained, no trials started')
 results=[]
 for t in trials:
  if any(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h for n,h in hashes.items()):raise SystemExit('Source changed; retain current runs')
  work=Path(t['output']);save(out/'progress.json',dict(state='running',completed=len(results),planned=len(trials),active=t['output'],pid=os.getpid(),updated=time.time()))
  if not (work/'episode/result.json').exists():
   if work.exists():raise SystemExit('Incomplete run retained; refusing overwrite')
   with (out/f'{work.name}.launcher.log').open('w') as log:
    proc=subprocess.run(t['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
   if proc.returncode or not (work/'episode/result.json').exists():raise SystemExit('Infrastructure error retained; do not count as a navigation outcome')
  answer=audit_task(work,t['instruction_condition']=='priority');results.append({**{k:v for k,v in t.items() if k!='command'},'task_outcome':answer});save(out/'task_results.json',results)
  print(json.dumps(dict(completed=len(results),run=work.name,task_success=answer['task_success'],failure=answer['failure_category'])),flush=True)
 save(out/'progress.json',dict(state='complete',completed=len(results),planned=len(trials),pid=os.getpid(),updated=time.time()))
if __name__=='__main__':main()
