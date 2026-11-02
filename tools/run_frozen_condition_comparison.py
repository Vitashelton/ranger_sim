"""Matched Nav2/native comparisons with the same recorded visual condition."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import contextlib,hashlib,io,json,subprocess,sys,threading
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src/rangermini_dynamic_semantic/scripts'),str(ROOT/'src/rangermini_dynamic_semantic'),str(ROOT/'tools')]
from motion_planning_research_scene import create_world
from freeze_static_global_reference import freeze
out=ROOT/'artifacts/task_attention_service_20261030/frozen_planner_comparison_v1';out.mkdir(parents=True,exist_ok=False)
world=out/'heading_passing.sdf';create_world(world,'heading_passing');ref=out/'static_reference.json'
with contextlib.redirect_stdout(io.StringIO()):freeze(world,(3.,-2.1),(12.8,-2.1),0.,0.,ref)
source=ROOT/'artifacts/task_attention_service_20261030/pair_v4/fragile/episode';default=source/'proposal_001_response.json';target=source/'proposal_002_response.json'
instruction='帮我把这份报告送到619门口。路上那辆推车放着易碎样品，麻烦经过时离它远一些。'
trials=[{'seed':seed,'planner':method,'domain':220+i} for i,(seed,method) in enumerate((seed,method) for seed in (67,71,73) for method in ('nav2','hybrid'))]
files=[Path(__file__),ROOT/'src/rangermini_dynamic_semantic/scripts/run_grounded_delivery.py',ROOT/'src/rangermini_dynamic_semantic/scripts/motion_planning_research_scene.py',ROOT/'integration/mppi_native_core/build/libranger_mppi_native.so',ROOT/'integration/nav2_mppi_adapter/build/libranger_nav2_mppi.so',ROOT/'install/ranger_sim_motion_cpp/lib/ranger_sim_motion_cpp/ranger_motion_adapter_cpp',world,ref,default,target]+list((ROOT/'src/rangermini_dynamic_semantic/rangermini_dynamic_semantic').glob('*.py'))
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(out/'protocol.json').write_text(json.dumps({'purpose':'Matched public and multimode MPPI planner comparison with identical recorded visual conditions','instruction':instruction,'trials':trials,'visual_condition':'Actual VLM outputs recorded in pair_v4, replayed with live object-ID validity gating. No new visual inference in these runs.','target_clearance_m':.85,'default_clearance_m':.65,'heading_policy':'route_heading','common':'same scenario, start, goal, static-map reference, actor trigger, evidence gate and 4WS execution per pair','planner_budget':'Nav2 1000 samples; native samples=8 with bounded sequence search','source_sha256':hashes},ensure_ascii=False,indent=2)+'\n')
lock=threading.Lock();results=[]
def run(t):
 for n,h in hashes.items():
  if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h:raise RuntimeError('Source changed during comparison: '+n)
 name=f"{t['planner']}_s{t['seed']}";path=out/name
 cmd=[sys.executable,str(ROOT/'src/rangermini_dynamic_semantic/scripts/run_grounded_delivery.py'),'--output',str(path),'--scenario','heading_passing','--seed',str(t['seed']),'--planner','nav2-omni' if t['planner']=='nav2' else 'ranger-mppi','--upstream-samples','1000','--motion-executor','cpp','--stable-modes','--obstacle-seeded-sampling','--verified-recovery','--map-recovery','--freeze-object-labels','--registered-object-evidence','--unrestricted-task-motion','--policy','grounded','--proposal-selector','vlm','--proposal-interface','task_condition','--spatial-conditions-only','--task-condition-replay',str(target),'--task-condition-default-replay',str(default),'--task-heading-scale-deg','10','--task-clearance-scale-m','.15','--samples','8','--max-sim-time','85','--max-wall-time','420','--real-time-factor','.7','--local-pose-goal','12.8','-2.1','0','--frozen-global-reference',str(ref),'--instruction',instruction,'--ros-domain-id',str(t['domain']),'--ign-partition','frozen_condition_'+name]
 if t['planner']=='hybrid':cmd.append('--hybrid-sequences')
 print('START',name,flush=True)
 with (out/(name+'.log')).open('w') as log:rc=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
 f=path/'episode/result.json';result=json.loads(f.read_text()) if f.exists() else {'status':'NO_TERMINAL_RECORD'}
 with lock:
  results.append({'trial':t,'return_code':rc,'result':result});(out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
 print('DONE',name,result['status'],flush=True)
with ThreadPoolExecutor(max_workers=2) as pool:
 for f in as_completed([pool.submit(run,t) for t in trials]):f.result()
