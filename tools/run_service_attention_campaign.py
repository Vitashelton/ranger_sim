"""Replicate frozen task-attention settings with new paired sampling seeds."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import contextlib,hashlib,io,json,subprocess,sys,threading
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src/rangermini_dynamic_semantic/scripts'),str(ROOT/'src/rangermini_dynamic_semantic'),str(ROOT/'tools')]
from motion_planning_research_scene import create_world
from freeze_static_global_reference import freeze
out=ROOT/'artifacts/task_attention_service_20261030/campaign_v1';out.mkdir(parents=True,exist_ok=False)
world=out/'heading_passing.sdf';create_world(world,'heading_passing');ref=out/'static_reference.json'
with contextlib.redirect_stdout(io.StringIO()):freeze(world,(3.,-2.1),(12.8,-2.1),0.,0.,ref)
instructions={'ordinary':'帮我把这份报告送到619门口。','fragile':'帮我把这份报告送到619门口。路上那辆推车放着易碎样品，麻烦经过时离它远一些。'}
files=[Path(__file__),ROOT/'src/rangermini_dynamic_semantic/scripts/run_grounded_delivery.py',ROOT/'integration/mppi_native_core/build/libranger_mppi_native.so',ROOT/'install/ranger_sim_motion_cpp/lib/ranger_sim_motion_cpp/ranger_motion_adapter_cpp',world,ref]
files += list((ROOT/'src/rangermini_dynamic_semantic/rangermini_dynamic_semantic').glob('*.py'))
trials=[{'seed':seed,'task':name,'domain':225+i} for i,(seed,name) in enumerate((seed,name) for seed in (71,73) for name in ('ordinary','fragile'))]
(out/'protocol.json').write_text(json.dumps({'purpose':'Independent repetitions of fixed task-attention corridor pair v4','instructions':instructions,'trials':trials,'task_clearance_scale_m':.15,'same_initialization_within_pair':True,'common_reference':'static-map A* excluding dynamic actors','source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},ensure_ascii=False,indent=2)+'\n')
lock=threading.Lock();results=[]
def execute(t):
 name=f"{t['task']}_s{t['seed']}";run=out/name
 cmd=[sys.executable,str(ROOT/'src/rangermini_dynamic_semantic/scripts/run_grounded_delivery.py'),'--output',str(run),'--scenario','heading_passing','--seed',str(t['seed']),'--planner','ranger-mppi','--motion-executor','cpp','--stable-modes','--hybrid-sequences','--obstacle-seeded-sampling','--verified-recovery','--map-recovery','--freeze-object-labels','--registered-object-evidence','--unrestricted-task-motion','--policy','grounded','--proposal-selector','vlm','--proposal-interface','task_condition','--spatial-conditions-only','--task-heading-scale-deg','10','--task-clearance-scale-m','.15','--samples','8','--max-sim-time','85','--max-wall-time','420','--real-time-factor','.7','--local-pose-goal','12.8','-2.1','0','--frozen-global-reference',str(ref),'--instruction',instructions[t['task']],'--ros-domain-id',str(t['domain']),'--ign-partition','task_attention_campaign_'+name]
 print('START',name,flush=True)
 with (out/(name+'.log')).open('w') as log:rc=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
 f=run/'episode/result.json';result=json.loads(f.read_text()) if f.exists() else {'status':'NO_TERMINAL_RECORD'}
 with lock:
  results.append({'trial':t,'return_code':rc,'result':result});(out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
 print('DONE',name,result['status'],flush=True)
with ThreadPoolExecutor(max_workers=2) as pool:
 for f in as_completed([pool.submit(execute,t) for t in trials]):f.result()
