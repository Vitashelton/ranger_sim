"""Run one archived-model-planned service mission after the matched batch."""
import hashlib,json,subprocess,sys,time,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
batch=ROOT/'artifacts/hybrid_sequence_20261022/frozen_native_v1'
start=time.monotonic()
while True:
 try:
  progress=json.loads((batch/'progress.json').read_text())
  if progress.get('running') is None and progress.get('complete')==12:break
 except (FileNotFoundError,json.JSONDecodeError):pass
 if time.monotonic()-start>2200:raise RuntimeError('Matched experiment did not complete; no service trial started')
 time.sleep(2)
lock=(ROOT/'artifacts/hybrid_sequence_20261022/campaign.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX)
source=ROOT/'artifacts/long_service_task_20261021/multiroom_visual_v5'
plan=json.loads((source/'long_task_plan.json').read_text())
out=ROOT/'artifacts/hybrid_sequence_20261022/service_static_native_v3'
if out.exists():raise RuntimeError('Service attempt already exists; preserve it')
cmd=[sys.executable,str(ROOT/'src/rangermini_dynamic_semantic/scripts/run_grounded_delivery.py'),'--output',str(out),'--planner','ranger-mppi','--hybrid-sequences','--long-task','--task-plan',str(source/'long_task_plan.json'),'--service-confirmation','simulation','--scenario','multiroom_service_static','--seed','17','--instruction',plan['instruction'],'--policy','predictive','--proposal-selector','cost','--verified-recovery','--map-recovery','--stable-modes','--samples','6','--max-sim-time','600','--max-wall-time','1500','--real-time-factor','.5','--ros-domain-id','188','--ign-partition','hybrid_service_static_native_v3']
print('START archived-model-planned service mission',flush=True)
rc=subprocess.run(cmd,cwd=ROOT).returncode
if out.exists():
 (out/'archived_task_parser_response.json').write_bytes((source/'task_parser_response.json').read_bytes())
 (out/'language_plan_provenance.json').write_text(json.dumps({'plan_kind':'archived real language-model output, not a fresh API query','source_plan':str((source/'long_task_plan.json').relative_to(ROOT)),'source_response':str((source/'task_parser_response.json').relative_to(ROOT)),'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [source/'long_task_plan.json',source/'task_parser_response.json']},'fresh_visual_queries':False,'service_confirmation':'simulation acknowledgement','runner_return_code':rc},indent=2)+'\n')
if not (out/'episode/result.json').exists():raise RuntimeError('Service attempt preserved without terminal result')
subprocess.run([sys.executable,str(ROOT/'tools/summarize_long_service_task.py'),str(out)],check=True,cwd=ROOT)
