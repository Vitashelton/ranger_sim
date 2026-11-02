"""Record one separate retry of a transport-failed matched trial."""
from pathlib import Path
import hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];base=ROOT/'artifacts/task_attention_service_20261030/campaign_v1';protocol=json.load(open(base/'protocol.json'))
for path,digest in protocol['source_sha256'].items():
 if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:raise RuntimeError('Frozen campaign source changed: '+path)
run=base/'fragile_s71_retry1'
if run.exists():raise RuntimeError('Attempt already exists')
cmd=[sys.executable,str(ROOT/'src/rangermini_dynamic_semantic/scripts/run_grounded_delivery.py'),'--output',str(run),'--scenario','heading_passing','--seed','71','--planner','ranger-mppi','--motion-executor','cpp','--stable-modes','--hybrid-sequences','--obstacle-seeded-sampling','--verified-recovery','--map-recovery','--freeze-object-labels','--registered-object-evidence','--unrestricted-task-motion','--policy','grounded','--proposal-selector','vlm','--proposal-interface','task_condition','--spatial-conditions-only','--task-heading-scale-deg','10','--task-clearance-scale-m','.15','--samples','8','--max-sim-time','85','--max-wall-time','420','--real-time-factor','.7','--local-pose-goal','12.8','-2.1','0','--frozen-global-reference',str(base/'static_reference.json'),'--instruction',protocol['instructions']['fragile'],'--ros-domain-id','232','--ign-partition','task_attention_fragile_s71_retry1']
print('START fragile_s71 retry',flush=True)
with (base/'fragile_s71_retry1.log').open('w') as log:rc=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
f=run/'episode/result.json';print('DONE fragile_s71 retry',json.load(open(f))['status'] if f.exists() else 'NO_TERMINAL_RECORD','exit',rc,flush=True)
