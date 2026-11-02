"""Public Nav2 Omni MPPI reference for the frozen fragile-trolley task."""
from pathlib import Path
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];base=ROOT/'artifacts/task_attention_service_20261030/pair_v4';out=ROOT/'artifacts/task_attention_service_20261030/nav2_reference_v1';out.mkdir(parents=True,exist_ok=False)
instruction='帮我把这份报告送到619门口。路上那辆推车放着易碎样品，麻烦经过时离它远一些。'
command=[sys.executable,str(ROOT/'src/rangermini_dynamic_semantic/scripts/run_grounded_delivery.py'),'--output',str(out/'run'),'--scenario','heading_passing','--seed','67','--planner','nav2-omni','--upstream-samples','1000','--motion-executor','cpp','--stable-modes','--obstacle-seeded-sampling','--verified-recovery','--map-recovery','--freeze-object-labels','--registered-object-evidence','--unrestricted-task-motion','--policy','grounded','--proposal-selector','vlm','--proposal-interface','task_condition','--spatial-conditions-only','--task-heading-scale-deg','10','--task-clearance-scale-m','.15','--samples','8','--max-sim-time','85','--max-wall-time','420','--real-time-factor','.7','--local-pose-goal','12.8','-2.1','0','--frozen-global-reference',str(base/'static_reference.json'),'--instruction',instruction,'--ros-domain-id','231','--ign-partition','task_attention_public_reference']
(out/'protocol.json').write_text(json.dumps({'purpose':'Public planner qualitative reference, identical scene, task, seed, execution and frozen static reference as pair v4; vision queries independent','command':command},ensure_ascii=False,indent=2)+'\n')
print('START nav2 reference',flush=True)
with (out/'runtime.log').open('w') as log:rc=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
f=out/'run/episode/result.json';print('DONE nav2 reference',json.loads(f.read_text()).get('status') if f.exists() else 'NO_TERMINAL_RECORD','exit',rc,flush=True)
