"""Two online task-conditioned corridor trials under a frozen common route."""
from pathlib import Path
import hashlib, json, os, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src/rangermini_dynamic_semantic/scripts'),str(ROOT/'src/rangermini_dynamic_semantic'),str(ROOT/'tools')]
from motion_planning_research_scene import create_world
from freeze_static_global_reference import freeze
out=ROOT/'artifacts/task_attention_service_20261030/pair_v2'
out.mkdir(parents=True,exist_ok=False)
world=out/'heading_passing.sdf'
create_world(world,'heading_passing')
ref=out/'static_reference.json'
freeze(world,(3.,-2.1),(12.8,-2.1),0.,0.,ref)
instructions={'ordinary':'帮我把这份报告送到619门口。','fragile':'帮我把这份报告送到619门口。路上那辆推车放着易碎样品，经过时当心点。'}
files=[Path(__file__),ROOT/'src/rangermini_dynamic_semantic/scripts/run_grounded_delivery.py',ROOT/'integration/mppi_native_core/build/libranger_mppi_native.so',ROOT/'install/ranger_sim_motion_cpp/lib/ranger_sim_motion_cpp/ranger_motion_adapter_cpp',world,ref]
meta={'purpose':'Matched online task-conditioned trolley passage from natural-language service requests','seed':67,'scenario':'heading_passing','start':[3.,-2.1,0.],'goal':[12.8,-2.1,0.],'instructions':instructions,'shared':'same static-map reference, dynamic obstacle scenario, C++ chassis, planner settings and random initialization','sample_population_per_family':8,'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
(out/'protocol.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
for i,(name,instruction) in enumerate(instructions.items()):
 run=out/name
 command=[sys.executable,str(ROOT/'src/rangermini_dynamic_semantic/scripts/run_grounded_delivery.py'),'--output',str(run),'--scenario','heading_passing','--seed','67','--planner','ranger-mppi','--motion-executor','cpp','--stable-modes','--hybrid-sequences','--obstacle-seeded-sampling','--verified-recovery','--map-recovery','--freeze-object-labels','--registered-object-evidence','--unrestricted-task-motion','--policy','grounded','--proposal-selector','vlm','--proposal-interface','task_condition','--spatial-conditions-only','--task-heading-scale-deg','10','--task-clearance-scale-m','.25','--samples','8','--max-sim-time','85','--max-wall-time','360','--real-time-factor','.7','--local-pose-goal','12.8','-2.1','0','--frozen-global-reference',str(ref),'--instruction',instruction,'--ros-domain-id',str(220+i),'--ign-partition','task_attention_20261030_'+name]
 print('START '+name,flush=True)
 with (out/(name+'.log')).open('w') as log:rc=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
 result=run/'episode/result.json'
 print('DONE',name,'exit',rc,'status',json.loads(result.read_text()).get('status') if result.exists() else 'NO_TERMINAL_RECORD',flush=True)
 if not result.exists():break
