#!/usr/bin/env python3
"""Compile one full instruction before simulation, with an inspectable archive."""
import argparse,json,sys,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph
from rangermini_dynamic_semantic.long_service_task import compile_task,validate_task,LongServiceTaskHarness,prepare_service_layout
p=argparse.ArgumentParser();p.add_argument('--world',required=True);p.add_argument('--output',required=True)
p.add_argument('--prepare-service-layout',action='store_true');p.add_argument('--instruction',required=True);p.add_argument('--plan-input',help='Explicit plan, not a model call')
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
world=Path(a.world)
if a.prepare_service_layout:
    shutil.copy2(world,out/'world.sdf');world=out/'world.sdf'
    (out/'layout_changes.json').write_text(json.dumps(prepare_service_layout(world),indent=2))
g=SpatialGraph(world)
plan=(dict(validate_task(json.loads(Path(a.plan_input).read_text()),g), source='supplied_plan')
      if a.plan_input else compile_task(a.instruction,g,out/'parser_response.json'))
h=LongServiceTaskHarness(g,a.instruction,plan)
(out/'task_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
(out/'preflight.json').write_text(json.dumps({'task_id':h.task_id,'service_poses':h.service_poses,
 'confirmation_topic':'/harness/service_confirmation','status':'compiled_not_executed'},indent=2))
print(json.dumps({'task_id':h.task_id,'rooms':[s['room'] for s in plan['stops']],
                  'source':plan['source'],'service_poses':h.service_poses},ensure_ascii=False,indent=2))
