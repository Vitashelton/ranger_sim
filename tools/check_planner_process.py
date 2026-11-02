#!/usr/bin/env python3
"""Focused equivalence check: same seed, inputs and repeated persistent calls."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/rangermini_dynamic_semantic'))
from rangermini_dynamic_semantic.command_semantics_mppi import MPPIConfig
from rangermini_dynamic_semantic.stable_interaction_mppi import StableInteractionMPPI
from rangermini_dynamic_semantic.planner_process import PlannerProcess
from rangermini_dynamic_semantic.grounded_navigation_harness import Track


def canonical(value):
    if isinstance(value,dict):
        return {k:canonical(v) for k,v in value.items() if 'time_ms' not in k and 'wall_ms' not in k}
    if isinstance(value,(list,tuple)):
        return [canonical(v) for v in value]
    if hasattr(value,'tolist'):return value.tolist()
    return value


def main():
    cfg=MPPIConfig(samples_per_behavior=12,horizon=40,dt=.12,max_speed=.35,random_seed=37,allow_reverse=False)
    local=StableInteractionMPPI(cfg,prediction=True,static_boxes=[])
    worker=PlannerProcess(StableInteractionMPPI(cfg,prediction=True,static_boxes=[]))
    checks=[]
    try:
        for i in range(3):
            kwargs=dict(current_mode='PARALLEL',current_steering_angles=[0.,0.,0.,0.],goal_yaw=0.)
            args=((0.,0.,0.),(2.,.2),([8.]*90,-1.57,.035),[Track('audit-object',(2.5,1.2),(-.1,0.),.15,i*.12,0.)],i*.12)
            a=local.bank(*args,**kwargs);b=worker.bank(*args,**kwargs)
            assert canonical(a)==canonical(b), 'worker changed planner output'
            selected=next(c['plan'] for c in a if c['feasible'])
            action=tuple(selected['action'])
            local.note_execution(selected,i*.12,action);worker.note_execution(selected,i*.12,action)
            assert local.rng.getstate()==worker.rng.getstate()
            assert local.committed_group==worker.committed_group
            checks.append(dict(iteration=i,identical=True,candidates=len(a)))
    finally:worker.close()
    out=Path('artifacts/executor_audit_20261021/planner_equivalence.json')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(checks,indent=2));print(out.read_text())

if __name__=='__main__':main()
