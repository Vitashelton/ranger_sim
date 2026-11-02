"""Task preference x navigation context x optimizer, with shared static routes."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import contextlib
import fcntl
import getpass
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src/rangermini_dynamic_semantic/scripts'),
              str(ROOT/'src/rangermini_dynamic_semantic')]
from motion_planning_research_scene import create_world, SPECS
from freeze_static_global_reference import freeze

TASKS={
    'ordinary':'把资料送到619室门外等候点并停稳。途中安全绕开推车和器材，完成配送。',
    'heading_clearance':'把资料送到619室门外等候点并停稳。经过推车和器材时尽量保持接近它们之前的车头朝向，并尽量留出0.8米净间距；通过后继续配送。'}


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[41,53,59])
    p.add_argument('--jobs',type=int,choices=[1,2],default=2)
    p.add_argument('--limit',type=int)
    p.add_argument('--ask-key',action='store_true')
    a=p.parse_args()
    if a.ask_key:os.environ['DEEPSEEK_API_KEY']=getpass.getpass('API credential (not saved): ')
    if not os.environ.get('DEEPSEEK_API_KEY'):raise SystemExit('DEEPSEEK_API_KEY required')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    contexts=['static_detour','heading_passing'];methods=['nav2','fixed','hybrid']
    references={}
    for context in contexts:
        world=out/f'{context}.sdf';reference=out/f'{context}_static_reference.json'
        if not world.exists():
            create_world(world,context)
            with contextlib.redirect_stdout(io.StringIO()):
                freeze(world,SPECS[context]['start'][:2],(12.8,-2.1),
                       SPECS[context]['start'][2],0.,reference)
        references[context]=reference
    source_files=list((ROOT/'src/rangermini_dynamic_semantic/rangermini_dynamic_semantic').glob('*.py'))
    source_files+=list((ROOT/'src/rangermini_dynamic_semantic/scripts').glob('*.py'))
    source_files += [Path(__file__),ROOT/'integration/nav2_mppi_adapter/bridge.cpp',
                    ROOT/'integration/nav2_mppi_adapter/build/libranger_nav2_mppi.so',
                    ROOT/'integration/mppi_native_core/build/libranger_mppi_native.so',
                    ROOT/'src/ranger_sim_motion_cpp/src/motion_adapter.cpp',
                    ROOT/'install/ranger_sim_motion_cpp/lib/ranger_sim_motion_cpp/ranger_motion_adapter_cpp']
    hashes={str(p.relative_to(ROOT)):digest(p) for p in source_files}
    trials=[]
    for seed in a.seeds:
        for context in contexts:
            for condition in TASKS:
                for method in methods:
                    trials.append(dict(seed=seed,context=context,condition=condition,method=method,
                        name=f'{context}_{condition}_{method}_s{seed}',domain=160+len(trials)))
    protocol=dict(kind='Task conditions are experimental inputs; optimizer is a separate comparison factor.',
        contexts=contexts,tasks=TASKS,methods=methods,seeds=a.seeds,trials=trials,
        planned_episodes=len(trials),independent_unit='one closed-loop Gazebo navigation episode',
        references={k:dict(path=str(v.relative_to(ROOT)),sha256=digest(v)) for k,v in references.items()},
        controls=['same scene/start/goal and initial wheel state within each matched condition',
                  'one frozen static-map global route per context; no moving-object bypass in the global route',
                  'same scan/tracking inputs, body geometry, finite-rate Ranger execution and label geometry',
                  'real online spatial VLM requests; no model-provided motion-family command'],
        budgets={'nav2':'1000 samples, official Omni optimizer/critics, lateral sampling retained',
                 'fixed':'101 rollout cap; equal phase boundaries',
                 'hybrid':'101 rollout cap; searched phase boundaries'},
        budget_note='Different optimizer work units; no equal-compute claim.',
        public_condition_adapter='Task heading in reference and nonlethal object-clearance costmap layer; public optimizer/critics unchanged.',
        native_command_mapping='Official Ranger Mini V2 lateral-priority semantics, no nearest-mode chooser',
        motion_executor='50 Hz rclcpp simulation node with shared C++ wheel model',
        public_critics='Official CostCritic for graded task layer; TwirlingCritic 10; goal-angle range 10m',
        task_preference_scales=dict(heading_deg=10.,clearance_m=.25),
        registered_static_equipment_evidence=True,
        interpretation='Do not rank the public controller until reference/native-command and independent geometry checks are reviewed.',
        repetitions='Three paired sampling seeds per task/context/optimizer; context and task coverage are separate factors, not more doorway seeds.',
        source_sha256=hashes)
    file=out/'protocol.json'
    if file.exists():assert json.loads(file.read_text())==protocol,'Frozen source/protocol changed'
    else:file.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    completed=[];pending=[];mutex=threading.Lock();running=set()
    for t in trials:
        run=out/t['name'];result=run/'episode/result.json'
        if result.exists():completed.append(dict(t,result=json.loads(result.read_text())))
        elif run.exists():raise RuntimeError('Partial attempt retained: '+str(run))
        else:pending.append(t)
    if a.limit is not None:pending=pending[:a.limit]
    def progress():
        (out/'progress.json').write_text(json.dumps(dict(running=sorted(running),complete=len(completed),planned=len(trials)),indent=2)+'\n')
        (out/'results.json').write_text(json.dumps(completed,ensure_ascii=False,indent=2)+'\n')
    def execute(t):
        assert all(digest(ROOT/k)==v for k,v in hashes.items()),'Source changed during frozen batch'
        run=out/t['name'];method=t['method']
        cmd=[sys.executable,str(ROOT/'src/rangermini_dynamic_semantic/scripts/run_grounded_delivery.py'),
            '--output',str(run),'--scenario',t['context'],'--seed',str(t['seed']),
            '--planner','nav2-omni' if method=='nav2' else 'ranger-mppi',
            '--stable-modes','--obstacle-seeded-sampling','--verified-recovery','--map-recovery',
            '--motion-executor','cpp',
            '--freeze-object-labels','--registered-object-evidence','--unrestricted-task-motion','--policy','grounded','--proposal-selector','vlm',
            '--proposal-interface','task_condition','--spatial-conditions-only',
            '--task-heading-scale-deg','10','--task-clearance-scale-m','.25',
            '--samples','6','--upstream-samples','1000','--max-sim-time','85',
            '--max-wall-time','360','--real-time-factor','.7',
            '--local-pose-goal','12.8','-2.1','0','--frozen-global-reference',str(references[t['context']]),
            '--instruction',TASKS[t['condition']],'--ros-domain-id',str(t['domain']),
            '--ign-partition',out.name+'_'+t['name']]
        if method!='nav2':cmd+=['--hybrid-sequences']
        if method=='fixed':cmd+=['--hybrid-fixed-timing']
        with mutex:running.add(t['name']);progress()
        print('START',t['name'],flush=True)
        with (out/(t['name']+'.log')).open('w') as log:
            rc=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
        result_file=run/'episode/result.json'
        result=json.loads(result_file.read_text()) if result_file.exists() else dict(status='NO_TERMINAL_RECORD')
        with mutex:
            running.remove(t['name']);completed.append(dict(t,return_code=rc,result=result));progress()
        print('DONE',t['name'],result['status'],flush=True)
    progress()
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        for f in as_completed([pool.submit(execute,t) for t in pending]):f.result()
    print('FINISHED',len(completed),'/',len(trials),'attempts',flush=True)


if __name__=='__main__':main()
