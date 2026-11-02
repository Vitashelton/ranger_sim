"""Same geometry, counterbalanced appearance and object-directed conversation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import getpass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src/rangermini_dynamic_semantic/scripts'),
              str(ROOT/'src/rangermini_dynamic_semantic')]
from motion_planning_research_scene import create_world
from freeze_static_global_reference import freeze


def appearance_free_tree(path):
    root=ET.parse(path).getroot()
    for visual in root.iter('visual'):
        for material in list(visual.findall('material')):
            visual.remove(material)
    return ET.tostring(root)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ask-key',action='store_true')
    p.add_argument('--limit',type=int)
    a=p.parse_args()
    if a.ask_key:os.environ['DEEPSEEK_API_KEY']=getpass.getpass('API credential (not saved): ')
    if not os.environ.get('DEEPSEEK_API_KEY'):raise SystemExit('Visual credential required')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    scenes=['object_reference_a','object_reference_b']
    for scene in scenes:
        create_world(out/f'{scene}.sdf',scene)
    assert appearance_free_tree(out/f'{scenes[0]}.sdf')==appearance_free_tree(out/f'{scenes[1]}.sdf')
    reference=out/'common_static_reference.json'
    if not reference.exists():freeze(out/f'{scenes[0]}.sdf',(3.,-2.1),(12.8,-2.1),0.,0.,reference)
    phrases={
        'red':'把资料送到619门口吧。路过装红箱子的那辆推车时，给它多留点地方；其他地方照常走就好。',
        'blue':'把资料送到619门口吧。路过装蓝箱子的那辆推车时，给它多留点地方；其他地方照常走就好。',
        'geometry':'把资料送到619门口吧，路上正常通行。'}
    scripts=list((ROOT/'src/rangermini_dynamic_semantic/rangermini_dynamic_semantic').glob('*.py'))
    scripts+=list((ROOT/'src/rangermini_dynamic_semantic/scripts').glob('*.py'))
    hashes={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in scripts}
    trials=[dict(scene=s,request=q,name=f'{s}_{q}_s41',domain=200+i)
            for i,(s,q) in enumerate((s,q) for s in scenes for q in ('red','blue','geometry'))]
    protocol=dict(kind='Object-reference pilot, not a generalization benchmark',seed=41,
        independent_episodes=len(trials),trials=trials,phrases=phrases,
        controls=['Identical collision and visual geometry; only payload materials swapped',
                  'Same object IDs, positions, route, destination and hybrid MPPI settings',
                  'No named chassis mode, pass side or metric clearance supplied in conversation',
                  'Geometry control is one separate episode per appearance, not duplicated across requests'],
        geometry_equivalence_verified=True,reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        source_sha256=hashes)
    file=out/'protocol.json'
    if file.exists():assert json.loads(file.read_text())==protocol,'Frozen pilot changed'
    else:file.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    pending=[t for t in trials if not (out/t['name']/'episode/result.json').exists()]
    if a.limit is not None:pending=pending[:a.limit]
    lock=threading.Lock();results=[dict(t,result=json.loads((out/t['name']/'episode/result.json').read_text())) for t in trials if (out/t['name']/'episode/result.json').exists()]
    def execute(t):
        assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in hashes.items())
        run=out/t['name']
        if run.exists():raise RuntimeError('Interrupted attempt retained: '+str(run))
        cmd=[sys.executable,str(ROOT/'src/rangermini_dynamic_semantic/scripts/run_grounded_delivery.py'),
            '--output',str(run),'--scenario',t['scene'],'--seed','41','--planner','ranger-mppi',
            '--motion-executor','cpp','--stable-modes','--hybrid-sequences',
            '--obstacle-seeded-sampling','--verified-recovery','--map-recovery',
            '--freeze-object-labels','--registered-object-evidence','--unrestricted-task-motion',
            '--task-heading-scale-deg','10','--task-clearance-scale-m','.25',
            '--samples','6','--max-sim-time','80','--max-wall-time','300','--real-time-factor','.7',
            '--local-pose-goal','12.8','-2.1','0','--frozen-global-reference',str(reference),
            '--instruction',phrases[t['request']],'--ros-domain-id',str(t['domain']),
            '--ign-partition',out.name+'_'+t['name']]
        if t['request']=='geometry':cmd+=['--policy','predictive','--proposal-selector','cost']
        else:cmd+=['--policy','grounded','--proposal-selector','vlm',
                   '--proposal-interface','task_condition','--spatial-conditions-only']
        print('START',t['name'],flush=True)
        with (out/(t['name']+'.log')).open('w') as log:
            rc=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
        f=run/'episode/result.json';result=json.loads(f.read_text()) if f.exists() else {'status':'NO_TERMINAL_RECORD'}
        with lock:
            results.append(dict(t,return_code=rc,result=result))
            (out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
        print('DONE',t['name'],result['status'],flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(execute,t) for t in pending]):future.result()


if __name__=='__main__':main()
