#!/usr/bin/env python3
"""Predeclared crossed task/image experiment; preserve every episode and call."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rangermini_dynamic_semantic.visual_behavior_selector import select_visual_route
from run_delivery_route_pilot import TASK_INSTRUCTIONS

def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--controls-only',action='store_true');a=p.parse_args()
 root=a.output.resolve();root.mkdir(parents=True,exist_ok=a.resume)
 if not os.environ.get('DEEPSEEK_API_KEY'):raise SystemExit('Export DEEPSEEK_API_KEY before running.')
 here=Path(__file__).resolve().parent
 cells=[(side,target,method,seed) for side in ['north','south'] for target in ['person','trolley'] for seed in [17,23,31] for method in ['geometric','fixed','vlm']]
 manifest=dict(problem='delivery route preferences depend on the instructed object, not simply the nearest person',
 instructions=TASK_INSTRUCTIONS,layouts=['person north + trolley south','person south + trolley north'],seeds=[17,23,31],
 methods=['geometric shared MPPI objective','fixed RGBD person-clearance policy (instruction independent)','VLM image + instruction, geometry-free candidate packet'],
 expected_routes={f'{side}_{target}':('RIGHT' if (side=='north')==(target=='person') else 'LEFT') for side in ['north','south'] for target in ['person','trolley']},
 criterion='correct route AND navigation arrival; additionally evaluate target footprint clearance >= 0.60m',
 interpretation='tests flexibility against a fixed person policy, not superiority over task-aware rules; no novelty conclusion from selection alone',
 controls='same canonical RGB/candidate packet per layout; full / no image / neutral instruction; 3 calls per task, condition and layout',
 created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
 source_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [here/'run_delivery_route_pilot.py',Path(__file__)]})
 manifest_path=root/'preregistered_protocol.json'
 if not manifest_path.exists():write(manifest_path,manifest)
 for side,target,method,seed in ([] if a.controls_only else cells):
  name=f'{side}_{target}_{method}_s{seed}';work=root/'runs'/name;rp=work/'episode/result.json'
  if rp.exists() and json.loads(rp.read_text()).get('status')=='ARRIVED':continue
  if work.exists():
   archive=root/'failed_attempts';archive.mkdir(exist_ok=True);work.rename(archive/f'{name}_{time.time_ns()}')
  with (root/f'{name}.log').open('w') as log:
   proc=subprocess.run([sys.executable,str(here/'run_delivery_route_pilot.py'),'--output',str(work),'--selector',method,'--seed',str(seed),'--person-side',side,'--task-target',target,'--paired-objects'],stdout=log,stderr=subprocess.STDOUT)
  result=json.loads(rp.read_text()) if rp.exists() else {}
  status=dict(cell=name,status=result.get('status','NO_RESULT'),route=result.get('route_id'),expected=manifest['expected_routes'][f'{side}_{target}'],exit_code=proc.returncode)
  with (root/'batch_status.jsonl').open('a') as log:log.write(json.dumps(status)+'\n')
  print(json.dumps(status),flush=True)
  if proc.returncode or status['status']!='ARRIVED':raise SystemExit('Episode failed; retained all evidence. Repair then --resume.')
 controls=root/'controls';controls.mkdir(exist_ok=True)
 packet=[dict(candidate_id=k,feasible=True,color=c) for k,c in [('LEFT','green'),('RIGHT','orange')]]
 neutral='把车载资料送到前面的实验室，选择一个可行绕行路线。'
 for side in ['north','south']:
  source=root/'runs'/f'{side}_person_geometric_s17'/'episode/selection_overlay.jpg';im=source.read_bytes()
  for target in ['person','trolley']:
   for condition in ['full','no_image','neutral_instruction']:
    for repeat in range(3):
     path=controls/f'{side}_{target}_{condition}_{repeat}.json'
     if path.exists():continue
     instruction=neutral if condition=='neutral_instruction' else TASK_INSTRUCTIONS[target]
     inp=dict(layout=side,task_target=target,condition=condition,repeat=repeat,instruction=instruction,candidates=packet,
       image_source=str(source.relative_to(root)),image_sha256=hashlib.sha256(im).hexdigest(),expected_route=manifest['expected_routes'][f'{side}_{target}'])
     write(path.with_suffix('.input.json'),inp)
     try:result=select_visual_route(None if condition=='no_image' else im,instruction,packet,timeout_s=90)
     except Exception as exc:
      write(path.with_name(path.stem+f'.{time.time_ns()}.error.json'),dict(error=str(exc)));raise
     write(path,dict(input=inp,response=result));print('CONTROL '+path.name+' '+result['decision']['candidate_id'],flush=True)
 if a.controls_only:
  print('CONTROLS_COMPLETE actual_calls=36',flush=True);return
 env=dict(os.environ,PYTHONNOUSERSITE='1')
 subprocess.run(['/usr/bin/python3',str(here/'summarize_delivery_grounding.py'),'--root',str(root)],env=env,check=True)
 subprocess.run([sys.executable,str(here/'build_grounding_meeting.py'),'--root',str(root)],check=True)
 subprocess.run([sys.executable,str(here/'update_grounding_manuscript.py'),'--root',str(root)],check=True)
 print('COMPLETE physical=36 controlled_calls=36; report, figures, videos, PPTX and manuscript source.',flush=True)
if __name__=='__main__':main()
