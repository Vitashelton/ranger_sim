#!/usr/bin/env python3
"""Focused repair: VLM localizes the task object; geometry selects the route."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--retry-failed',action='store_true');a=p.parse_args();root=a.root.resolve();here=Path(__file__).resolve().parent
 if not os.environ.get('DEEPSEEK_API_KEY'):raise SystemExit('Export API key first.')
 protocol=root/'grounded_variant_protocol.json'
 if not protocol.exists():protocol.write_text(json.dumps(dict(
  motivation='direct route selector confused a person with an equipment area in south/trolley/seed31; retain that failure',
  method='VLM visible target bounding box -> measured depth grounding -> maximize conservative target gap on dense shared guides; current-scan MPPI remains shared',
  layouts=['north','south'],targets=['person','trolley'],seeds=[17,23,31],
  target_proxy_radii_m=dict(person=.18,trolley=.41),depth_quantile=.20,guide_spacing_m=.05,
  absent_target_policy='stop and fail, no hidden geometry fallback',criterion='same registered target side and actual navigation arrival; independent measured target gap >= .60m',
  scope='12 new actual episodes; no removal/replacement of 36 direct-selector baseline episodes'),ensure_ascii=False,indent=2))
 # Start with the failure condition, then complete all registered cells.
 cells=[(s,t,n) for s in ['south','north'] for t in ['trolley','person'] for n in [31,17,23]]
 for side,target,seed in cells:
  name=f'{side}_{target}_grounded_s{seed}';work=root/'runs'/name;rp=work/'episode/result.json'
  if rp.exists():
   if json.loads(rp.read_text()).get('status')=='ARRIVED':continue
   if not a.retry_failed:raise SystemExit('Retained failed episode; inspect before explicitly retrying.')
   archive=root/'failed_attempts';archive.mkdir(exist_ok=True);work.rename(archive/f'{name}_{time.time_ns()}')
  with (root/f'{name}.log').open('w') as log:
   proc=subprocess.run([sys.executable,str(here/'run_delivery_route_pilot.py'),'--output',str(work),'--selector','grounded','--seed',str(seed),'--person-side',side,'--task-target',target,'--paired-objects'],stdout=log,stderr=subprocess.STDOUT)
  r=json.loads(rp.read_text()) if rp.exists() else {};status=dict(cell=name,status=r.get('status','NO_RESULT'),route=r.get('route_id'),expected='RIGHT' if (side=='north')==(target=='person') else 'LEFT',exit_code=proc.returncode)
  with (root/'grounded_status.jsonl').open('a') as f:f.write(json.dumps(status)+'\n')
  print(json.dumps(status),flush=True)
  if proc.returncode or status['status']!='ARRIVED':raise SystemExit('Grounded episode failed; evidence retained.')
 env=dict(os.environ,PYTHONNOUSERSITE='1')
 subprocess.run(['/usr/bin/python3',str(here/'summarize_delivery_grounding.py'),'--root',str(root),'--grounded'],env=env,check=True)
 subprocess.run([sys.executable,str(here/'build_grounding_meeting.py'),'--root',str(root),'--grounded'],check=True)
 subprocess.run([sys.executable,str(here/'write_grounded_manuscript.py'),'--root',str(root)],check=True)
 print('COMPLETE grounded motion episodes=12; figures, actual videos, meeting PPTX and manuscript.',flush=True)
if __name__=='__main__':main()
