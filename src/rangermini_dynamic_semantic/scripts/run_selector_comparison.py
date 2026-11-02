#!/usr/bin/env python3
"""Serial, matched selection-only Gazebo comparison; never pool old revisions."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
RUNNER=Path(__file__).with_name('run_grounded_delivery.py')
from run_research_motion_matrix import INSTRUCTIONS
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--seeds',nargs='+',type=int,default=[37,41,53]);p.add_argument('--scenarios',nargs='+',default=['headon','heading_passing','encounter_entry']);a=p.parse_args()
 if not os.getenv('DEEPSEEK_API_KEY'):raise SystemExit('Vision API credential is missing')
 out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
 sources=list((ROOT/'src/rangermini_dynamic_semantic/rangermini_dynamic_semantic').glob('*.py'))+[RUNNER,Path(__file__),RUNNER.with_name('run_research_motion_matrix.py'),RUNNER.with_name('audit_grounded_delivery.py')]
 hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
 spec={'kind':'matched selector evaluation, not external planner benchmark','seeds':a.seeds,'scenarios':a.scenarios,'selectors':['cost','vlm'],'policy':'grounded','shared_flags':['--stable-modes','--verified-recovery','--map-recovery','--freeze-object-labels'],'source_sha256':hashes,'scenario_instructions':{k:INSTRUCTIONS[k] for k in a.scenarios if k in INSTRUCTIONS},'note':'Different HTTP delay is included in deployed selector effect; not a delay-matched pure reasoning comparison.'}
 specpath=out/'comparison_spec.json'
 if specpath.exists() and json.loads(specpath.read_text())!=spec:raise SystemExit('Refusing mixed source/configuration resume')
 specpath.write_text(json.dumps(spec,indent=2));results=[]
 for scenario in a.scenarios:
  for seed in a.seeds:
   # Alternate order across seeds; no simultaneous Gazebo/MPPI host load.
   for selector in (['cost','vlm'] if a.seeds.index(seed)%2==0 else ['vlm','cost']):
    if any(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()!=h for f,h in hashes.items()):raise SystemExit('Source changed during batch; refusing to continue')
    name=f'{scenario}_{selector}_s{seed}';run=out/name
    if not run.exists():
     command=[sys.executable,str(RUNNER),'--output',str(run),'--scenario',scenario,'--policy','grounded','--seed',str(seed),'--proposal-selector',selector,'--freeze-object-labels','--stable-modes','--verified-recovery','--map-recovery','--max-sim-time','110','--max-wall-time','420','--ros-domain-id','187','--ign-partition','ranger_matched_selectors']
     if scenario in INSTRUCTIONS:command.extend(['--instruction',INSTRUCTIONS[scenario]])
     with (out/f'{name}.log').open('w') as log:
      proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
     if proc.returncode:raise SystemExit(f'Run failed; preserved attempt: {name}')
    resultfile=run/'episode/result.json'
    if not resultfile.exists():raise SystemExit(f'Incomplete preserved attempt: {name}')
    r=json.loads(resultfile.read_text());record={'run':name,'selector':selector,'seed':seed,'scenario':scenario,**r}
    if r['status']!='VLM_REQUEST_FAILED':
     subprocess.run([sys.executable,str(RUNNER.with_name('audit_grounded_delivery.py')),str(run)],check=True,stdout=subprocess.DEVNULL)
    audit=run/'interaction_audit.json'
    if audit.exists():record['interaction_audit']=json.loads(audit.read_text())
    results.append(record);(out/'comparison_results.json').write_text(json.dumps(results,indent=2))
    print(json.dumps({'completed':len(results),'run':name,'status':r['status'],'time_s':r['elapsed_sim_s'],'queries':r.get('vlm_api_requests'),'interaction':record.get('interaction_audit',{}).get('interaction_success')},ensure_ascii=False),flush=True)
    if r['status']=='VLM_REQUEST_FAILED':raise SystemExit('Initial visual request failed; batch stopped, no success imputed')
 print('Matched batch complete',flush=True)
if __name__=='__main__':main()
