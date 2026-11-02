"""Three physical scenes, official Omni MPPI plus one internal steering ablation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/rangermini_dynamic_semantic/scripts'))
from audit_grounded_delivery import audit

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
 p.add_argument('--seeds',type=int,nargs='+',default=[17,37,41])
 p.add_argument('--scenarios',nargs='+',default=['static_detour','heading_constrained_passage','offset_portal'])
 p.add_argument('--methods',nargs='+',default=['nav2-omni','multimode','no-steering-prediction'])
 a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
 files=list((ROOT/'src/rangermini_dynamic_semantic/rangermini_dynamic_semantic').glob('*.py'))
 files+=list((ROOT/'src/rangermini_dynamic_semantic/scripts').glob('*.py'))
 files+=[ROOT/'integration/nav2_mppi_adapter/bridge.cpp',ROOT/'integration/nav2_mppi_adapter/build/libranger_nav2_mppi.so',Path(__file__).resolve()]
 hashes={str(f.relative_to(ROOT)):digest(f) for f in files}
 protocol={'scenarios':a.scenarios,'methods':a.methods,'seeds':a.seeds,'source_sha256':hashes,
  'upstream':json.loads((ROOT/'integration/nav2_mppi_adapter/provenance.json').read_text()),
  'common':{'execution':'finite-rate four-wheel kinematic adapter','heading_requirement':'0 +/-18 deg during APPROACH and x<=10 m in heading_constrained_passage; released before door entry','local_reference':'shared FootprintRouteGuide and task harness','arrival':'same event verification and stopping thresholds','vlm_calls':False},
  'budgets':{'nav2_omni_samples':1000,'multimode_samples':25,'horizon':40,'model_dt':.12,'note':'Practical configurations, not equal-sample or equal-compute comparison'},
  'ablation':'Only rollout steering assumption changes; execution, reference, task, obstacles and family sampling unchanged',
  'intended_runs':len(a.seeds)*len(a.scenarios)*len(a.methods)}
 frozen=out/'protocol.json'
 if frozen.exists() and json.loads(frozen.read_text())!=protocol:raise RuntimeError('Source/protocol changed; use a separate batch')
 frozen.write_text(json.dumps(protocol,indent=2));results=[]
 for scenario in a.scenarios:
  for seed in a.seeds:
   order=a.methods[seed%len(a.methods):]+a.methods[:seed%len(a.methods)]
   for method in order:
    if any(digest(ROOT/f)!=h for f,h in hashes.items()):raise RuntimeError('Frozen source changed')
    name=f'{scenario}_{method}_s{seed}';run=out/name
    if (run/'episode/result.json').exists():
     result=audit(run);results.append(dict(result,method=method));continue
    if run.exists():raise RuntimeError('Partial attempt preserved: '+str(run))
    args=[sys.executable,str(ROOT/'src/rangermini_dynamic_semantic/scripts/run_grounded_delivery.py'),
      '--output',str(run),'--planner','nav2-omni' if method=='nav2-omni' else 'ranger-mppi',
      '--scenario',scenario,'--seed',str(seed),'--policy','predictive','--proposal-selector','cost',
      '--verified-recovery','--map-recovery','--stable-modes','--samples','6','--upstream-samples','1000',
      '--max-sim-time','90','--max-wall-time','240','--real-time-factor','1',
      '--ros-domain-id','176','--ign-partition','nav2_matched_'+name.replace('-','_')]
    if scenario in ('heading_passing','heading_constrained_passage'):args+=['--retain-corridor-heading']
    if method=='no-steering-prediction':args+=['--ideal-steering-rollout']
    print('START',name,flush=True)
    (out/'progress.json').write_text(json.dumps({'running':name,'complete':len(results),'intended':protocol['intended_runs']},indent=2))
    with (out/'campaign.log').open('a') as stream:
     rc=subprocess.run(args,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT).returncode
    if not (run/'episode/result.json').exists():
     (out/'interruption.json').write_text(json.dumps({'run':name,'return_code':rc},indent=2))
     raise RuntimeError('No terminal record; preserve and diagnose '+name)
    result=audit(run);results.append(dict(result,method=method))
    (out/'results.json').write_text(json.dumps(results,indent=2))
    print('DONE',name,result['status'],'proxy_clear',result['interaction_success'],flush=True)
 (out/'progress.json').write_text(json.dumps({'running':None,'complete':len(results),'intended':protocol['intended_runs']},indent=2))
 print('COMPLETE',len(results),flush=True)
if __name__=='__main__':main()
