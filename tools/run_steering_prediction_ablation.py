"""Paired Gazebo ablation: change only the steering prediction assumption."""
from pathlib import Path
import hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/steering_prediction_ablation_20261028'
TRIALS=[(seed,mode) for seed in (37,53) for mode in ('finite','instant')]
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 sources=[ROOT/'src/rangermini_dynamic_semantic/scripts/run_grounded_delivery.py',ROOT/'src/rangermini_dynamic_semantic/rangermini_dynamic_semantic/hybrid_sequence_mppi.py',ROOT/'src/rangermini_dynamic_semantic/rangermini_dynamic_semantic/observation_mppi.py',ROOT/'integration/mppi_native_core/core.cpp',ROOT/'integration/mppi_native_core/build/libranger_mppi_native.so']
 protocol={'purpose':'Planning-side steering prediction ablation, not a public planner baseline','scenario':'heading_passing','seeds':[37,53],'methods':['finite','instant'],'common':'Same hybrid MPPI, scene, goal, task input, execution adapter and limits. Only --ideal-steering-rollout differs. No live VLM.', 'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
 f=OUT/'protocol.json'
 if f.exists():
  if json.loads(f.read_text())!=protocol:raise RuntimeError('Frozen source or protocol changed')
 else:f.write_text(json.dumps(protocol,indent=2)+'\n')
 for i,(seed,mode) in enumerate(TRIALS):
  name=f'{mode}_s{seed}';run=OUT/name;result=run/'episode/result.json'
  if result.exists():continue
  if run.exists():raise RuntimeError('Partial run preserved: '+name)
  cmd=[sys.executable,str(ROOT/'src/rangermini_dynamic_semantic/scripts/run_grounded_delivery.py'),'--output',str(run),'--scenario','heading_passing','--seed',str(seed),'--planner','ranger-mppi','--policy','predictive','--proposal-selector','cost','--motion-executor','cpp','--stable-modes','--hybrid-sequences','--verified-recovery','--map-recovery','--instruction','把资料送到619门口。','--samples','6','--max-sim-time','100','--max-wall-time','300','--real-time-factor','.7','--ros-domain-id',str(211+i),'--ign-partition','steering_ablation_'+name]
  if mode=='instant':cmd+=['--ideal-steering-rollout']
  (OUT/'progress.json').write_text(json.dumps({'running':name,'completed':i,'total':len(TRIALS)},indent=2)+'\n')
  print('START',name,flush=True)
  with (OUT/(name+'.log')).open('w') as log:rc=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
  if not result.exists():
   (OUT/'progress.json').write_text(json.dumps({'failed':name,'returncode':rc,'completed':i,'total':len(TRIALS)},indent=2)+'\n');raise RuntimeError('No result; attempt retained')
  print('DONE',name,json.loads(result.read_text()).get('status'),flush=True)
 (OUT/'progress.json').write_text(json.dumps({'running':None,'completed':len(TRIALS),'total':len(TRIALS)},indent=2)+'\n')
if __name__=='__main__':main()
