"""Prediction-only ideal-steering ablation, matched to the public-pair hybrid trials."""
from pathlib import Path
import hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];base=ROOT/'artifacts/offset_pose_public_pair_20261029';out=base/'steering_ablation_v1';ref=out/'reference.json'
files=[ROOT/'src/rangermini_dynamic_semantic/scripts/run_grounded_delivery.py',ROOT/'src/rangermini_dynamic_semantic/rangermini_dynamic_semantic/observation_mppi.py',ROOT/'src/rangermini_dynamic_semantic/rangermini_dynamic_semantic/hybrid_sequence_mppi.py',ROOT/'integration/mppi_native_core/core.cpp',ROOT/'integration/mppi_native_core/build/libranger_mppi_native.so',ref,Path(__file__)]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
hashes={str(p.relative_to(ROOT)):digest(p) for p in files}
(out/'protocol.json').write_text(json.dumps({'pair':'matched_v1/hybrid_s{seed}','seeds':[53,37,41],'change':'ideal wheel-state prediction only; same actuator execution, sequence search, cost, seed, goal, frozen map reference','source_sha256':hashes},indent=2)+'\n')
for i,seed in enumerate((53,37,41)):
 if any(digest(ROOT/k)!=v for k,v in hashes.items()):raise RuntimeError('source changed')
 run=out/f'ideal_s{seed}'
 if run.exists():raise RuntimeError('existing run is preserved')
 (out/'progress.json').write_text(json.dumps({'running':seed,'completed':i,'total':3})+'\n')
 cmd=[sys.executable,str(ROOT/'src/rangermini_dynamic_semantic/scripts/run_grounded_delivery.py'),'--output',str(run),'--scenario','offset_portal','--seed',str(seed),'--planner','ranger-mppi','--local-pose-goal','12.8','0.55','1.5707963267948966','--policy','predictive','--proposal-selector','cost','--motion-executor','cpp','--verified-recovery','--stable-modes','--map-recovery','--frozen-global-reference',str(ref),'--hybrid-sequences','--samples','6','--ideal-steering-rollout','--max-sim-time','90','--max-wall-time','300','--real-time-factor','.5','--ros-domain-id',str(230+i),'--ign-partition',f'offset_ideal_20261029_{seed}']
 with (out/f'ideal_s{seed}.log').open('w') as f:rc=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
 p=run/'episode/result.json'
 if not p.exists():raise RuntimeError(f'incomplete run {seed}, rc={rc}; partial data retained')
 d=json.loads(p.read_text());print(seed,d.get('status'),d.get('elapsed_sim_s'),flush=True)
(out/'progress.json').write_text(json.dumps({'running':None,'completed':3,'total':3})+'\n')
