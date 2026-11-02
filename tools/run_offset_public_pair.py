"""Matched Ranger offset-pose evaluation with a frozen static global route."""
from pathlib import Path
import hashlib,json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/offset_pose_public_pair_20261029/matched_v1'
ref=OUT/'reference.json'
files=[ROOT/'src/rangermini_dynamic_semantic/rangermini_dynamic_semantic/nav2_omni_mppi.py',ROOT/'src/rangermini_dynamic_semantic/scripts/run_grounded_delivery.py',ROOT/'integration/mppi_native_core/core.cpp',ROOT/'integration/mppi_native_core/build/libranger_mppi_native.so',ROOT/'integration/nav2_mppi_adapter/bridge.cpp',ROOT/'integration/nav2_mppi_adapter/build/libranger_nav2_mppi.so',ref,Path(__file__)]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
hashes={str(p.relative_to(ROOT)):digest(p) for p in files}
trials=[(seed,m) for seed in (53,37,41) for m in ('nav2','hybrid')]
protocol={'goal':[12.8,.55,1.5707963267948966],'scenario':'offset_portal','trials':trials,'common':'frozen map-only reference, initial state, goal, observation and C++ execution; independently run trials','public':'Nav2 Omni MPPI with 1000 samples and near-zero translation spin deadband','hybrid':'Ranger multimode MPPI, 6 samples per family and bounded transition search','source_sha256':hashes}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
for i,(seed,method) in enumerate(trials):
 if any(digest(ROOT/k)!=v for k,v in hashes.items()):raise RuntimeError('Source changed; preserve and stop campaign')
 name=f'{method}_s{seed}';run=OUT/name;result=run/'episode/result.json'
 if result.exists():continue
 if run.exists():raise RuntimeError('Partial run preserved: '+name)
 (OUT/'progress.json').write_text(json.dumps({'running':name,'completed':i,'total':len(trials)})+'\n')
 cmd=[sys.executable,str(ROOT/'src/rangermini_dynamic_semantic/scripts/run_grounded_delivery.py'),'--output',str(run),'--scenario','offset_portal','--seed',str(seed),'--planner','nav2-omni' if method=='nav2' else 'ranger-mppi','--local-pose-goal','12.8','0.55','1.5707963267948966','--policy','predictive','--proposal-selector','cost','--motion-executor','cpp','--verified-recovery','--stable-modes','--map-recovery','--frozen-global-reference',str(ref),'--max-sim-time','90','--max-wall-time','300','--real-time-factor','.5','--ros-domain-id',str(220+i),'--ign-partition','offset_pair_20261029_'+name]
 cmd+=['--upstream-samples','1000'] if method=='nav2' else ['--hybrid-sequences','--samples','6']
 with (OUT/(name+'.log')).open('w') as log:rc=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
 if not result.exists():raise RuntimeError(f'{name} failed rc={rc}; partial run retained')
 r=json.loads(result.read_text());print(name,r.get('status'),r.get('elapsed_sim_s'),flush=True)
(OUT/'progress.json').write_text(json.dumps({'running':None,'completed':len(trials),'total':len(trials)})+'\n')
