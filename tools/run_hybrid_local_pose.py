"""Frozen local-pose experiment: public MPPI and matched sequence/timing ablations."""
import argparse,hashlib,json,math,random,subprocess,sys,fcntl,os,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/rangermini_dynamic_semantic/scripts'))
from audit_grounded_delivery import audit
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+',default=[37,41,53]);p.add_argument('--methods',nargs='+',choices=['nav2','primitive','fixed','hybrid'],default=['nav2','primitive','fixed','hybrid']);p.add_argument('--direct-goal-reference',action='store_true');a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
lock=(ROOT/'artifacts/hybrid_sequence_20261022/campaign.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
files=list((ROOT/'src/rangermini_dynamic_semantic/rangermini_dynamic_semantic').glob('*.py'))+list((ROOT/'src/rangermini_dynamic_semantic/scripts').glob('*.py'))+[Path(__file__).resolve(),ROOT/'integration/nav2_mppi_adapter/bridge.cpp',ROOT/'integration/nav2_mppi_adapter/build/libranger_nav2_mppi.so']
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files += [ROOT/'integration/mppi_native_core'/name for name in ('core.cpp','CMakeLists.txt','build.py','provenance.json','build/libranger_mppi_native.so')]
hashes={str(p.relative_to(ROOT)):digest(p) for p in files}
method_order=['nav2','primitive','fixed','hybrid'];seed_order=sorted(a.seeds)
domain_map={f'offset_pose_{method}_s{seed}':100+method_order.index(method)*len(seed_order)+seed_order.index(seed)
            for method in a.methods for seed in seed_order}
protocol={'kind':'single local-pose navigation, not preset doorway stages','scene':'offset_portal','portal_clear_width_m':1.5,'goal':[12.8,.55,math.pi/2],'seeds':a.seeds,'methods':a.methods,'ros_domain_id_by_trial':domain_map,'max_sim_s':90,'max_wall_s':300,'requested_real_time_factor':.5,'internal_rollout_cap':101,'public_samples':1000,'horizon_steps':40,'model_dt_s':.12,'common':['world geometry','pose goal and measured arrival rule','finite-rate wheel execution','current laser observations',('direct pose-goal reference' if a.direct_goal_reference else 'registered route geometry')],'internal_shared':([] if a.direct_goal_reference else ['route preview'])+['goal/obstacle/preparation/switching costs','hard rollout-evaluation cap'],'changed':{'primitive':'zero requested transitions','fixed':'up to two transitions, equal-duration phases','hybrid':'up to two transitions, searched boundaries and one-step refinement','nav2':'installed official Omni MPPI optimizer and critics; lateral sampling retained'},'source_sha256':hashes}
f=out/'protocol.json'
if f.exists() and json.loads(f.read_text())!=protocol:raise RuntimeError('Frozen source changed; separate output required')
f.write_text(json.dumps(protocol,indent=2)+'\n');results=[]
for seed in a.seeds:
 order=a.methods[:];random.Random(seed).shuffle(order)
 for method in order:
  if any(digest(ROOT/k)!=v for k,v in hashes.items()):raise RuntimeError('Source changed during frozen experiment')
  name=f'offset_pose_{method}_s{seed}';run=out/name
  (out/'progress.json').write_text(json.dumps(dict(running=name,complete=len(results),intended=len(a.seeds)*len(a.methods)),indent=2))
  if not (run/'episode/result.json').exists():
   if run.exists():raise RuntimeError('Partial attempt preserved: '+name)
   cmd=[sys.executable,str(ROOT/'src/rangermini_dynamic_semantic/scripts/run_grounded_delivery.py'),'--output',str(run),'--planner','nav2-omni' if method=='nav2' else 'ranger-mppi','--local-pose-goal',*map(str,protocol['goal']),'--scenario','offset_portal','--seed',str(seed),'--policy','predictive','--proposal-selector','cost','--verified-recovery','--stable-modes','--samples','6','--upstream-samples','1000','--max-sim-time','90','--max-wall-time','300','--real-time-factor','.5','--ros-domain-id',str(domain_map[name]),'--ign-partition','hybrid_native_'+name]
   if not a.direct_goal_reference:cmd+=['--map-recovery']
   if method!='nav2':cmd+=['--hybrid-sequences','--hybrid-max-switches','0' if method=='primitive' else '2']
   if method=='fixed':cmd+=['--hybrid-fixed-timing']
   print('START',name,flush=True)
   with (out/'campaign.log').open('a') as stream:
    child=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    try:rc=child.wait()
    except KeyboardInterrupt:
     child.send_signal(signal.SIGINT)
     try:child.wait(timeout=20)
     except subprocess.TimeoutExpired:
      os.killpg(child.pid,signal.SIGTERM)
      try:child.wait(timeout=5)
      except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
     raise
   if not (run/'episode/result.json').exists():raise RuntimeError(f'Attempt preserved without terminal result: {name}, rc={rc}')
  r=audit(run);results.append(dict(r,method=method,seed=seed));(out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
  print('DONE',name,r['status'],flush=True)
(out/'progress.json').write_text(json.dumps(dict(running=None,complete=len(results),intended=len(results)),indent=2)+'\n')
