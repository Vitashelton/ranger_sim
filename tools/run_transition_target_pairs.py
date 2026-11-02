"""Deterministic 3-target x 2-seed x 2-strategy Gazebo comparison."""
from pathlib import Path
import argparse,json,hashlib,subprocess,sys,random,fcntl
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src/rangermini_dynamic_semantic/scripts'),str(ROOT/'src/rangermini_dynamic_semantic')]
from motion_planning_research_scene import create_world
from rangermini_dynamic_semantic.footprint_route_guide import FootprintRouteGuide
import math,numpy as np
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph
from audit_grounded_delivery import audit

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def freeze(world,start,goal,start_yaw,goal_yaw,output):
 graph=SpatialGraph(world)
 class SharedStaticGuide(FootprintRouteGuide):
  def free(self,points,yaw,margin=.10):
   # Conservative map-only reference, feasible across sampled orientations.
   return np.logical_and.reduce([super(SharedStaticGuide,self).free(points,a,margin) for a in np.linspace(0,math.pi,13)])
 planner=SharedStaticGuide(graph)
 _,points=planner.guide((*start,start_yaw),goal)
 points=[tuple(map(float,p[:2])) for p in points] or [tuple(start),tuple(goal)]
 if math.dist(points[0],start)>1e-6:points.insert(0,tuple(start))
 if math.dist(points[-1],goal)>1e-6:points.append(tuple(goal))
 # A 4WS translation reference need not align body heading to path tangent.
 assert all(planner.segment(a,b,start_yaw) for a,b in zip(points,points[1:])), 'Static path not feasible at common reference heading'
 assert graph.supports_pose((*goal,goal_yaw),margin=.10)
 dense=[points[0]]
 for a,b in zip(points,points[1:]):
  n=max(1,math.ceil(math.dist(a,b)/.12));dense.extend((a[0]+(b[0]-a[0])*i/n,a[1]+(b[1]-a[1])*i/n) for i in range(1,n+1))
 payload=dict(format='ranger_static_global_reference_v1',planner='FootprintRouteGuide static-map A*, conservative multi-heading clearance',start_pose=[*start,start_yaw],goal_pose=[*goal,goal_yaw],path_xy=dense,path_length_m=sum(math.dist(a,b) for a,b in zip(dense,dense[1:])),static_geometry_sha256=hashlib.sha256(np.asarray(graph.static_boxes,dtype='<f8').tobytes()).hexdigest(),all_segments_passed_static_footprint_check=True,validation_body_heading_rad=start_yaw,heading_note='Translation path validated at common body heading; terminal orientation checked separately. Local planner evaluates steering transitions.',dynamic_actor_policy='Excluded from global route')
 Path(output).write_text(json.dumps(payload,indent=2)+'\n')

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
 lock=(out/'campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 goals=[(12.65,.55,1.5707963267948966),(12.8,.55,1.5707963267948966),(12.95,.55,1.5707963267948966)]
 world=out/'offset_portal.sdf'
 if not world.exists():create_world(world,'offset_portal')
 g=SpatialGraph(world);assert all(g.supports_pose(q,margin=.10) for q in goals)
 files=[*list((ROOT/'src/rangermini_dynamic_semantic/rangermini_dynamic_semantic').glob('*.py')),*list((ROOT/'src/rangermini_dynamic_semantic/scripts').glob('*.py')),Path(__file__).resolve(),ROOT/'integration/mppi_native_core/core.cpp',ROOT/'integration/mppi_native_core/build/libranger_mppi_native.so']
 hashes={str(p.relative_to(ROOT)):digest(p) for p in files}
 trials=[]
 for j,goal in enumerate(goals):
  for seed in [37,53]:
   methods=['fixed','hybrid'];random.Random(seed+j).shuffle(methods)
   for method in methods:trials.append(dict(name=f'target{j}_{method}_s{seed}',target=j,goal=goal,seed=seed,method=method,domain=180+len(trials)))
 protocol=dict(kind='Deterministic task-goal geometry comparison, not an online VLM trial',scenario='offset_portal',goals=goals,seeds=[37,53],strategies=['fixed','hybrid'],planned_episodes=12,rollout_cap=101,methods_interpretation='Two search strategies; equal cap does not assert identical candidate libraries or random streams.',shared=['same world/start/initial wheel state','same map-based FootprintRouteGuide implementation, with no hand-authored route','same local pose arrival and execution model','no online semantic API variability'],source_sha256=hashes,trials=trials)
 pf=out/'protocol.json'
 if pf.exists():assert json.loads(pf.read_text())==json.loads(json.dumps(protocol)),'Frozen source/protocol changed'
 else:pf.write_text(json.dumps(protocol,indent=2)+'\n')
 results=[]
 for t in trials:
  run=out/t['name'];result=run/'episode/result.json'
  if not result.exists():
   if run.exists():raise RuntimeError('Partial attempt retained '+str(run))
   assert all(digest(ROOT/k)==v for k,v in hashes.items())
   (out/'progress.json').write_text(json.dumps(dict(complete=len(results),planned=12,running=t['name']),indent=2))
   cmd=[sys.executable,str(ROOT/'src/rangermini_dynamic_semantic/scripts/run_grounded_delivery.py'),'--output',str(run),'--scenario','offset_portal','--seed',str(t['seed']),'--planner','ranger-mppi','--local-pose-goal',*map(str,t['goal']),'--policy','predictive','--proposal-selector','cost','--motion-executor','cpp','--verified-recovery','--stable-modes','--map-recovery','--hybrid-sequences','--samples','6','--max-sim-time','90','--max-wall-time','300','--real-time-factor','.5','--ros-domain-id',str(t['domain']),'--ign-partition',out.name+'_'+t['name']]
   if t['method']=='fixed':cmd+=['--hybrid-fixed-timing']
   print('START',t['name'],flush=True)
   with (out/(t['name']+'.log')).open('w') as log:rc=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
   if not result.exists():
    (out/'progress.json').write_text(json.dumps(dict(complete=len(results),planned=12,stopped=t['name'],return_code=rc),indent=2));raise RuntimeError('No terminal record, attempt retained')
  actual=json.loads(result.read_text());manifest=json.loads((run/'episode/manifest.json').read_text());assert actual['seed']==t['seed'] and manifest['mppi_config']['random_seed']==t['seed'],'Actual sampler seed does not match protocol';row=dict(t,result=actual,audit=audit(run));results.append(row)
  (out/'results.json').write_text(json.dumps(results,indent=2)+'\n');print('DONE',t['name'],row['result'].get('status'),flush=True)
 (out/'progress.json').write_text(json.dumps(dict(complete=len(results),planned=12,running=None),indent=2))
if __name__=='__main__':main()
