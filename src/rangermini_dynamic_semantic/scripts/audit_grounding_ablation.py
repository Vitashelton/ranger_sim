#!/usr/bin/env python3
"""Offline matched-perception decision ablations, not new motion experiments."""
import argparse,csv,json,math,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
routes={'LEFT':[(4.80,-1.45),(6.45,-1.45),(7.25,-2.10)],'RIGHT':[(4.80,-2.90),(6.45,-2.90),(7.25,-2.10)]}
def score(anchor,pose,kind,feasible):
 radius=.18 if kind=='person' else .41;px,py=anchor;scores={}
 for k in feasible:
  points=[pose[:2]]+routes[k];gaps=[]
  for a,b in zip(points,points[1:]):
   for t in np.linspace(0,1,max(2,int(math.dist(a,b)/.05)+1)):
    qx=a[0]+t*(b[0]-a[0]);qy=a[1]+t*(b[1]-a[1])
    gaps.append(max(0.,math.hypot(max(abs(px-qx)-.751/2,0),max(abs(py-qy)-.500/2,0))-radius))
  scores[k]=min(gaps)
 return max(scores,key=scores.get),scores
records=[];sensitivity=[]
for base in sorted((root/'runs').glob('*_grounded_s*/episode')):
 name=base.parent.name;side,target,_,seed=name.split('_');response=json.loads((base/'selection_output.json').read_text());inp=json.loads((base/'selection_input.json').read_text())
 box=response['evidence']['bbox'];pose=inp['capture_pose'];feasible=[k for k,v in inp['candidates'].items() if v['selected_trajectory'] and v['selected_behavior']!='STOP']
 expected='RIGHT' if (side=='north')==(target=='person') else 'LEFT'
 image_side='RIGHT' if (box[0]+box[2])/2<.5 else 'LEFT'
 region=np.load(base/'grounding_depth_roi.npy');yy,xx=np.where(np.isfinite(region)&(region>.2)&(region<8));depths=region[yy,xx];f=640/(2*math.tan(1.396/2))
 for quantile in [.05,.10,.20,.30,.50]:
  keep=depths<=np.quantile(depths,quantile);zs=depths[keep];us=xx[keep]+int(box[0]*640);bx=.34+zs;by=-(us-320)*zs/f;x,y,psi=pose
  anchor=[float(np.median(x+math.cos(psi)*bx-math.sin(psi)*by)),float(np.median(y+math.sin(psi)*bx+math.cos(psi)*by))]
  chosen,scores=score(anchor,pose,target,feasible)
  sensitivity.append(dict(episode=name,quantile=quantile,route=chosen,expected=expected,correct=chosen==expected,anchor_x=anchor[0],anchor_y=anchor[1],left_score=scores.get('LEFT'),right_score=scores.get('RIGHT')))
  if quantile==.20:
   replay_route=chosen;replay_error=math.dist(anchor,response['grounded_target_xy'])
 metric=response['decision']['candidate_id']
 target_y=(-.95 if side=='north' else -3.30) if target=='person' else (-3.30 if side=='north' else -.95)
 oracle,oracle_scores=score([4.9,target_y],pose,target,feasible)
 records.append(dict(episode=name,layout=side,target=target,seed=int(seed[1:]),expected=expected,recorded_metric_route=metric,image_only_route=image_side,image_only_correct=image_side==expected,metric_correct=metric==expected,oracle_route=oracle,oracle_correct=oracle==expected,replay_route=replay_route,replay_matches=metric==replay_route,replay_anchor_error_m=replay_error))
assert len(records)==12
for name,data in [('matched_decisions.csv',records),('depth_quantile_sensitivity.csv',sensitivity)]:
 with (out/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
summary=dict(evidence_type='offline recomputation from saved actual VLM boxes and sensor depth; no additional physical episodes or model calls',n=12,
 image_only_correct=sum(r['image_only_correct'] for r in records),metric_correct=sum(r['metric_correct'] for r in records),oracle_correct=sum(r['oracle_correct'] for r in records),
 replay_matches=sum(r['replay_matches'] for r in records),maximum_replay_anchor_error_m=max(r['replay_anchor_error_m'] for r in records),
 sensitivity=[dict(quantile=q,n=12,correct=sum(r['correct'] for r in sensitivity if r['quantile']==q)) for q in [.05,.10,.20,.30,.50]],
 interpretation='Image-only rule is task-aware and uses exactly the same perceived target box. Oracle uses evaluation truth, so is only an upper bound, not a perception-matched baseline. Neither is claimed as new physically executed trials.')
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False))

