"""Summarize valid paired target placements, retaining the invalid-domain attempt."""
from pathlib import Path
import json,csv,math,statistics,sys,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src/rangermini_dynamic_semantic/scripts'),str(ROOT/'src/rangermini_dynamic_semantic'),str(ROOT/'tools'),'/home/USER/.codex/skills/nature-figure/scripts']
from audit_grounded_delivery import audit,rect_gap
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph
from robotics_plot_primitives import draw_map
from audit_panel_alignment import require_matplotlib_panel_alignment

def main():
 base=ROOT/'artifacts/transition_target_pairs_20261027/task_pairs_v5';out=ROOT/'results/10_transition_target_pairs';out.mkdir(parents=True,exist_ok=True)
 protocol=json.loads((base/'protocol.json').read_text());records=[];traces={};sources=[base/'protocol.json']
 for t in protocol['trials']:
  retry=t['target']==2 and t['method']=='hybrid' and t['seed']==53
  run=(ROOT/'artifacts/transition_target_pairs_20261027/target2_hybrid_s53_retry/run') if retry else base/t['name']
  result=json.loads((run/'episode/result.json').read_text());manifest=json.loads((run/'episode/manifest.json').read_text());assert result['seed']==t['seed']==manifest['mppi_config']['random_seed'];assert result['elapsed_sim_s']>=0
  trace=[json.loads(l) for l in (run/'episode/trace.jsonl').read_text().splitlines()];times=np.array([r['timestamp'] for r in trace]);assert np.all(np.diff(times)>=0) and times[0]<10,'Clock preflight violated'
  proxy=audit(run);graph=SpatialGraph(run/'world.sdf');gaps=[min(rect_gap(r['pose'],b[:3],bl=2*b[3],bw=2*b[4]) for b in graph.static_boxes) for r in trace]
  modes=[];holds=0
  for r,dt in zip(trace,np.diff(times,append=times[-1])):
   a=r['actuation'];u=a.get('executed',{})
   if math.hypot(u.get('vx',0),u.get('vy',0))>.02 or abs(u.get('wz',0))>.02:modes.append(a['mode'])
   holds+=dt*bool(a.get('steering_transition_stop'))
  z=dict(target=t['target'],method=t['method'],seed=t['seed'],status=result['status'],elapsed_s=result['elapsed_sim_s'],path_m=result['path_length_m'],nonzero_mode_changes=sum(a!=b for a,b in zip(modes,modes[1:])),alignment_s=float(holds),minimum_static_gap_m=min(gaps),proxy_overlap_samples=proxy['proxy_overlap_trace_count'],qualified=bool(result['task_success'] and min(gaps)>0 and not proxy['proxy_overlap_trace_count']),run=str(run.relative_to(ROOT)),infrastructure_retry=retry)
  records.append(z);traces[(t['target'],t['method'],t['seed'])]=trace;sources+=[run/'episode/result.json',run/'episode/trace.jsonl',run/'world.sdf']
 with (out/'metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 summary={m:{'arrival_count':sum(r['status']=='ARRIVED' for r in records if r['method']==m),'qualified_count':sum(r['qualified'] for r in records if r['method']==m),'n':6,'mean_completion_s':statistics.mean(r['elapsed_s'] for r in records if r['method']==m),'sd_completion_s':statistics.stdev(r['elapsed_s'] for r in records if r['method']==m),'mean_nonzero_mode_changes':statistics.mean(r['nonzero_mode_changes'] for r in records if r['method']==m),'mean_alignment_s':statistics.mean(r['alignment_s'] for r in records if r['method']==m)} for m in ['fixed','hybrid']}
 wins=sum(next(r for r in records if (r['target'],r['seed'],r['method'])==(target,seed,'hybrid'))['elapsed_s']<next(r for r in records if (r['target'],r['seed'],r['method'])==(target,seed,'fixed'))['elapsed_s'] for target in range(3) for seed in [37,53])
 (out/'summary.json').write_text(json.dumps(dict(kind='Three terminal placements in one static scene, deterministic goal conditions, two sampling initializations each',records=records,summary=summary,paired_faster_count=wins,paired_count=6,invalid_attempt='task_pairs_v5/target2_hybrid_s53 excluded due to multiple clocks from a retained diagnostic process; identical frozen code retried in domain219',source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),indent=2)+'\n')
 plt.rcParams.update({'font.family':'Liberation Serif','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(1,3,figsize=(7.16,2.7));fig.subplots_adjust(left=.07,right=.985,bottom=.22,top=.91,wspace=.28);world=base/'target0_hybrid_s37/world.sdf';g=SpatialGraph(world);lim=(10.6,13.5,-2.,.95)
 for i,ax in enumerate(axes):
  draw_map(ax,g.static_boxes,lim)
  for m,color,ls in [('fixed','#c87d43','--'),('hybrid','#117d81','-')]:
   a=traces[(i,m,37)];xy=np.asarray([r['pose'][:2] for r in a]);ax.plot(xy[:,0],xy[:,1],c=color,ls=ls,lw=1.2)
  ax.plot(11,-1.65,'o',ms=4,mfc='white',mec='#26343a');ax.plot(protocol['goals'][i][0],protocol['goals'][i][1],'*',ms=7,c='#26343a');ax.set(xlabel='x (m)',ylabel='y (m)');ax.set_title('('+chr(97+i)+') Service pose '+chr(65+i),loc='left',fontsize=8);ax.tick_params(labelsize=6.5)
 fig.legend([Line2D([0],[0],color='#c87d43',ls='--'),Line2D([0],[0],color='#117d81')],['Fixed boundaries','Searched boundaries'],loc='lower center',ncol=2,frameon=False,fontsize=7)
 require_matplotlib_panel_alignment(fig,json_out=out/'target_trajectories.alignment.json',strict=True)
 for ext in ['pdf','svg','png']:fig.savefig(out/f'target_trajectories.{ext}',dpi=400)
 print(json.dumps(summary,indent=2));print('Faster pairs',wins,'/6')
if __name__=='__main__':main()
