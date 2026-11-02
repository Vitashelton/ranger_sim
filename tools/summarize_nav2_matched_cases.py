"""Aggregate paired measured trajectories; failed attempts remain in source rows."""
import argparse,hashlib,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('analysis',type=Path);a=p.parse_args();data=[];sources={}
for seed in (17,37,41):
 f=a.analysis/f's{seed}/nav2_matched_metrics.json';x=json.loads(f.read_text());assert x['episodes']==9
 data+=x['metrics'];sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
assert len(data)==27
out=[]
tex=[r'\begin{table*}[t]\centering\footnotesize',r'\caption{Matched local-planner results, three seeds per scene. L and t: means for arrived trials; d: minimum sampled footprint separation; N: mean nonzero executed-mode changes. All attempts are retained.}\label{tab:nav2matched}',r'\begin{tabular}{llrrrrr}\toprule',r'Scene & Method & Arrived & L (m) & t (s) & d (m) & N \\\midrule']
labels={'static_detour':'Static detour','heading_constrained_passage':'Heading-constrained passage','offset_portal':'Offset doorway'}
names={'nav2-omni':'Nav2 Omni MPPI','multimode':'Multimode MPPI','no-steering-prediction':'No steering prediction'}
for scene in labels:
 for method in names:
  rr=[r for r in data if r['scene']==scene and r['method']==method];assert len(rr)==3
  good=[r for r in rr if r['arrived']]
  avg=lambda field:statistics.mean(r[field] for r in good) if good else None
  v=dict(scene=scene,method=method,n=3,arrived=len(good),qualified=sum(r['qualified'] for r in rr),mean_arrived_path_m=avg('path_m'),mean_arrived_duration_s=avg('elapsed_sim_s'),minimum_sampled_environment_gap_m=min(r['minimum_environment_gap_m'] for r in rr),mean_nonzero_mode_changes=statistics.mean(r['moving_mode_switches'] for r in rr),mean_alignment_s=statistics.mean(r['steering_alignment_s'] for r in rr),peak_heading_error_deg=max(r['approach_peak_heading_error_deg'] for r in rr))
  out.append(v)
  fmt=lambda x:'--' if x is None else f'{x:.2f}'
  gap=v['minimum_sampled_environment_gap_m']; gap_text=(f'${gap/1e-5:.1f}\\times10^{{-5}}$' if abs(gap)<.0005 else f'{gap:.3f}')
  values=[fmt(v['mean_arrived_path_m']),fmt(v['mean_arrived_duration_s']),gap_text,fmt(v['mean_nonzero_mode_changes'])]
  tex.append(labels[scene]+' & '+names[method]+f" & {len(good)}/3 & "+' & '.join(values)+r' \\')
tex += [r'\bottomrule\end{tabular}\end{table*}']
(a.analysis/'aggregate.json').write_text(json.dumps(dict(episodes=27,rows=out,all_attempts=data,source_sha256=sources,renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
(a.analysis/'planner_comparison_table.tex').write_text('\n'.join(tex)+'\n')
print(json.dumps(out,indent=2))
