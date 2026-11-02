"""Paired observed outcomes from the frozen task-conditioned and transition studies."""
from pathlib import Path
import hashlib,json,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,'/home/USER/.codex/skills/nature-figure/scripts')
from audit_panel_alignment import require_matplotlib_panel_alignment

def main():
    sources=[ROOT/'artifacts/task_condition_benchmark_20261023/task_matrix_v8/analysis/metrics.json',ROOT/'artifacts/hybrid_sequence_20261022/offset_portal_domain_isolated_matrix_v6/analysis/metrics.json']
    passage=json.loads(sources[0].read_text())['metrics'];entry=json.loads(sources[1].read_text())['metrics']
    passage=[r for r in passage if r['context']=='heading_passing' and r['condition']=='heading_clearance' and r['method'] in ('fixed','hybrid')]
    entry=[r for r in entry if r['method'] in ('fixed','hybrid')]
    plt.rcParams.update({'font.family':'Liberation Serif','font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,1,figsize=(3.45,2.65),sharex=True);fig.subplots_adjust(left=.18,right=.97,bottom=.19,top=.91,hspace=.55)
    plotted=[]
    for ax,data,key,title,ylabel in [(axes[0],passage,'interaction_heading_rms_deg','(a) Heading-constrained passage','Heading RMS (deg)'),(axes[1],entry,'elapsed_sim_s','(b) Offset entry','Completion time (s)')]:
        seeds=sorted(set(r['seed'] for r in data))
        pairs=[]
        for seed in seeds:
            pair=[next(r for r in data if r['seed']==seed and r['method']==m) for m in ('fixed','hybrid')]
            assert all(r.get('qualified',r.get('arrived',False)) for r in pair)
            y=[r[key] for r in pair];pairs.append(y)
            ax.plot([0,1],y,c='#B4BABE',lw=.75,zorder=1)
            ax.scatter([0,1],y,c=['#777E84','#137E83'],s=14,zorder=3)
        mean=np.mean(pairs,axis=0)
        ax.scatter([0,1],mean,c=['#777E84','#137E83'],marker='D',s=28,edgecolors='white',linewidths=.5,zorder=4)
        ax.set(xlim=(-.28,1.28),ylabel=ylabel);ax.set_title(title,loc='left',fontsize=8,pad=3)
        plotted.append({'panel':title,'values_by_seed':dict(zip(map(str,seeds),pairs)),'mean':mean.tolist(),'n':len(seeds)})
    axes[-1].set_xticks([0,1],['Fixed boundaries','Searched boundaries'])
    require_matplotlib_panel_alignment(fig,json_out=ROOT/'paper/figures/paired_planner_effects.alignment.json',strict=True)
    out=ROOT/'paper/figures/paired_planner_effects'
    for ext in ('pdf','svg','png'):fig.savefig(str(out)+'.'+ext,dpi=450)
    Path(str(out)+'.provenance.json').write_text(json.dumps({'kind':'Paired real closed-loop runs, two distinct frozen studies',
       'panels':plotted,'pairing':'Lines pair methods with the same random seed within each study, no seed-axis interpolation.',
       'eligibility':'Both matched variants completed without evaluated overlap in passage; all entry runs arrived.',
       'scope':'These panels do not combine or replace results of the complete 36-episode benchmark.',
       'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}},indent=2)+'\n')
if __name__=='__main__':main()
