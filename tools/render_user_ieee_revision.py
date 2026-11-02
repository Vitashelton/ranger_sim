"""Adapt the author's IEEE figure layouts to verified local records.

Original assets and generator are preserved in paper/figures/user_revision.
This dependency-light renderer preserves the author's panel roles, colors and
single/double-column sizes, clips world polygons, and pairs actor/robot poses
at each run's own minimum-gap sample. No measured samples are synthesized.
"""
import csv
import hashlib
import json
import statistics
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results'
OUT=ROOT/'paper/figures/user_revision/final'
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'src/rangermini_dynamic_semantic'))
sys.path.insert(0,str(ROOT/'tools'))
sys.path.insert(0,'/home/USER/.codex/skills/nature-figure/scripts')
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph
from plot_nav2_matched_cases import corners,clipped_polygon
from audit_panel_alignment import require_matplotlib_panel_alignment

COL={'primitive':'#85898C','fixed':'#C77533','hybrid':'#087A89'}
MOT={'DUAL_ACKERMANN':'#C77533','PARALLEL':'#087A89','SPINNING':'#806D9A','STOPPED':'#6B737A'}
sources={}
plt.rcParams.update({'font.family':'Liberation Serif','font.size':8,'axes.labelsize':8,
    'xtick.labelsize':7.2,'ytick.labelsize':7.2,'legend.fontsize':7.2,'axes.linewidth':.65,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none',
    'mathtext.fontset':'stix','legend.frameon':False})


def source(p):
    sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return p


def rows(p):
    result=[]
    with source(p).open()as f:
        for r in csv.DictReader(f):
            for k,v in r.items():
                if v in ('True','False'):r[k]=v=='True'
                elif v:
                    try:r[k]=float(v)
                    except ValueError:pass
            result.append(r)
    return result


def states(group,name):
    return rows(DATA/group/name/'executed_states.csv')


def geo(ax,world,bounds):
    for box in SpatialGraph(source(world)).static_boxes:
        poly=clipped_polygon(corners(box[:3],*box[3:]),bounds)
        if poly:ax.add_patch(Polygon(poly,fc='#D3D8DC',ec='#8C969D',lw=.5,zorder=1))
    ax.set(xlim=bounds[:2],ylim=bounds[2:],xlabel='$x$ (m)',ylabel='$y$ (m)')
    ax.set_aspect('equal',adjustable='box')
    ax.tick_params(length=2.5,pad=2)


def footprint(ax,r,color):
    pose=[r['x_m'],r['y_m'],r['yaw_rad']]
    ax.add_patch(Polygon(corners(pose),fc=matplotlib.colors.to_rgba(color,.10),ec=color,lw=.8,zorder=5))
    ax.plot([pose[0],pose[0]+.35*np.cos(pose[2])],
            [pose[1],pose[1]+.35*np.sin(pose[2])],color=color,lw=.7,zorder=6)


def save(fig,stem,**details):
    fig.canvas.draw()
    if len(fig.axes)>1:
        if stem.startswith('Fig05'):
            image_axis,plot_axis=fig.axes
            rect=plot_axis.get_position()
            old=image_axis.get_position()
            image_width=rect.height*fig.get_figheight()*4/3/fig.get_figwidth()
            image_axis.set_position([old.x0,rect.y0,image_width,rect.height])
            fig.canvas.draw()
            require_matplotlib_panel_alignment(fig,axes=fig.axes,panel_ids=['a','b'],
                row_groups=[['a','b']],column_groups=[],json_out=OUT/f'{stem}.alignment.json')
        else:
            require_matplotlib_panel_alignment(fig,axes=fig.axes,panel_ids=list('abcd'),
                row_groups=[],column_groups=[['b','c','d']],json_out=OUT/f'{stem}.alignment.json')
    fig.savefig(OUT/f'{stem}.pdf')
    fig.savefig(OUT/f'{stem}.svg')
    fig.savefig(OUT/f'{stem}.png',dpi=600)
    plt.close(fig)
    provenance=dict(adaptation='User-supplied IEEE layouts, verified measured records',
        transforms='No pose smoothing or interpolation; world boxes clipped to display region',
        original_package='paper/figures/user_revision/import_provenance.json',
        source_sha256={**sources,str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},**details)
    (OUT/f'{stem}.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


def main():
    group='01_planner_comparison'
    fig=plt.figure(figsize=(3.5,3.55));ax=fig.add_axes([.17,.15,.79,.77])
    geo(ax,DATA/group/'world.sdf',(10.52,13.31,-1.93,.92))
    for key,label,ls in [('primitive','Primitive',(0,(3,2))),('fixed','Fixed timing',(0,(4,1.5))),('hybrid','Optimized timing','-')]:
        rr=states(group,f'offset_pose_{key}_s37')
        ax.plot([r['x_m']for r in rr],[r['y_m']for r in rr],c=COL[key],lw=1.55,ls=ls,label=label,zorder=4)
        if key=='hybrid':
            for i in [int(len(rr)*.19),int(len(rr)*.73)]:footprint(ax,rr[i],COL[key])
    ax.plot(11,-1.65,'o',ms=4,mfc='white',mec='#23282D',zorder=8)
    ax.plot(12.8,.55,'*',ms=8,color='#2A383D',zorder=8)
    ax.legend(loc='upper left',fontsize=7.5,handlelength=2.1)
    save(fig,'Fig02_Matched_offset_doorway_s37',seed=37,footprints='Two measured optimized-trajectory poses at original author-selected indices')

    rr=rows(DATA/group/'episode_metrics.csv');seeds=sorted({int(r['seed'])for r in rr})
    fig=plt.figure(figsize=(3.5,2.47));ax=fig.add_axes([.22,.20,.74,.61])
    for index,seed in enumerate(seeds):
        vals=[next(r['elapsed_sim_s']for r in rr if r['seed']==seed and r['method']==m)for m in ['fixed','hybrid']]
        y=len(seeds)-1-index
        ax.plot(vals,[y,y],c='#BCC3C6',lw=1)
        for v,m,mark in zip(vals,['fixed','hybrid'],['s','o']):ax.plot(v,y,marker=mark,ms=5.3,c=COL[m])
    ax.set(xlim=(14.8,20.1),ylim=(-.58,4.62),xlabel='Arrival time (s)',ylabel='Matched seed')
    ax.set_yticks(list(range(5))[::-1],[str(s)for s in seeds]);ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y',length=0)
    ax.legend(handles=[Line2D([],[],marker=k,ls='',c=COL[m],label=l)for m,k,l in
        [('fixed','s','Fixed timing'),('hybrid','o','Optimized timing')]],loc='lower center',bbox_to_anchor=(.5,1.11),ncol=2,fontsize=7,columnspacing=.8,handletextpad=.2)
    save(fig,'Fig03_Paired_arrival_times',paired_seeds=seeds,pairing='Each line links the two runs with the same seed; no trend across seeds')

    fig=plt.figure(figsize=(3.5,1.75));ax=fig.add_axes([.22,.30,.74,.48])
    for y,key in [(1,'fixed'),(0,'hybrid')]:
        rr=states(group,f'offset_pose_{key}_s37')
        for a,b in zip(rr,rr[1:]):
            color=MOT[a['mode']]
            ax.plot([a['elapsed_s'],b['elapsed_s']],[y,y],lw=8,color=color,solid_capstyle='butt')
            if a['steering_alignment_stop']:
                ax.plot([a['elapsed_s'],b['elapsed_s']],[y-.25,y-.25],lw=2,color='#23282D',solid_capstyle='butt')
    ax.set(xlim=(0,20),ylim=(-.55,1.55),xlabel='Elapsed simulation time (s)',xticks=[0,5,10,15,20])
    ax.set_yticks([1,0],['Fixed timing','Optimized']);ax.tick_params(axis='y',length=0)
    ax.spines['left'].set_visible(False)
    fig.legend(handles=[Patch(fc=MOT[k],label=l)for k,l in [('DUAL_ACKERMANN','Curved'),('PARALLEL','Parallel'),('SPINNING','Spin')]]+
        [Line2D([],[],c='#23282D',lw=2,label='Alignment')],loc='upper center',bbox_to_anchor=(.51,.97),ncol=4,fontsize=6.5,handlelength=.9,columnspacing=.8)
    save(fig,'Fig04_Executed_mode_timeline_s37',seed=37,alignment='Black underline is recorded steering hold; bar color is native execution mode')

    group='02_visual_condition'
    fig=plt.figure(figsize=(7.16,2.85))
    a=fig.add_axes([.025,.13,.40,.77]);raw=source(DATA/group/'model_records/proposal_002_raw_rgb.jpg')
    a.imshow(plt.imread(raw));a.axis('off')
    ax=fig.add_axes([.51,.18,.47,.71]);geo(ax,DATA/group/'world.sdf',(3.9,9.2,-3.8,-1.1))
    interaction=[]
    for key,label,col,ls in [('no_condition','Without visual condition','#C77533','--'),('vlm','With visual condition','#087A89','-')]:
        rr=states(group,key);gaps=rows(DATA/group/key/'clearance.csv');g=min(gaps,key=lambda g:g['trolley_signed_SAT_gap_m'])
        r=min(rr,key=lambda r:abs(r['sim_time_s']-g['timestamp']))
        assert abs(r['sim_time_s']-g['timestamp'])<1e-8
        ax.plot([r['x_m']for r in rr],[r['y_m']for r in rr],c=col,lw=1.6,ls=ls,label=label,zorder=4)
        footprint(ax,r,col)
        cart=json.loads(g['trolley_pose'])
        ax.add_patch(Polygon(corners(cart,.35,.21),fc='none',ec=col,lw=.8,ls=ls,zorder=3))
        interaction.append(dict(method=key,sim_time_s=r['sim_time_s'],vehicle_pose=[r['x_m'],r['y_m'],r['yaw_rad']],trolley_pose=cart,gap_m=g['trolley_signed_SAT_gap_m']))
    ax.legend(loc='upper left',fontsize=7.2)
    fig.text(.025,.95,'(a) Camera frame for the visual request',fontsize=8.3)
    fig.text(.51,.95,'(b) Recorded interaction trajectories',fontsize=8.3)
    save(fig,'Fig05_Online_VLM_RGB_and_paths',interaction_pairs=interaction,
        camera='Original RGB source frame, not the full composite API payload',
        footprint_pairing='Each vehicle/trolley pair is sampled at its own run minimum gap; no common-time claim')

    group='03_multistop_service'
    fig=plt.figure(figsize=(7.16,3.25));ax=fig.add_axes([.07,.18,.66,.69])
    geo(ax,DATA/group/'world.sdf',(2,14.35,-5.56,1.55));rr=states(group,'.')
    xy=np.array([[r['x_m'],r['y_m']]for r in rr]);segments=np.stack([xy[:-1],xy[1:]],axis=1)
    ax.add_collection(LineCollection(segments,colors=[MOT[r['mode']]for r in rr[:-1]],lw=1.6,zorder=4))
    events=[e for e in rows(DATA/group/'events.csv')if e['event']=='SERVICE_ACTION_CONFIRMED']
    locations=[]
    for number,e in enumerate(events,1):
        r=min(rr,key=lambda r:abs(r['sim_time_s']-e['sim_time']));locations.append(dict(event=number,room=e['room'],sim_time_s=e['sim_time'],measured_sample_time_s=r['sim_time_s']))
        if number==3:continue
        ax.plot(r['x_m'],r['y_m'],'o',mfc='white',mec='#23282D',ms=4,zorder=7)
        dx,dy=(5,4)if number!=2 else(7,2)
        ax.annotate('2, 3'if number==2 else str(number),(r['x_m'],r['y_m']),xytext=(dx,dy),textcoords='offset points',fontsize=7,zorder=8)
    ax.plot(rr[0]['x_m'],rr[0]['y_m'],'o',ms=3,c='#23282D',zorder=7)
    fig.text(.07,.94,'(a) Measured service-navigation trajectory',fontsize=8.5)
    fig.legend(handles=[Patch(fc=MOT[k],label=l)for k,l in [('DUAL_ACKERMANN','Curved'),('PARALLEL','Parallel/lateral'),('SPINNING','Spin')]],
        loc='lower left',bbox_to_anchor=(.10,.005),ncol=3,fontsize=7,handlelength=1,columnspacing=1)
    for j,(name,room)in enumerate([('00073.jpg','617'),('00227.jpg','614'),('00484.jpg','619')]):
        a=fig.add_axes([.77,.67-j*.265,.22,.21]);a.imshow(plt.imread(source(DATA/group/'rgb'/name)));a.axis('off')
        a.set_title(f'({chr(98+j)}) Office {room}',loc='left',fontsize=8,pad=4)
    save(fig,'Fig06_Service_mission_spatial_trace',service_events=locations,
        plan='Replayed archived language-model plan; service acknowledgements simulated; no fresh visual query in this mission')
    metrics=rows(DATA/'01_planner_comparison/episode_metrics.csv')
    summary=[]
    for method in ('primitive','fixed','hybrid'):
        rr=[r for r in metrics if r['method']==method]
        stats={k:dict(mean=statistics.mean(r[k]for r in rr),sd=statistics.stdev(r[k]for r in rr))
               for k in ('path_m','elapsed_sim_s','nonzero_mode_changes','sampled_alignment_s')}
        summary.append(dict(method=method,n=len(rr),statistics=stats,
                            minimum_static_gap_m=min(r['min_static_signed_SAT_m']for r in rr)))
    table=[r'\begin{table}[t]\centering',
        r'\caption{Matched offset-doorway navigation (five seeds). Entries are means; $t$ is arrival time, $L$ path length, $N_s$ nonzero mode changes and $T_a$ alignment hold. All runs arrived.}\label{tab:transitionoutcomes}',
        r'\footnotesize\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{lrrrr}\toprule',
        r'Configuration & $t$ (s) & $L$ (m) & $N_s$ & $T_a$ (s) \\ \midrule']
    for r,label in zip(summary,('Primitive MPPI','Fixed timing','Optimized timing')):
        stats=r['statistics']
        table.append(label+' & '+' & '.join(f"{stats[k]['mean']:.2f}"for k in
            ('elapsed_sim_s','path_m','nonzero_mode_changes','sampled_alignment_s'))+r' \\')
    table.extend([r'\bottomrule\end{tabular}\end{table}'])
    (ROOT/'paper/figures/hybrid_transition_outcomes.tex').write_text('\n'.join(table)+'\n')
    stats={r['method']:r['statistics']for r in summary}
    paired=[]
    for seed in (37,41,53,59,61):
        rr={r['method']:r for r in metrics if r['seed']==seed}
        dt=rr['fixed']['elapsed_sim_s']-rr['hybrid']['elapsed_sim_s']
        dh=rr['fixed']['sampled_alignment_s']-rr['hybrid']['sampled_alignment_s']
        paired.append(dict(seed=seed,completion_reduction_s=dt,alignment_reduction_s=dh,
                           reduction_outside_alignment_holds_s=dt-dh))
    (ROOT/'results/paired_transition_decomposition.csv').write_text(
        ','.join(paired[0])+'\n'+'\n'.join(','.join(str(v)for v in r.values())for r in paired)+'\n')
    provenance=dict(all_summary_seeds=[37,41,53,59,61],statistics=summary,
        table='Single-column table following the user IEEE revision; uncertainty is reported with paired results in the main text',
        mean_completion_reduction_s=statistics.mean(r['completion_reduction_s']for r in paired),
        mean_alignment_reduction_s=statistics.mean(r['alignment_reduction_s']for r in paired),
        source_sha256={str(DATA/'01_planner_comparison/episode_metrics.csv'):hashlib.sha256((DATA/'01_planner_comparison/episode_metrics.csv').read_bytes()).hexdigest()},
        table_sha256=hashlib.sha256((ROOT/'paper/figures/hybrid_transition_outcomes.tex').read_bytes()).hexdigest())
    (ROOT/'paper/figures/hybrid_transition_outcomes.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Rendered five user-layout evidence figures from verified records.')


if __name__=='__main__':main()
