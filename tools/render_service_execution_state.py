"""Measured kinematic response after receipt acknowledgement in one service mission."""
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
    run=ROOT/'artifacts/task_guided_online_20261023/mission_v6_cpp_s41/run'
    read=lambda p:[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    source=run/'episode/trace.jsonl';events_path=run/'episode/selection_events.jsonl'
    events=read(events_path)
    ack=next(e for e in events if e['event']=='SERVICE_ACTION_CONFIRMED' and e['room']=='614' and e['action']['type']=='pickup')
    end=next(e for e in events if e['event']=='SERVICE_ROUTE_ADVANCED' and e.get('old_stage')=='EXIT_CLEAR' and e.get('stop_index')==2)
    data=[r for r in read(source) if ack['sim_time']<=r['timestamp']<=end['sim_time']]
    t=np.array([r['timestamp']-ack['sim_time'] for r in data]);yaw=np.rad2deg(np.unwrap([r['pose'][2] for r in data]))
    plt.rcParams.update({'font.family':'Liberation Serif','font.size':8,'axes.labelsize':8,'xtick.labelsize':7,
       'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,1,figsize=(3.45,2.35),sharex=True)
    fig.subplots_adjust(left=.19,right=.985,bottom=.20,top=.96,hspace=.30)
    axes[0].plot(t,yaw-yaw[0],c='#137E83',lw=1.1);axes[0].set_ylabel(r'$\Delta\psi$ (deg)')
    for key,c,ls,label in [('vx','#137E83','-',r'$v_x$'),('vy','#777E84','--',r'$v_y$')]:
        axes[1].plot(t,[r['actuation']['executed'][key] for r in data],c=c,ls=ls,lw=1.,label=label)
    axes[1].set(xlabel='Time after receipt acquisition (s)',ylabel='Body velocity (m/s)',ylim=(-.38,.5))
    axes[1].legend(loc='lower left',frameon=False,fontsize=7,ncol=2,handlelength=1.4,columnspacing=.8)
    require_matplotlib_panel_alignment(fig,json_out=ROOT/'paper/figures/service_execution_state.alignment.json',strict=True)
    out=ROOT/'paper/figures/service_execution_state'
    for ext in ('pdf','svg','png'):fig.savefig(str(out)+'.'+ext,dpi=450)
    Path(str(out)+'.provenance.json').write_text(json.dumps({'kind':'Actual one-run body pose and executed kinematic velocities after simulated receipt acquisition',
      'window_start':ack,'window_end':end,'heading_transform':'World yaw unwrapped, relative to the first recorded state after receipt acknowledgement, converted to degrees; no smoothing.',
      'velocity_frame':'Executed body-frame vx,vy from actuation feedback in kinematic simulation.',
      'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,events_path]}},indent=2)+'\n')

if __name__=='__main__':main()
