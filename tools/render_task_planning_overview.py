"""Single-column planning dataflow for the four-wheel-steered robot."""
from pathlib import Path
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch
ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'Liberation Serif','font.size':7.5,'pdf.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=(3.42,3.0));ax=fig.add_axes([.02,.01,.96,.98]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
def box(x,y,w,h,text,weight='normal',fs=7.5):
 ax.add_patch(Rectangle((x,y),w,h,fc='white',ec='#222222',lw=.7));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,fontweight=weight)
def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=7,lw=.7,color='#222222',shrinkA=0,shrinkB=0))
box(.04,.83,.40,.12,'Service request',weight='bold');box(.56,.83,.40,.12,'RGB and tracked objects',weight='bold')
box(.20,.64,.60,.12,'VLM grounding\nGoal and spatial condition');arrow((.24,.83),(.35,.76));arrow((.76,.83),(.65,.76))
ax.add_patch(Rectangle((.08,.22),.84,.36,fc='white',ec='#333333',lw=.7));ax.text(.50,.555,'Multimode MPPI',ha='center',va='center',fontweight='bold',fontsize=8)
box(.18,.45,.64,.07,'Motion sequences and transitions',fs=7.1);box(.18,.36,.64,.07,'Wheel-state and footprint rollout',fs=7.1);box(.18,.27,.64,.07,'Task cost and compatible update',fs=7.1)
arrow((.5,.64),(.5,.58));arrow((.5,.45),(.5,.43));arrow((.5,.36),(.5,.34));arrow((.5,.27),(.5,.22))
box(.29,.04,.42,.11,'Execute first control',weight='bold');arrow((.5,.22),(.5,.15))
ax.plot([.71,.97,.97,.82],[.095,.095,.395,.395],color='#555555',lw=.7);arrow((.97,.395),(.82,.395))
out=ROOT/'paper/figures/task_grounded_overview'
for ext in ('pdf','svg','png'):fig.savefig(str(out)+'.'+ext,dpi=450)
out.with_suffix('.provenance.json').write_text(json.dumps({'kind':'Single-column planning dataflow schematic','renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
