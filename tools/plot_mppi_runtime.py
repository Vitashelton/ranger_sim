#!/usr/bin/env python3
"""Group-meeting chart. Source values are measured exclusive profiling phases."""
import json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SKILL=Path('/home/USER/.codex/skills/nature-figure/scripts')
sys.path.insert(0,str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment
OUT=ROOT/'post/group_meeting_20261021/mppi_runtime'
d=json.loads((OUT/'runtime_profile.json').read_text())
# Claim: footprint geometry and 4WS rollout dominate the measured phase budget.
# Experimental unit: one recorded-state full-bank replay (26 snapshots, one run).
# P50/P95 describe this replay workload, not independent navigation trials or CI.
font_path='/home/USER/.local/share/fonts/FandolHei-Regular.otf'
font_manager.fontManager.addfont(font_path)
font=font_manager.FontProperties(fname=font_path)
plt.rcParams.update({'font.family':font.get_name(),'font.size':12,'pdf.fonttype':42,
    'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':.8,'xtick.major.size':0,'axes.unicode_minus':False,
    'legend.frameon':False,'savefig.facecolor':'white'})
keys=['sampling','rollout','collision','cost','postprocess']
labels=['采样与\n参考准备','四轮运动学\n推演','整车足迹\n碰撞检测','代价\n计算','候选\n后处理']
x=np.arange(len(keys));w=.31
p50=np.array([d['phase_summary'][k]['p50_ms'] for k in keys])
p95=np.array([d['phase_summary'][k]['p95_ms'] for k in keys])
fig,ax=plt.subplots(figsize=(10.6,4.8))
fig.subplots_adjust(left=.095,right=.975,bottom=.20,top=.79)
ax.set_axisbelow(True);ax.grid(False)
bars1=ax.bar(x-w/2,p50,w,color='#008C95',label='P50（中位数）',zorder=3)
bars2=ax.bar(x+w/2,p95,w,color='#D97828',label='P95（95% 分位数）',zorder=3)
ceiling=float(max(p95))*1.23
ax.set_ylim(0,ceiling);ax.set_xlim(-.6,4.6)
ax.set_ylabel('耗时 / ms',labelpad=10)
ax.set_xticks(x,labels,fontsize=13)
ax.tick_params(axis='x',pad=10);ax.tick_params(axis='y',labelsize=11)
for bars,values in [(bars1,p50),(bars2,p95)]:
    for bar,value in zip(bars,values):
        ax.text(bar.get_x()+bar.get_width()/2,value+ceiling*.017,f'{value:.1f}',
                ha='center',va='bottom',fontsize=11,color='#30383E')
ax.legend(loc='upper right',fontsize=11)
fig.text(.095,.93,'MPPI 候选库耗时分解',ha='left',va='center',fontsize=20,weight='bold',color='#202A32')
fig.text(.095,.855,f"插桩重放 · {d['n_snapshots']} 个记录状态 · 每运动族 {d['config']['samples_per_behavior']} 样本 × {d['config']['horizon']} 步",ha='left',va='center',fontsize=12,color='#56616A')
fig.canvas.draw()
require_matplotlib_panel_alignment(fig,json_out=OUT/'runtime_breakdown.alignment.json',
    overlay_svg=OUT/'runtime_breakdown.alignment.svg',strict=True)
for ext in ['pdf','svg','png']:
    fig.savefig(OUT/f'runtime_breakdown.{ext}',dpi=300)
plt.close(fig)
caption=('图：记录状态重放中的完整 MPPI 候选库耗时。柱表示26个状态的分项P50和P95，非置信区间。'
         '同一状态的插桩与原始规划输出逐项一致；分项计时互斥，已排除计时器内部记账时间，仍有残余插桩开销。'
         '无插桩整库P50 %.1f ms、P95 %.1f ms；插桩墙钟P50 %.1f ms、P95 %.1f ms。'
         '分项分位数不可相加，也不是对历史660 ms在线窗口的追溯拆分。计时不含ROS通信、进程IPC、A*、VLM和录像。') % (
             d['unprofiled_total_ms']['p50'],d['unprofiled_total_ms']['p95'],
             d['instrumented_total_ms']['p50'],d['instrumented_total_ms']['p95'])
(OUT/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>MPPI 计算开销</title>
<style>body{font:17px/1.65 system-ui;color:#26333c;margin:30px auto;max-width:1180px;padding:0 24px}img{width:100%}a{color:#00838a}p{max-width:1050px}</style>
<h1>MPPI 候选库：计算瓶颈在哪里？</h1><img src="runtime_breakdown.svg" alt="MPPI五阶段P50和P95耗时柱状图"><p>'''+caption+'''</p><p><a href="runtime_breakdown.png">PNG</a> · <a href="runtime_breakdown.pdf">PDF</a> · <a href="runtime_breakdown.svg">SVG</a> · <a href="runtime_profile.csv">分项统计</a> · <a href="runtime_profile.json">逐状态计时与来源</a> · <a href="hotspot_calls.json">函数调用热点</a></p>
<p>优先优化：缓存扫描几何与障碍物预测；将足迹检测和四轮运动学批量化或迁移到C++内核。独立ROS节点解决调度隔离，不会自动降低这些数值计算的耗时。</p><p><a href="../index.html">返回组会展示</a></p>''')
print(OUT/'runtime_breakdown.png')
