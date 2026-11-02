#!/usr/bin/env python3
import argparse,csv,json,math,statistics,sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,'/home/USER/.codex/skills/nature-figure/scripts')
from audit_panel_alignment import require_matplotlib_panel_alignment
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();out=root/'output';out.mkdir(exist_ok=True)
methods=['geometric','image_rule','grounded'];names=['几何 MPPI','任务感知图像规则','深度评分 + MPPI'];colors=['#66717b','#4379a8','#b5653c']
records=[];runs={}
def gap(pose):
 x,y,psi=pose;dx,dy=8.8-x,-.95-y;bx=math.cos(psi)*dx+math.sin(psi)*dy;by=-math.sin(psi)*dx+math.cos(psi)*dy
 return max(0.,math.hypot(max(abs(bx)-.751/2,0),max(abs(by)-.500/2,0))-.18)
import hashlib
worlds=set()
for seed in [17,23,31]:
 for method in methods:
  base=root/'runs'/f'{method}_s{seed}'/'episode';result=json.loads((base/'result.json').read_text());rows=[json.loads(s) for s in (base/'trace.jsonl').read_text().splitlines()]
  assert result['status']=='ARRIVED'
  poses=[r['pose'] for r in rows]+[result['final_pose']];length=0.;prev=(3.,-2.1)
  for pose in poses:length+=math.dist(prev,pose[:2]);prev=pose[:2]
  target=min(map(gap,poses));record=dict(method=method,seed=seed,status=result['status'],route=result['route_id'],min_person_proxy_gap_m=target,target_gap_met=target>=.60,path_m=length,motion_s=result['elapsed_sim_s'])
  records.append(record);runs[(method,seed)]=(base,rows,result,record);worlds.add(hashlib.sha256((base.parent/'world.sdf').read_bytes()).hexdigest())
assert len(worlds)==1
with (out/'episodes.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
aggregates=[]
for m in methods:
 rs=[r for r in records if r['method']==m]
 aggregates.append(dict(method=m,n=3,arrived=sum(r['status']=='ARRIVED' for r in rs),target_gap_met=sum(r['target_gap_met'] for r in rs),routes=[r['route'] for r in rs],gap_mean=statistics.mean(r['min_person_proxy_gap_m'] for r in rs),gap_sd=statistics.stdev(r['min_person_proxy_gap_m'] for r in rs),path_mean=statistics.mean(r['path_m'] for r in rs),path_sd=statistics.stdev(r['path_m'] for r in rs),time_mean=statistics.mean(r['motion_s'] for r in rs),time_sd=statistics.stdev(r['motion_s'] for r in rs)))
(out/'summary.json').write_text(json.dumps(dict(episodes=9,scene_count=1,aggregate=aggregates,perception='same target-localizer prompt/model/taxonomy, fresh calls not identical outputs',source_rule='target-proxy score exact ties <=1e-9 broken by shared MPPI objective',scope='held-out remote target beyond navigation goal; all methods may meet target clearance; compare detour behavior, not target-class accuracy'),ensure_ascii=False,indent=2))
plt.rcParams.update({'font.size':7.2,'pdf.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=(7.2,3.9));fig.text(.08,.97,'距离场景：人员在终点之外，近处推车在另一侧',fontsize=9);fig.text(.08,.92,'上：真实目标与路线候选；下：经过箱子的实际关键帧与完整实测轨迹。',fontsize=7)
for col,method in enumerate(methods):
 base,rows,result,record=runs[(method,17)];left=.08+.307*col
 ax=fig.add_axes([left,.46,.275,.39]);ax.set_label(chr(97+col));ax.imshow(Image.open(base/'selection_overlay.jpg'));ax.axis('off');ax.set_title(f'{chr(97+col)} {names[col]}：{record["route"]}',fontsize=7.4,loc='left',pad=4)
 critical=min(rows,key=lambda r:abs(r['pose'][0]-5.4));meta=[json.loads(s) for s in (base/'overview_frames.jsonl').read_text().splitlines()];frame=min(meta,key=lambda r:abs(r['sim_time']-critical['timestamp']))
 ax=fig.add_axes([left,.095,.275,.31]);ax.set_label(chr(100+col));ax.imshow(Image.open(base/'overview_frames'/f"{frame['frame']:05d}.jpg"))
 poses=np.array([r['pose'] for r in rows]+[result['final_pose']]);f=1280/(2*math.tan(.55));ax.plot(640-f*(poses[:,0]-5.2)/8,360+f*(poses[:,1]+2.1)/8,color=colors[col],lw=1.45);ax.axis('off');ax.set_title(f'{chr(100+col)} 实际运动与路径',fontsize=7,loc='left',pad=4)
 fig.text(left,.06,f"{record['motion_s']:.2f} s；{record['path_m']:.2f} m；人员留距 {record['min_person_proxy_gap_m']:.2f} m",fontsize=6.4)
fig.text(.08,.012,'代表seed17；各3种子。定位提示与控制器共享，但定位返回不是逐次完全相同。',fontsize=6.2)
font='/tmp/ranger_delivery_figure_font.ttf'
for artist in fig.findobj(matplotlib.text.Text):
 props=artist.get_fontproperties().copy();props.set_file(font);artist.set_fontproperties(props)
fig.canvas.draw();require_matplotlib_panel_alignment(fig,json_out=out/'comparison.alignment.json',tolerance_pt=1.5,strict=True,row_groups=[['a','b','c'],['d','e','f']],column_groups=[['a','d'],['b','e'],['c','f']])
for ext in ['pdf','png','svg']:fig.savefig(out/f'comparison.{ext}',dpi=300)
translations={
 '距离场景：人员在终点之外，近处推车在另一侧':'Range case: person beyond the goal, nearby trolley on the other side',
 '上：真实目标与路线候选；下：经过箱子的实际关键帧与完整实测轨迹。':'Top: actual scene and candidates. Bottom: actual overhead frames and measured trajectories.',
 '代表seed17；各3种子。定位提示与控制器共享，但定位返回不是逐次完全相同。':'Representative seed17; three seeds each. Same locator prompt and controller, fresh locator calls.'}
for col,method in enumerate(methods):
 record=runs[(method,17)][3]
 translations[f'{chr(97+col)} {names[col]}：{record["route"]}']=f'{chr(97+col)} '+['Geometric MPPI','Task-aware image rule','Metric preference + MPPI'][col]+f': {record["route"]}'
 translations[f'{chr(100+col)} 实际运动与路径']=f'{chr(100+col)} Actual motion and measured route'
 translations[f"{record['motion_s']:.2f} s；{record['path_m']:.2f} m；人员留距 {record['min_person_proxy_gap_m']:.2f} m"]=f"{record['motion_s']:.2f} s; {record['path_m']:.2f} m; person gap {record['min_person_proxy_gap_m']:.2f} m"
for artist in fig.findobj(matplotlib.text.Text):
 if artist.get_text() in translations:artist.set_text(translations[artist.get_text()])
fig.canvas.draw();require_matplotlib_panel_alignment(fig,json_out=out/'comparison_en.alignment.json',tolerance_pt=1.5,strict=True,row_groups=[['a','b','c'],['d','e','f']],column_groups=[['a','d'],['b','e'],['c','f']])
fig.savefig(out/'comparison_en.pdf');fig.savefig(out/'comparison_en.svg');fig.savefig(out/'comparison_en.png',dpi=300)
plt.close(fig)
table='\n'.join(f"|{r['method']}|{r['arrived']}/3|{r['target_gap_met']}/3|{r['gap_mean']:.3f} ± {r['gap_sd']:.3f}|{r['time_mean']:.2f} ± {r['time_sd']:.2f}|{r['path_mean']:.3f} ± {r['path_sd']:.3f}|" for r in aggregates)
report=f'''# 远处任务对象的留距与绕行实验\n\n先冻结protocol.json，再实际运行9回合。人员位于(8.8,-.95)，在终点(7.25,-2.1)之外；推车保持在近处另一侧。比较几何选择、使用同一定位API与任务类别的图像侧别规则、深度评分方法。图像规则与深度方法使用新鲜定位调用，返回框不保证相同。只在目标评分相差<=1e-9时，以共有MPPI目标打破平局；不是修改避障器。\n\n|方法|到站|0.60m人员代理留距|最小留距 m，均值±SD|运动时间 s，均值±SD|采样路径 m，均值±SD|\n|---|---|---|---|---|---|\n{table}\n\n这一个静态场景的3规划种子只支持所示路线与效率差异。所有方法可能满足人员留距，不能把不同路线直接记为任务失败，不能将多次种子称为多种语义场景。表内时间从开始运动算起，含局部规划器停车，不含初始API等待。深度与图像规则的对比包含几何平局处理；不据此证明通用深度必要性或一般模型优势。原48回合与原失败不重算不替换。\n'''
(out/'REPORT_ZH.md').write_text(report)
(out/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>远处目标的语义绕行</title><style>body{max-width:1400px;margin:30px auto;font-family:sans-serif;line-height:1.7}img{width:100%}</style><h1>远处目标是否造成不必要绕行？</h1><p>左几何、中任务感知图像规则、右深度评分。全部原始轨迹保留。</p><img src="comparison.png"><p><a href="REPORT_ZH.md">全部9回合统计与边界</a>　<a href="episodes.csv">回合CSV</a></p></html>')
print(json.dumps(aggregates,ensure_ascii=False))

