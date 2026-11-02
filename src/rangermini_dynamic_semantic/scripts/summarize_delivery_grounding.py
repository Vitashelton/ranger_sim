#!/usr/bin/env python3
"""All registered trials, task-target clearance, real image plates and movies."""
import argparse,bisect,csv,hashlib,json,math,statistics,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
sys.path.insert(0,'/home/USER/.codex/skills/nature-figure/scripts')
from audit_panel_alignment import require_matplotlib_panel_alignment
from summarize_unified_gazebo import rectangle,polygon_distance,static_rectangles
from summarize_delivery_route_pilot import clearance
METHODS=['geometric','fixed','vlm']
NAMES=['几何 MPPI','固定行人规则 + MPPI','DeepSeek + MPPI']
COLORS=['#66717b','#4379a8','#b5653c']

def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def target_gap(pose,side,target):
 py=-.95 if side=='north' else -3.30
 if target=='person':return clearance(pose,(4.9,py))
 cy=-3.30 if side=='north' else -.95
 return polygon_distance(rectangle(*pose,.751,.500),rectangle(4.9,cy,0.,.70,.42))

def movie(base,rows,result,out):
 streams=[];start=rows[0]['timestamp']-rows[0]['elapsed']
 for directory,log in [('frames','frames.jsonl'),('overview_frames','overview_frames.jsonl')]:
  metas=[json.loads(s) for s in (base/log).read_text().splitlines()];streams.append((directory,metas,[m['sim_time'] for m in metas]))
 folder=out.with_suffix('');folder.mkdir(exist_ok=False);max_age=0.
 for i,t in enumerate(np.arange(0,result['elapsed_sim_s']+.05,.1)):
  canvas=Image.new('RGB',(640,840),'white')
  for (directory,metas,times),y,size in zip(streams,[0,480],[(640,480),(640,360)]):
   j=max(0,bisect.bisect_right(times,start+t)-1);meta=metas[j];max_age=max(max_age,max(0,start+t-meta['sim_time']))
   image=Image.open(base/directory/f"{meta['frame']:05d}.jpg");canvas.paste(image.resize(size,Image.LANCZOS),(0,y))
  canvas.save(folder/f'{i:05d}.jpg',quality=90)
 subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate','10','-i',str(folder/'%05d.jpg'),'-c:v','libx264','-crf','22','-pix_fmt','yuv420p',str(out)],check=True)
 for f in folder.glob('*.jpg'):f.unlink()
 folder.rmdir();return max_age

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--grounded',action='store_true');a=p.parse_args();root=a.root.resolve();out=root/('output_grounded' if a.grounded else 'output');
 if a.grounded:METHODS[2]='grounded';NAMES[2]='任务对象定位 + MPPI'
 out.mkdir(exist_ok=True)
 protocol=json.loads((root/'preregistered_protocol.json').read_text());records=[];runs={};provenance={}
 for side in ['north','south']:
  worlds=set()
  for target in ['person','trolley']:
   for method in METHODS:
    for seed in [17,23,31]:
     name=f'{side}_{target}_{method}_s{seed}';base=root/'runs'/name/'episode'
     result=json.loads((base/'result.json').read_text());manifest=json.loads((base/'manifest.json').read_text())
     rows=[json.loads(s) for s in (base/'trace.jsonl').read_text().splitlines()]
     assert result['status']=='ARRIVED' and manifest['paired_objects'] and manifest['task_target']==target
     poses=[r['pose'] for r in rows]+[result['final_pose']];path=0.;last=(3.,-2.1)
     for pose in poses:path+=math.dist(last,pose[:2]);last=pose[:2]
     target_min=min(target_gap(pose,side,target) for pose in poses)
     polygons,_=static_rectangles(base.parent/'world.sdf')
     static_min=min(polygon_distance(rectangle(*pose,.751,.500),poly) for pose in poses for poly in polygons)
     record=dict(layout=side,target=target,method=method,seed=seed,route=result['route_id'],expected=protocol['expected_routes'][f'{side}_{target}'],arrived=result['navigation_arrived'],
      instruction_route_met=result['route_id']==protocol['expected_routes'][f'{side}_{target}'],min_target_proxy_gap_m=target_min,target_gap_criterion_met=target_min>=.60,
      elapsed_motion_s=result['elapsed_sim_s'],path_m=path,min_static_box_proxy_gap_m=static_min,
      api_called=result['api_called'],served_model=result['semantic'].get('served_model'))
     records.append(record);runs[name]=(base,rows,result,record)
     for file in [base/'result.json',base/'manifest.json',base/'selection_input.json',base/'selection_output.json',base/'selection_overlay.jpg',base/'trace.jsonl',base.parent/'world.sdf']:
      provenance[str(file.relative_to(root))]=hashlib.sha256(file.read_bytes()).hexdigest()
     worlds.add(hashlib.sha256((base.parent/'world.sdf').read_bytes()).hexdigest())
  assert len(worlds)==1,'Physics changed across tasks or methods within a layout'
 with (out/'episodes.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 aggregates=[]
 for side in ['north','south']:
  for target in ['person','trolley']:
   for method in METHODS:
    rs=[r for r in records if (r['layout'],r['target'],r['method'])==(side,target,method)]
    aggregates.append(dict(layout=side,target=target,method=method,n=len(rs),arrived=sum(r['arrived'] for r in rs),instruction_met=sum(r['instruction_route_met'] for r in rs),gap_met=sum(r['target_gap_criterion_met'] for r in rs),
      gap_mean=statistics.mean(r['min_target_proxy_gap_m'] for r in rs),gap_sd=statistics.stdev(r['min_target_proxy_gap_m'] for r in rs),time_mean=statistics.mean(r['elapsed_motion_s'] for r in rs),path_mean=statistics.mean(r['path_m'] for r in rs)))
 controls=[json.loads(f.read_text()) for f in sorted((root/'controls').glob('*.json')) if not f.name.endswith(('.input.json','.error.json'))]
 assert len(controls)==36
 control_summary=[]
 for cond in ['full','no_image','neutral_instruction']:
  calls=[c for c in controls if c['input']['condition']==cond]
  correct=sum(c['response']['decision']['candidate_id']==c['input']['expected_route'] for c in calls)
  flips=0
  for side in ['north','south']:
   for rep in range(3):
    pair=[c for c in calls if c['input']['layout']==side and c['input']['repeat']==rep]
    flips+=len({c['response']['decision']['candidate_id'] for c in pair})==2
  control_summary.append(dict(condition=cond,n=len(calls),matches_registered_route=correct,task_pair_route_flips=flips,pairs=6,
   interpretation='neutral instruction has no task-correctness ground truth; match and flip are diagnostic only' if cond=='neutral_instruction' else 'same image hash and candidate packet in each layout'))
 write(out/'summary.json',dict(physical_episodes=len(records),actual_motion_api_calls=sum(r['api_called'] for r in records),controlled_api_calls=len(controls),recorded_api_error_attempts=len(list((root/'controls').glob('*.error.json'))),aggregate=aggregates,controls=control_summary,
  scope='two static paired-object layouts, two instructions, three planning seeds; not semantic-scene generalization or steering dynamics'))
 write(out/'provenance.json',provenance)
 plt.rcParams.update({'font.size':7.2,'pdf.fonttype':42,'svg.fonttype':'none'})
 font=Path('/tmp/ranger_delivery_figure_font.ttf')
 if not font.exists():
  import urllib.request
  from fontTools.ttLib import TTFont
  from fontTools.varLib.instancer import instantiateVariableFont
  variable=Path('/tmp/ranger_notosanssc_variable.ttf')
  if not variable.exists():urllib.request.urlretrieve('https://raw.githubusercontent.com/google/fonts/main/ofl/notosanssc/NotoSansSC%5Bwght%5D.ttf',variable)
  instantiateVariableFont(TTFont(variable),{'wght':400},inplace=True).save(font)
 panels=[];video_audit=[]
 for side in ['north','south']:
  for target in ['person','trolley']:
   stem=f'{side}_{target}';fig=plt.figure(figsize=(7.2,3.9));target_name='同学' if target=='person' else '器材推车'
   fig.text(.08,.97,f'同布局换指令：配送经过箱子时，远离{target_name}一侧',fontsize=9)
   fig.text(.08,.92,'上：真实相机与候选；下：同回合俯视关键帧与实测轨迹。几何与执行配置相同。',fontsize=7)
   videos=[]
   for col,method in enumerate(METHODS):
    base,rows,result,record=runs[f'{side}_{target}_{method}_s17'];left=.08+col*.307
    ax=fig.add_axes([left,.46,.275,.39]);ax.set_label(chr(97+col));top_image=Image.open(base/'selection_overlay.jpg').copy()
    if method=='grounded':
     response=json.loads((base/'selection_output.json').read_text());box=response['evidence']['bbox'];xy=[box[0]*640,box[1]*480,box[2]*640,box[3]*480]
     ImageDraw.Draw(top_image).rectangle(xy,outline=(54,116,172),width=3);top_image.save(base/'grounding_overlay.jpg',quality=95)
    ax.imshow(top_image);ax.axis('off')
    ax.set_title(f'{chr(97+col)}  {NAMES[col]}：{record["route"]}',loc='left',fontsize=7.4,pad=4)
    critical=min(rows,key=lambda r:target_gap(r['pose'],side,target));metas=[json.loads(s) for s in (base/'overview_frames.jsonl').read_text().splitlines()]
    meta=min(metas,key=lambda r:abs(r['sim_time']-critical['timestamp']));im=base/'overview_frames'/f"{meta['frame']:05d}.jpg"
    ax=fig.add_axes([left,.095,.275,.31]);ax.set_label(chr(100+col));ax.imshow(Image.open(im))
    poses=np.array([r['pose'] for r in rows]+[result['final_pose']]);f=1280/(2*math.tan(.55));u=640-f*(poses[:,0]-5.2)/8;v=360+f*(poses[:,1]+2.1)/8
    ax.plot(u,v,lw=1.45,color=COLORS[col]);ax.axis('off');ax.set_title(f'{chr(100+col)}  实际运动；要求'+('满足' if record['instruction_route_met'] else '未满足'),loc='left',fontsize=7,pad=4)
    fig.text(left,.06,f"目标代理留距 {record['min_target_proxy_gap_m']:.2f} m；{record['elapsed_motion_s']:.2f} s",fontsize=6.8)
    video=out/f'{stem}_{method}_s17_actual.mp4'
    if not video.exists():age=movie(base,rows,result,video);video_audit.append(dict(video=video.name,max_frame_age_s=age))
    videos.append(video)
   fig.text(.08,.012,'代表 seed17；每格3种子。固定策略只远离行人，未宣称优于任务感知规则。',fontsize=6.2)
   for artist in fig.findobj(matplotlib.text.Text):
    props=artist.get_fontproperties().copy();props.set_file(str(font));artist.set_fontproperties(props)
   fig.canvas.draw();require_matplotlib_panel_alignment(fig,json_out=out/f'{stem}.alignment.json',strict=True,tolerance_pt=1.5,row_groups=[['a','b','c'],['d','e','f']],column_groups=[['a','d'],['b','e'],['c','f']])
   for ext in ['pdf','svg','png']:fig.savefig(out/f'{stem}.{ext}',dpi=300)
   english={
    f'同布局换指令：配送经过箱子时，远离{target_name}一侧':f'Same layout, different instruction: keep away from the {target}',
    '上：真实相机与候选；下：同回合俯视关键帧与实测轨迹。几何与执行配置相同.':'',
    '上：真实相机与候选；下：同回合俯视关键帧与实测轨迹。几何与执行配置相同。':'Top: actual RGB and candidates. Bottom: actual overhead frame and measured trajectory.',
    '代表 seed17；每格3种子。固定策略只远离行人，未宣称优于任务感知规则。':'Representative seed17; three seeds per cell. Fixed policy avoids people, independent of instruction.'}
   for col,method in enumerate(METHODS):
    record=runs[f'{side}_{target}_{method}_s17'][3]
    enname=['Geometric MPPI','Person rule + MPPI',('Grounded target + MPPI' if a.grounded else 'DeepSeek + MPPI')][col]
    english[f'{chr(97+col)}  {NAMES[col]}：{record["route"]}']=f'{chr(97+col)}  {enname}: {record["route"]}'
    english[f'{chr(100+col)}  实际运动；要求'+('满足' if record['instruction_route_met'] else '未满足')]=f'{chr(100+col)}  Measured motion; preference '+('met' if record['instruction_route_met'] else 'unmet')
    english[f"目标代理留距 {record['min_target_proxy_gap_m']:.2f} m；{record['elapsed_motion_s']:.2f} s"]=f"Target gap {record['min_target_proxy_gap_m']:.2f} m; {record['elapsed_motion_s']:.2f} s"
   for artist in fig.findobj(matplotlib.text.Text):
    if artist.get_text() in english:artist.set_text(english[artist.get_text()])
   fig.canvas.draw();require_matplotlib_panel_alignment(fig,json_out=out/f'{stem}_en.alignment.json',strict=True,tolerance_pt=1.5,row_groups=[['a','b','c'],['d','e','f']],column_groups=[['a','d'],['b','e'],['c','f']])
   fig.savefig(out/f'{stem}_en.pdf');fig.savefig(out/f'{stem}_en.svg');fig.savefig(out/f'{stem}_en.png',dpi=300)
   plt.close(fig)
   video=out/f'{stem}_three_methods.mp4'
   subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y',*[arg for v in videos for arg in ['-i',str(v)]],'-filter_complex','[0:v][1:v][2:v]hstack=inputs=3:shortest=0[v]','-map','[v]','-c:v','libx264','-crf','22','-pix_fmt','yuv420p',str(video)],check=True)
   panels.append(f'<h2>{side}布局：远离{target_name}</h2><img src="{stem}.png"><video controls preload="metadata" src="{video.name}"></video>')
 write(out/'video_audit.json',video_audit)
 table='\n'.join(f"|{r['layout']}|{r['target']}|{r['method']}|{r['arrived']}/3|{r['instruction_met']}/3|{r['gap_mean']:.3f} ± {r['gap_sd']:.3f}|{r['time_mean']:.2f}|{r['path_mean']:.3f}|" for r in aggregates)
 ctrl='\n'.join(f"|{r['condition']}|{r['matches_registered_route']}/{r['n']}|{r['task_pair_route_flips']}/{r['pairs']}|" for r in control_summary)
 report=f'''# 配送绕行的指令与视觉交叉实验

## 问题与方法
同一箱子两侧都可通行，但远离等候同学与给器材推车留操作空间对应不同路线。
方法仍为共享MPPI候选投影、DeepSeek结合RGB与配送指令返回可行路由ID、实时几何MPPI执行；本轮移除API候选里的数值净空，仅传ID、颜色、可行性，避免用几何数字冒充视觉依据。
复用已有实体实验室、人员、推车与控制器。原world不变，输出副本移动推车组合。36回合：两布局×两指令×三选择器×三规划种子；另36次取得有效返回的真实API控制调用；API失败尝试单独保留，不算路线。模型未获目标的真值位置或正确路由标签。

## 完整运动结果
|布局|任务目标|选择器|到站|指令侧满足|目标代理最小留距 m，均值±SD|运动时间 s|采样路径 m|
|---|---|---|---|---|---|---|---|
{table}

## 同图控制调用
每个布局均复用同一张实际RGB与完全相同的候选包，两条指令各重复3次。没有图时没有对象位置输入；中性指令没有指定避让对象。
|输入|与登记任务侧相符|换任务后路线翻转|
|---|---|---|
{ctrl}
中性指令项不是任务正确率，只显示在没有指定对象时，结果是否仍呈现任务相关分化。小样本、静态两布局；不据此宣称一般视觉因果机制或语言泛化。

## 解释边界
- 固定基线是完整保留的YOLO+RGBD行人留距策略；它不接受器材优先指令。此对照检验指定策略的任务适应性，不能证明VLM优于加入推车检测与任务分支的规则系统。人工标注对象类别与位置的任务感知规则可给出同样的期望侧，不作为感知同等的基线。
- 0.60m是预先登记的目标留距判据，不是所有人/设备的通用安全标准。人员采用半径0.18m圆柱、推车采用0.70×0.42m包围矩形；这是评估代理净空，非完整mesh/接触验证。
- 物理布局在同一侧条件的所有方法、指令和种子间SHA256一致。左右引导偏移不完全对称，耗时差不全部归因于语义。
- 一次出发前真实API选择，等待时停车；执行过程中MPPI持续重算。不宣称移动中的语义更新、Nav2贡献、转向执行器验收、实机或完整资料交接。
- 新实验能够补充指令依赖和图像输入对照，候选投影与ID选择思想仍参照VLM-MPPI；尚不能把工程迁移本身写成算法原创。

## 文件与复现
完整记录episodes.csv，调用与输入在../controls/，原始逐帧图像与轨迹在../runs/，协议在../preregistered_protocol.json。图选seed17而非择优，视频直接取同回合原始帧，上相机下俯视，不生成运动。
运行：先source ROS与本工作区install，在进程环境export密钥，然后运行run_delivery_grounding_benchmark.py --output 新目录；聚合运行summarize_delivery_grounding.py --root 该目录。密钥不写入文件。
'''
 if a.grounded:
  report=report.replace('方法仍为共享MPPI候选投影、DeepSeek结合RGB与配送指令返回可行路由ID、实时几何MPPI执行；本轮移除API候选里的数值净空，仅传ID、颜色、可行性，避免用几何数字冒充视觉依据。','方法修订为：DeepSeek在原始RGB定位指定任务对象的可见框，当前深度将框内最近20%有效深度点投到平面，得到可见表面深度锚点（不是已标定对象中心）；确定性代码在每条共享引导的0.05m采样点上计算保守目标净空，选择间隙更大的当前可行候选。随后同一MPPI执行。VLM不决定速度或世界坐标。仅支持预定义person/trolley任务类别，不能声称任意自由指令解析。对象缺失或深度不足时停止，不隐式退回规则。')
  report=report.replace('36回合：两布局×两指令×三选择器×三规划种子；','本表36回合：复用24个既有几何/固定基线回合，另跑12个对象定位方法回合；加上原直接选路12回合，总共48个独立实际运动记录。')
  report=report.replace('## 同图控制调用','## 原直接选路接口的同图控制调用（不是定位方法消融）')
  report+='\n## 针对原失败的修订\n原直接VLM在south/trolley/seed31将人员所在侧误称器材区。此回合保留不删。新方法将语义输出限制为目标可见框，再用实际深度与确定性路线净空推导选路；定位框、深度ROI、三维投影与路线评分均保留在对应episode中。其对照是同布局同指令同控制器的原直接选路接口，不据此证明通用视觉理解或所有定位误差已解决。\n'
 (out/'REPORT_ZH.md').write_text(report)
 html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>配送语义交叉实验</title><style>body{max-width:1400px;margin:30px auto;font-family:sans-serif;line-height:1.7}img,video{width:100%}td,th{padding:5px;border:1px solid #ddd}</style><h1>一个问题：同一路况，配送约束变了，该走哪一侧？</h1><p>四种条件的真实Gazebo对照。左：几何MPPI；中：固定行人留距规则；右：'+('真实DeepSeek任务对象定位' if a.grounded else '真实DeepSeek直接选路')+'。</p><p><a href="REPORT_ZH.md">完整结果与边界</a>　<a href="episodes.csv">36回合CSV</a>　<a href="summary.json">运动与消融统计</a></p>'+''.join(panels)+'</html>'
 (out/'index.html').write_text(html);print(json.dumps(dict(output=str(out),episodes=len(records),controls=control_summary),ensure_ascii=False))
if __name__=='__main__':main()
