#!/usr/bin/env python3
"""Build one evidence plate and real-video comparison from delivery pilot logs."""
import argparse
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import urllib.request

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

SKILL=Path('/home/USER/.codex/skills/nature-figure')
sys.path.insert(0,str(SKILL/'scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment
from summarize_unified_gazebo import static_rectangles,rectangle,polygon_distance

METHODS=['geometric','fixed','vlm']
NAMES={'geometric':'几何 MPPI','fixed':'固定规则 + MPPI','vlm':'DeepSeek + MPPI'}
COLOR={'geometric':'#66717b','fixed':'#4379a8','vlm':'#b5653c'}
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
PERSON=(4.9,-.95)


def clearance(pose,person=None):
    person=PERSON if person is None else person
    x,y,psi=pose;dx,dy=person[0]-x,person[1]-y
    bx=math.cos(psi)*dx+math.sin(psi)*dy;by=-math.sin(psi)*dx+math.cos(psi)*dy
    return max(0.,math.hypot(max(abs(bx)-.751/2,0),max(abs(by)-.500/2,0))-.18)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--person-side',choices=['north','south'],default='north')
    parser.add_argument('--figures-only',action='store_true')
    args=parser.parse_args();root=args.root.resolve()
    global PERSON
    PERSON=(4.9,-.95 if args.person_side=='north' else -3.30)
    run_directory='runs' if args.person_side=='north' else 'south_runs'
    out=root/('output' if args.person_side=='north' else 'output_south');out.mkdir(exist_ok=True)
    runs=[];provenance={};samples=[]
    for seed in [17,23,31]:
        for method in METHODS:
            base=root/run_directory/f'{method}_s{seed}'/'episode'
            result=json.loads((base/'result.json').read_text());manifest=json.loads((base/'manifest.json').read_text())
            rows=[json.loads(s) for s in (base/'trace.jsonl').read_text().splitlines()]
            if not rows:raise RuntimeError(f'no motion trace: {base}')
            assert manifest['config']['random_seed']==seed and manifest['config']['horizon']==36
            assert result['method']==method
            path=0.;last=(3.,-2.1)
            for row in rows:
                path+=math.dist(last,row['pose'][:2]);last=row['pose'][:2]
                samples.append(dict(method=method,seed=seed,elapsed_sim_s=row['elapsed'],x=row['pose'][0],y=row['pose'][1],yaw=row['pose'][2],person_proxy_clearance_m=clearance(row['pose'])))
            path+=math.dist(last,result['final_pose'][:2])
            clear=[clearance(r['pose']) for r in rows]+[clearance(result['final_pose'])]
            world=base.parent/'world.sdf';polygons,_=static_rectangles(world)
            # Evaluation-only scene geometry, never provided to the selector.
            min_static=float('inf')
            for row in rows:
                robot=rectangle(*row['pose'],.751,.500)
                min_static=min(min_static,min(polygon_distance(robot,p) for p in polygons))
            modes=[r['plan']['command_mode'] for r in rows]
            duration=result['elapsed_sim_s']
            record=dict(method=method,seed=seed,status=result['status'],route=result['route_id'],
                navigation_arrived=result['navigation_arrived'],elapsed_sim_s=duration,
                sampled_path_length_m=path,min_person_proxy_clearance_m=min(clear),
                clearance_requirement_m=.6,person_space_requirement_met=min(clear)>=.6,
                min_static_box_proxy_clearance_m=min_static,
                parallel_command_fraction=sum(m=='PARALLEL' for m in modes)/len(modes),
                vlm_latency_s=result['semantic'].get('latency_s'),api_called=result['api_called'],
                served_model=result['semantic'].get('served_model'))
            runs.append(dict(record=record,base=base,rows=rows,result=result,clearances=clear))
            for file in [base/'selection_rgb.jpg',base/'selection_overlay.jpg',base/'selection_input.json',base/'selection_output.json',base/'trace.jsonl',world]:
                provenance[str(file.relative_to(root))]=hashlib.sha256(file.read_bytes()).hexdigest()
    # Matched configurations/worlds across selectors; no trial is excluded.
    assert len({hashlib.sha256((r['base'].parent/'world.sdf').read_bytes()).hexdigest() for r in runs})==1
    agg=[]
    for method in METHODS:
        group=[r['record'] for r in runs if r['record']['method']==method]
        agg.append(dict(method=method,n=3,arrived=sum(r['navigation_arrived'] for r in group),
            space_met=sum(r['person_space_requirement_met'] for r in group),
            min_person_clearance_mean_m=statistics.mean(r['min_person_proxy_clearance_m'] for r in group),
            min_person_clearance_sd_m=statistics.stdev(r['min_person_proxy_clearance_m'] for r in group),
            elapsed_mean_s=statistics.mean(r['elapsed_sim_s'] for r in group),
            path_mean_m=statistics.mean(r['sampled_path_length_m'] for r in group)))
    records=[r['record'] for r in runs]
    summary=dict(evidence_type='actual_Gazebo_with_actual_DeepSeek_vision_calls',episodes=9,scene_count=1,
        n_definition='3 planner seeds per selector in the same static scene, not 3 independent semantic scenes',
        aggregate=agg,episodes_detail=records,complete_handover_measured=False,
        conclusion='Visual route selection is operational; fixed person-clearance rule matches VLM in this scene.')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    for filename,data in [('episodes.csv',records),('trajectory_samples.csv',samples)]:
        with (out/filename).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))

    # One figure question: can a scene-conditioned selector change a feasible
    # delivery route while the geometric execution backend remains identical?
    # Top: evidence/choice. Middle: actual motion. Bottom: measured consequence.
    font=font_manager.FontProperties(fname=FONT).get_name()
    # A standalone TrueType face avoids old Matplotlib's TTC/CFF embedding bug.
    standalone_font=Path('/tmp/ranger_delivery_figure_font.ttf')
    if not standalone_font.exists():
        variable=Path('/tmp/ranger_notosanssc_variable.ttf')
        if not variable.exists():urllib.request.urlretrieve(
            'https://raw.githubusercontent.com/google/fonts/main/ofl/notosanssc/NotoSansSC%5Bwght%5D.ttf',variable)
        face=instantiateVariableFont(TTFont(variable),{'wght':400},inplace=True)
        face.save(standalone_font)
    plt.rcParams.update({'font.family':font,'font.size':7.2,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6})
    fig=plt.figure(figsize=(7.2,3.9))
    fig.text(.08,.97,'资料配送途中绕行：同一场景、同一起终点、同一 MPPI 执行后端',fontsize=9,weight='bold')
    fig.text(.08,.92,'指令：经过箱子时避开站在旁边的同学，别从他身边挤过去。',fontsize=7.5)
    source_images=[]
    for col,method in enumerate(METHODS):
        run=next(r for r in runs if r['record']['method']==method and r['record']['seed']==17)
        base=run['base'];record=run['record'];left=.08+col*.307
        ax=fig.add_axes([left,.46,.275,.39]);ax.set_label(chr(97+col))
        ax.imshow(Image.open(base/'selection_overlay.jpg'));ax.axis('off')
        route='左侧' if record['route']=='LEFT' else '右侧'
        ax.set_title(f'{chr(97+col)}  {NAMES[method]}：选择{route}',fontsize=7.7,loc='left',pad=4)
        source_images.append(str((base/'selection_overlay.jpg').relative_to(root)))
        metadata=[json.loads(s) for s in (base/'overview_frames.jsonl').read_text().splitlines()]
        critical=min(run['rows'],key=lambda r:clearance(r['pose']))
        frame=min(metadata,key=lambda r:abs(r['sim_time']-critical['timestamp']))
        image_path=base/'overview_frames'/f"{frame['frame']:05d}.jpg"
        ax=fig.add_axes([left,.095,.275,.31]);ax.set_label(chr(100+col))
        ax.imshow(Image.open(image_path));f=1280/(2*math.tan(1.1/2))
        poses=np.array([r['pose'] for r in run['rows']]+[run['result']['final_pose']])
        u=640-f*(poses[:,0]-5.2)/8;v=360+f*(poses[:,1]+2.1)/8
        ax.plot(u,v,color=COLOR[method],lw=1.45)
        ax.scatter([u[0]],[v[0]],s=13,color=COLOR[method],marker='o')
        ax.scatter([u[-1]],[v[-1]],s=18,color=COLOR[method],marker='x',linewidths=.9)
        ax.axis('off');ax.set_title(f'{chr(100+col)}  实际俯视关键帧与全回合轨迹',fontsize=7,loc='left',pad=4)
        source_images.append(str(image_path.relative_to(root)))
        fig.text(left,.060,f"最近留距 {record['min_person_proxy_clearance_m']:.2f} m，{record['elapsed_sim_s']:.2f} s 到站",fontsize=6.8)
    fig.text(.08,.012,'代表回合 seed17；完整统计为每组3种子。VLM 与固定规则在此场景选择一致。',fontsize=6.2)
    for artist in fig.findobj(matplotlib.text.Text):
        props=artist.get_fontproperties().copy();props.set_file(str(standalone_font));artist.set_fontproperties(props)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=out/'comparison.alignment.json',tolerance_pt=1.5,strict=True,
        row_groups=[['a','b','c'],['d','e','f']],
        column_groups=[['a','d'],['b','e'],['c','f']])
    fig.savefig(out/'comparison.pdf')
    fig.savefig(out/'comparison.svg')
    fig.savefig(out/'comparison.png',dpi=300)
    fig.savefig(out/'comparison.tiff',dpi=600)
    translations={
        '资料配送途中绕行：同一场景、同一起终点、同一 MPPI 执行后端':'Document delivery: semantic choice among feasible MPPI route candidates',
        '指令：经过箱子时避开站在旁边的同学，别从他身边挤过去。':'Instruction: Pass the box without squeezing past the waiting student.',
        '代表回合 seed17；完整统计为每组3种子。VLM 与固定规则在此场景选择一致。':'Representative seed17; three planner seeds per selector. VLM and the person rule agree in this scene.'}
    english_names={'geometric':'Geometric MPPI','fixed':'Person rule + MPPI','vlm':'DeepSeek + MPPI'}
    for col,method in enumerate(METHODS):
        run=next(r for r in runs if r['record']['method']==method and r['record']['seed']==17)
        r=run['record'];route='左侧' if r['route']=='LEFT' else '右侧'
        translations[f'{chr(97+col)}  {NAMES[method]}：选择{route}']=f"{chr(97+col)}  {english_names[method]}: {r['route']}"
        translations[f'{chr(100+col)}  实际俯视关键帧与全回合轨迹']=f'{chr(100+col)}  Top-down frame and measured route'
        translations[f"最近留距 {r['min_person_proxy_clearance_m']:.2f} m，{r['elapsed_sim_s']:.2f} s 到站"]=f"Min. gap {r['min_person_proxy_clearance_m']:.2f} m; arrival {r['elapsed_sim_s']:.2f} s"
    for artist in fig.findobj(matplotlib.text.Text):
        if artist.get_text() in translations:artist.set_text(translations[artist.get_text()])
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=out/'comparison_en.alignment.json',tolerance_pt=1.5,strict=True,
        row_groups=[['a','b','c'],['d','e','f']],column_groups=[['a','d'],['b','e'],['c','f']])
    fig.savefig(out/'comparison_en.pdf');fig.savefig(out/'comparison_en.svg');fig.savefig(out/'comparison_en.png',dpi=300)
    plt.close(fig)
    (out/'figure_source_images.json').write_text(json.dumps(source_images,ensure_ascii=False,indent=2))
    if args.figures_only:return

    # Actual camera and overview video for each representative episode. Select
    # logged images on one common simulation-time grid; do not synthesize motion.
    videos=[];video_audit=[]
    for method in METHODS:
        run=next(r for r in runs if r['record']['method']==method and r['record']['seed']==17)
        base=run['base'];start=run['rows'][0]['timestamp']-run['rows'][0]['elapsed']
        streams=[]
        for directory,log in [('frames','frames.jsonl'),('overview_frames','overview_frames.jsonl')]:
            rows=[json.loads(s) for s in (base/log).read_text().splitlines()]
            streams.append((directory,rows,[r['sim_time'] for r in rows]))
        folder=out/f'{method}_video_frames';folder.mkdir(exist_ok=True)
        max_age=0.
        for i,t in enumerate(np.arange(0,run['record']['elapsed_sim_s']+.05,.1)):
            canvas=Image.new('RGB',(640,840),'white')
            for (directory,rows,times),y,size in zip(streams,[0,480],[(640,480),(640,360)]):
                j=max(0,bisect.bisect_right(times,start+t)-1);meta=rows[j]
                max_age=max(max_age,max(0,start+t-meta['sim_time']))
                image=Image.open(base/directory/f"{meta['frame']:05d}.jpg")
                if image.size!=size:image=image.resize(size,Image.LANCZOS)
                canvas.paste(image,(0,y))
            canvas.save(folder/f'{i:05d}.jpg',quality=90)
        video=out/f'{method}_s17_actual.mp4'
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate','10','-i',str(folder/'%05d.jpg'),'-c:v','libx264','-crf','22','-pix_fmt','yuv420p',str(video)],check=True)
        videos.append(video);video_audit.append(dict(method=method,video=video.name,max_logged_frame_age_s=max_age,
            encoding='10Hz simulation-time sampling of actual robot/overview image streams; no interpolation'))
        # The input images are retained in their source episode; remove only
        # the derived encoder frames created by this summarizer.
        for p in folder.glob('*.jpg'):p.unlink()
        folder.rmdir()
    (out/'video_timing_audit.json').write_text(json.dumps(video_audit,indent=2))
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y',*[arg for v in videos for arg in ['-i',str(v)]],
        '-filter_complex','[0:v][1:v][2:v]hstack=inputs=3:shortest=0[v]','-map','[v]','-c:v','libx264','-crf','22','-pix_fmt','yuv420p',str(out/'three_methods_actual.mp4')],check=True)
    table='\n'.join(f"| {NAMES[a['method']]} | {a['arrived']}/3 | {a['space_met']}/3 | {a['min_person_clearance_mean_m']:.3f} ± {a['min_person_clearance_sd_m']:.3f} | {a['elapsed_mean_s']:.2f} | {a['path_mean_m']:.3f} |" for a in agg)
    report=f'''# 资料配送途中语义选路：实际 Gazebo 预实验

## 一个问题与一个方法

**问题：**箱子两侧均有可执行路线，但靠近等待人员的一侧不符合“别从同学身边挤过去”的配送指令。
**方法：**共享的多模式 MPPI 从两条注册绕行引导路径生成可行候选；将当前候选投影到实际机器人 RGB，DeepSeek结合指令选择路由编号。所选路由由 MPPI 用当前激光与位姿持续重新求解，Ranger模式适配器负责实际执行。
行为路由 LEFT/RIGHT 与底盘 Ackermann/Parallel/Spin 是不同概念，模型不输出速度或强制蟹行。

## 本次结果

本图人员布局：{args.person_side}。9回合，三选择器各3个规划种子（17、23、31），全部实际到站。DeepSeek真实调用3次，实际候选返回与执行见episodes.csv。

| 选择器 | 到站 | 0.60m留距满足 | 最近人员代理净空 / m，均值±SD | 平均运动时间 / 仿真秒 | 平均采样轨迹长度 / m |
| --- | --- | --- | --- | --- | --- |
{table}

**这一场景证明视觉选路到物理运动已经贯通；没有证明VLM优于固定规则。** 固定规则是YOLO人检测+深度定位后选择留距更大的候选，不是刻意削弱的字符串规则。

## 对照与图像

`comparison.png/pdf/svg/tiff`为上下对应的长图：上排实际输入与所选路由，下排同回合实际俯视关键帧叠加完整实测轨迹，图下注明最近人员留距和到站时间。seed17为代表回合，所有9回合均在episodes.csv和trajectory_samples.csv中，没有择优排除。
`three_methods_actual.mp4`左至右为几何、固定规则、DeepSeek；每列上方为实际机器人相机，下方为实际俯视相机。按相同运动仿真时间取原始帧，不插值生成运动；较早结束列在最后原帧停留。

## 运行与边界

- 原家具化实验室SDF未改动。输出副本移动既有箱子、克隆站立人员、增加人员碰撞圆柱代理、调整机器人初始位姿和俯视相机。scene truth仅用于独立评估留距，不进入语义选路或在线避障。
- 起点(3.0,-2.1)，终点(7.25,-2.1)。左右路由为共享的注册几何引导，不是模型凭空生成的坐标。所有组均允许相同的行驶/横移/自旋候选。
- H=36、dt=0.12、每运动组16样本、速度上限0.40 m/s、禁止倒车。左右引导路径不完全对称（偏移0.65/0.80 m），因此时间/路径差异不能全部归因于VLM。
- DeepSeek只在出发前异步选一次路由，等待首次返回时停车；随后MPPI持续执行。这次未验证移动中语义更新。时间表是运动时间，不包含出发前API等待，API实测延迟单列于episodes.csv。
- 留距定义为0.751×0.500m车体矩形到人员半径0.18m圆柱代理的平面间隙；0.60m是试验前登记的任务要求，不是人因舒适度标准。独立几何评估只覆盖静态box/cylinder代理，不能宣称完整mesh碰撞或真实人身安全已验收。
- 仅一个静态场景、同一指令、3个规划种子；没有证明复杂视觉关系理解、语言泛化或跨布局优势。下一轮应改变人员位置/关系与指令检验语义选择。
- 相机和扫描是Gazebo实际传感器流；底盘是VelocityControl运动学代理，visual轮子，不是舵机/轮地动力学验收。这里是配送途中的导航阶段，不是完整资料交接成功。
- 父类遗留的view_visible_fraction/target_geometry_source字段不作为本试验指标；以本报告、selection_input/output、实测trajectory_samples为准。
- 根目录早期geometric_s17/fixed_s17/geometric_s17_h36为投影可见性排查记录，全部保留；最终9回合只取runs/。未把排查记录混入正式表。

## 可复现入口

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 src/rangermini_dynamic_semantic/scripts/run_delivery_route_pilot.py \\
  --selector vlm --seed 17 --output artifacts/delivery_route_next
```

VLM组要求进程环境已有DEEPSEEK_API_KEY。密钥不写入源码和记录；API失败不冒充固定规则结果。`--selector geometric`与`--selector fixed`无需API。
本批使用既有隔离安装`/tmp/ranger_sim_ws_install_consistency`的ROS适配器，规划模块直接导入当前源代码。

方法参照：VLM-MPPI §III，https://arxiv.org/html/2609.18451v1 。引用其候选投影与语义选择思想，不把原文方法或Ranger平台替换本身当作已证明的新颖贡献。
'''
    (out/'README_ZH.md').write_text(report)
    (out/'figure_contract.json').write_text(json.dumps(dict(question='Scene-conditioned choice among feasible delivery routes',
        archetype='image plate + quant',claim=summary['conclusion'],n=summary['n_definition'],
        dimensions_inches=[7.2,3.9],selected_backend='python',
        image_integrity='Raw RGB preserved; candidates are projected actual MPPI rollouts; actual trace overlaid on same-episode overhead photo. No generated scene content.',
        safety_scope='Evaluation cylinder and box proxies only'),indent=2))
    (out/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>资料配送语义选路</title><style>body{max-width:1400px;margin:30px auto;font-family:sans-serif;color:#233e52;line-height:1.7}img,video{width:100%}p{font-size:18px}</style><h1>一个问题：配送绕行时给等待人员留出空间</h1><p>左：几何MPPI；中：固定规则；右：真实DeepSeek视觉选择。9回合均到站。固定规则与VLM在本场景表现一致。</p><img src="comparison.png"><h2>实际Gazebo对照录像</h2><p>每列上方是机器人相机，下方是同回合俯视相机。按运动仿真时间对齐，结束后保持最后原帧。</p><video controls src="three_methods_actual.mp4"></video><p><a href="comparison.pdf">论文长图PDF</a>　<a href="README_ZH.md">设置与结果</a>　<a href="episodes.csv">全部回合CSV</a></p></html>''')
    print(json.dumps(dict(episodes=len(runs),aggregate=agg,output=str(out)),ensure_ascii=False))


if __name__=='__main__':main()
