#!/usr/bin/env python3
"""Consolidate the completed two-scene matrix and create the meeting PPTX."""
import argparse
import csv
import html
import json
from pathlib import Path
import subprocess
from PIL import Image
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor

NAMES={'geometric':'几何 MPPI','fixed':'固定规则 + MPPI','vlm':'DeepSeek + MPPI'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    root=args.root.resolve();out=root/'output'
    north=json.loads((out/'summary.json').read_text());south=json.loads((root/'output_south/summary.json').read_text())
    all_records=[]
    for scene,summary in [('north',north),('south',south)]:
        for r in summary['episodes_detail']:all_records.append(dict(scene=scene,**r))
    assert len(all_records)==18 and all(r['navigation_arrived'] for r in all_records)
    assert sum(r['api_called'] for r in all_records)==6
    consolidated=dict(episodes=18,scenes=2,methods=3,seeds_per_cell=3,real_visual_api_calls=6,
        arrived=18,records=all_records,full_handover=False,vlm_advantage_over_fixed_rule_demonstrated=False)
    (out/'complete_matrix.json').write_text(json.dumps(consolidated,ensure_ascii=False,indent=2))
    with (out/'complete_matrix.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_records[0]));writer.writeheader();writer.writerows(all_records)
    rows='\n'.join(f"| {'人员左侧' if scene=='north' else '人员右侧'} | {NAMES[a['method']]} | {a['arrived']}/3 | {a['space_met']}/3 | {a['min_person_clearance_mean_m']:.3f} ± {a['min_person_clearance_sd_m']:.3f} | {a['elapsed_mean_s']:.2f} |" for scene,s in [('north',north),('south',south)] for a in s['aggregate'])
    report=f'''# 配送途中的视觉语义绕行：完整交付

## 问题与方法

**一个问题：**配送途中箱子两侧都能通过，但“别从等待人员身边挤过去”这条要求需要选择与现场人员位置相符的路线。
**一个方法：**共享MPPI生成不同绕行候选，将实际候选投影到机器人真实相机图像；DeepSeek结合指令选择路由，MPPI再用当前几何连续求解并驱动Ranger。既有蟹行用于避障，不作为新底盘算法。

## 完整实验矩阵

两种人员位置×三选择器×三个规划种子，**18回合全部到站，6次真实DeepSeek视觉调用**。人员换到另一侧后，DeepSeek从RIGHT改选LEFT；几何选择始终LEFT；固定规则与DeepSeek在这两个简单场景中一致。

| 布局 | 方法 | 到站 | 0.60m留距满足 | 最近人员代理净空 / m，均值±SD | 平均运动时间 / 仿真秒 |
| --- | --- | --- | --- | --- | --- |
{rows}

统计单位为同一布局的3个规划随机种子，场景数2、指令数1，不是18种独立语义任务。VLM在出发前调用一次，表内运动时间不含API等待；逐回合API延迟、路径和执行记录在complete_matrix.csv。

## 直接使用的产物

- 本目录index.html：两布局的论文长图与三组实际录像。
- comparison.pdf/svg/tiff/png：人员左侧的上下对应长图。
- ../output_south/comparison.pdf/svg/tiff/png：人员右侧的对应长图。
- 配送语义绕行_组会汇报.pptx：6页组会材料，内嵌两段三组实际对照视频。
- complete_matrix.csv/json：完整18回合数据；每布局trajectory_samples.csv为逐周期实际位姿与人员代理留距。
- 原始相机、候选、API选择、当前激光规划日志、world副本在../runs/和../south_runs/，没有改动原家具场景。

## 据实结论

已形成当前图像、真实视觉模型选择、MPPI重新规划及实际Gazebo运动的闭环。两个布局中模型会根据图像变化改变绕行选择，人员左侧布局的几何路线留距不足。
**固定检测规则在本次同样有效，因此当前数据没有证明VLM优于固定规则，也不能将平台替换或复现VLM-MPPI本身当成新颖贡献。** 配送仅验证途中导航阶段，未测完整交接。
当前是VelocityControl车体运动学代理；轮子visual，未验收真实4WS舵角、轮地动力学或完整mesh碰撞。人员留距只对预先定义的圆柱/车体矩形评估，0.60m为协议要求，不能外推人因舒适度或实机安全。
机器人位姿由Gazebo模型位姿转为odom，不是独立定位/SLAM输出。旧manifest中的scene_truth_used_by_controller=false只表达未读取人员实体语义真值，不应理解成位姿不来自仿真真值；新生成器已改为明确记录semantic_entity_truth和localization_source。
左右注册引导偏移0.65/0.80m，不完全对称。单个候选初始预测可行，不等于整条注册路径已全局证明可行；执行中由当前扫描持续检查。全部路线、footprint、候选及参数在三组间共享。

## 一次完成运行

```bash
cd /home/USER/ranger_sim_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
# 进程环境应已配置 DEEPSEEK_API_KEY，不将密钥写入源码或报告。
python3 src/rangermini_dynamic_semantic/scripts/run_delivery_benchmark.py \\
  --output artifacts/delivery_benchmark_next
```

入口会依次完成18回合、两个布局的评估/长图/录像和合并组会材料；失败会保存日志并停止，`--resume`跳过完成回合并保留失败尝试。输出目录不可覆盖。
本批实际启动使用既有隔离ROS安装；导出器用系统NumPy/Matplotlib（PYTHONNOUSERSITE=1）避免用户NumPy2与系统绘图库冲突，未修改全局环境。

方法参照[VLM-MPPI §III](https://arxiv.org/html/2609.18451v1)。详细图像来源、排查记录及代理评估边界见README_ZH.md和../output_south/README_ZH.md。
'''
    (out/'COMPLETE_REPORT_ZH.md').write_text(report)
    prs=Presentation();prs.slide_width=Inches(13.333);prs.slide_height=Inches(7.5)
    notes=[]
    def text(s,content,x,y,w,h,size=23,color='233E52',bold=False):
        tf=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h)).text_frame
        tf.word_wrap=True;tf.margin_left=tf.margin_right=tf.margin_top=0
        for i,line in enumerate(content.split('\n')):
            p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line
            p.font.name='Noto Sans CJK SC';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=RGBColor.from_string(color);p.space_after=Pt(12)
    def slide(title,note):
        s=prs.slides.add_slide(prs.slide_layouts[6]);text(s,title,.6,.35,12.2,.65,32,bold=True)
        s.notes_slide.notes_text_frame.text=note;notes.append((title,note));return s
    def image(s,path):
        iw,ih=Image.open(path).size;scale=min(12.1/iw,5.95/ih);w,h=iw*scale,ih*scale
        s.shapes.add_picture(str(path),Inches(.6+(12.1-w)/2),Inches(1.15),width=Inches(w),height=Inches(h))
    s=slide('配送途中：语义要求怎样改变避障路线','来源：COMPLETE_REPORT_ZH.md和complete_matrix.csv。一个问题、一个方法。没有扩展任务记忆/跨层状态一致性研究。')
    text(s,'问题：两侧都能绕过箱子，但应给等待人员留出空间。',.65,1.6,12,1,30)
    text(s,'方法：MPPI 生成可行候选，DeepSeek 看图与指令选路，\n共享 MPPI 与 Ranger 适配器执行当前轨迹。',.65,3.15,12,1.6,28,color='4379A8')
    text(s,'2 种人员位置 × 3 种方法 × 3 个种子\n18 回合全部到站，6 次真实视觉 API 调用',.65,5.55,12,1.2,26)
    s=slide('人员在左侧：几何路线与语义路线分开','来源：runs/全部9回合；图示seed17为代表，不是最佳回合。上排当前真实机器人图像和初始MPPI候选，下排同回合实际俯视关键帧加完整实测轨迹。人员代理留距独立评估。')
    image(s,out/'comparison.png')
    s=slide('人员换到右侧：VLM 随画面改变选择','来源：south_runs/全部9回合，指令、目标、箱子和控制参数未改变，仅人员位置及朝向改变。几何方法仍LEFT，固定规则与VLM都改为LEFT。本例检验视觉条件变化，不能证明复杂语义理解。')
    image(s,root/'output_south/comparison.png')
    for title,folder in [('人员左侧：三组实际 Gazebo 录像',out),('人员右侧：三组实际 Gazebo 录像',root/'output_south')]:
        s=slide(title,'左中右依次几何、固定规则、DeepSeek。各列上方机器人相机，下方俯视相机。依照实际运动仿真时间取原始帧，无动画插值，结束后保持最后原帧。来源：三方法seed17原始frames和overview_frames日志。')
        poster=folder/'actual_video_poster.jpg'
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(folder/'three_methods_actual.mp4'),'-frames:v','1',str(poster)],check=True)
        s.shapes.add_movie(str(folder/'three_methods_actual.mp4'),Inches(.6),Inches(1.3),Inches(12.1),Inches(5.294),poster_frame_image=str(poster),mime_type='video/mp4')
        text(s,'左：几何 MPPI       中：固定规则       右：真实 DeepSeek 视觉选择',.65,6.85,12,.45,20)
    s=slide('完整矩阵结果与当前结论','来源：complete_matrix.csv/json；每布局每方法n=3规划种子。人员净空为车体矩形到圆柱代理的平面间隙，0.60m为预先定义的试验要求。没有完整交接、人因或实机安全结论。固定规则与VLM在本次一致，不能宣称VLM性能优势。')
    data=[['布局','选择器','到站','留距要求','最近净空 / m']]
    for scene,summary in [('人员左侧',north),('人员右侧',south)]:
        for a in summary['aggregate']:data.append([scene,NAMES[a['method']],f"{a['arrived']}/3",f"{a['space_met']}/3",f"{a['min_person_clearance_mean_m']:.3f} ± {a['min_person_clearance_sd_m']:.3f}"])
    shape=s.shapes.add_table(len(data),5,Inches(.65),Inches(1.45),Inches(12),Inches(4.25));t=shape.table
    for j,w in enumerate([2,3.5,1.5,2,3]):t.columns[j].width=Inches(w)
    for i,row in enumerate(data):
        for j,value in enumerate(row):
            c=t.cell(i,j);c.text=value;c.margin_top=c.margin_bottom=Inches(.06);c.fill.solid();c.fill.fore_color.rgb=RGBColor.from_string('233E52' if i==0 else 'FFFFFF')
            for p in c.text_frame.paragraphs:p.font.name='Noto Sans CJK SC';p.font.size=Pt(20);p.font.color.rgb=RGBColor.from_string('FFFFFF' if i==0 else '233E52')
    text(s,'视觉选路已接到实际运动。固定规则在本次同样有效。\n这是配送途中导航实验，尚未证明 VLM 优势或完整交接成功。',.65,6,12,1,22,color='B5653C')
    deck=out/'配送语义绕行_组会汇报.pptx';prs.save(deck)
    (out/'讲稿与来源.md').write_text('\n\n'.join(f'## {i+1}. {t}\n\n{n}' for i,(t,n) in enumerate(notes)))
    (out/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>配送语义绕行：完整交付</title><style>body{max-width:1500px;margin:30px auto;font-family:sans-serif;color:#233e52;line-height:1.7}img,video{width:100%}p{font-size:18px}</style><h1>配送途中给等待人员留出空间</h1><p>完整矩阵：2种人员位置、3种方法、每组3种子。18回合全部到站，6次真实DeepSeek调用。固定规则与VLM在本次表现一致。</p><p><a href="配送语义绕行_组会汇报.pptx">组会PPTX</a>　<a href="COMPLETE_REPORT_ZH.md">完整报告</a>　<a href="complete_matrix.csv">18回合数据</a></p><h2>人员在左侧</h2><img src="comparison.png"><p>左：几何MPPI；中：固定规则；右：DeepSeek。上排真实相机与候选，下排实际运动。</p><video controls src="three_methods_actual.mp4"></video><h2>人员换到另一侧</h2><img src="../output_south/comparison.png"><video controls src="../output_south/three_methods_actual.mp4"></video><p>资料交接未测；底盘为运动学代理。两张长图均有PDF/SVG/TIFF导出，所有原始日志保留。</p></html>''')
    if (out/'manuscript/vlm_route_manuscript.pdf').exists():
        page=(out/'index.html').read_text().replace('<h2>人员在左侧</h2>',
            '<p><a href="manuscript/vlm_route_manuscript.pdf">英文论文初稿（5页）</a>　<a href="comparison_en.pdf">英文长图：人员左侧</a>　<a href="../output_south/comparison_en.pdf">英文长图：人员右侧</a></p><h2>人员在左侧</h2>')
        (out/'index.html').write_text(page)
    print(json.dumps(dict(episodes=18,api_calls=6,deck=str(deck),report=str(out/'COMPLETE_REPORT_ZH.md')),ensure_ascii=False))


if __name__=='__main__':main()
