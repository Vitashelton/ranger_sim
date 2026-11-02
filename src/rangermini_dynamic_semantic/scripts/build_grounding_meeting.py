#!/usr/bin/env python3
import argparse,json,subprocess
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--grounded',action='store_true');a=p.parse_args();out=a.root.resolve()/('output_grounded' if a.grounded else 'output');data=json.loads((out/'summary.json').read_text())
 prs=Presentation();prs.slide_width=Inches(13.333);prs.slide_height=Inches(7.5)
 def text(slide,content,x,y,w,h,size=24):
  tf=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h)).text_frame;tf.word_wrap=True
  for i,line in enumerate(content.split('\n')):
   p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.font.name='Noto Sans CJK SC';p.font.size=Pt(size);p.font.color.rgb=RGBColor.from_string('233E52');p.space_after=Pt(10)
 def slide(title):
  sl=prs.slides.add_slide(prs.slide_layouts[6]);text(sl,title,.5,.2,12.3,.65,29);return sl
 sl=slide('一个问题：同一路况，配送约束改变后应走哪一侧？')
 text(sl,'服务任务：把资料送到实验室，途中绕过走廊箱子。\n要求 A：避开等候同学，减少打扰。\n要求 B：远离器材推车，给器材取用留空间。\n两侧都能通行；目标位置本身无法表达这两种偏好。',.65,1.2,12,3.5,27)
 text(sl,('明确方法：DeepSeek 在 RGB 定位任务对象 → 实测深度锚点 → 共享候选目标留距评分 → 当前激光 MPPI 执行。' if a.grounded else '明确方法：共享 MPPI 生成候选 → 真实 RGB 叠加轨迹 → DeepSeek 结合指令选择路由 ID → MPPI 用当前几何执行。'),.65,5.25,12,1.4,25)
 sl.notes_slide.notes_text_frame.text='不是新底盘控制算法；候选投影与语义选择参照VLM-MPPI。DeepSeek不输出速度。所有图像来自Gazebo原始记录。'
 for side in ['north','south']:
  for target in ['person','trolley']:
   label='避让同学' if target=='person' else '给推车留操作空间';stem=f'{side}_{target}'
   sl=slide(f'{side}布局：{label}');sl.shapes.add_picture(str(out/f'{stem}.png'),Inches(.9),Inches(1),width=Inches(11.5))
   sl.notes_slide.notes_text_frame.text='代表回合seed17；每条件3种子。上下同回合，图中轨迹来自实测位姿。全部数据见episodes.csv；非动画示意。'
 sl=slide('主方法对照：复用24个基线 + 12个对象定位回合' if a.grounded else '完整运动结果：36回合，所有回合保留')
 rows=data['aggregate'];shape=sl.shapes.add_table(len(rows)+1,6,Inches(.55),Inches(1.05),Inches(12.2),Inches(5.5));table=shape.table
 headers=['布局/目标','方法','到站','任务侧','目标间隙 m ± SD','运动时间 s']
 for col,h in enumerate(headers):table.cell(0,col).text=h
 for row,r in enumerate(rows,1):
  vals=[r['layout']+'/'+r['target'],r['method'],f"{r['arrived']}/3",f"{r['instruction_met']}/3",f"{r['gap_mean']:.3f} ± {r['gap_sd']:.3f}",f"{r['time_mean']:.2f}"]
  for col,value in enumerate(vals):table.cell(row,col).text=value
 for row in table.rows:
  for cell in row.cells:
   for p in cell.text_frame.paragraphs:p.font.name='Noto Sans CJK SC';p.font.size=Pt(16)
 text(sl,'统计单位：同一静态条件的3个规划种子；目标间隙是评估代理，不是完整mesh或实机安全验收。',.6,6.75,12,.4,17)
 sl=slide('原直接选路接口的同图消融' if a.grounded else '同图消融：语言和图像各起什么作用？')
 lines=[]
 for c in data['controls']:lines.append(f"{c['condition']}：登记侧匹配 {c['matches_registered_route']}/{c['n']}；换指令路线翻转 {c['task_pair_route_flips']}/6")
 text(sl,'两布局 × 两指令 × 三种输入 × 三次重复，共36次真实API调用。\n每个布局复用相同RGB与相同候选包；候选不包含数值净空。\n'+'\n'.join(lines),.65,1.3,12,3.8,26)
 text(sl,'固定基线是行人留距策略；不能据此宣称VLM优于任务感知规则。中性指令没有指定目标，其匹配数不是任务正确率。',.65,5.35,12,1.25,22)
 for side in ['north','south']:
  for target in ['person','trolley']:
   stem=f'{side}_{target}';video=out/f'{stem}_three_methods.mp4';poster=out/f'{stem}_poster.jpg'
   subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-ss','2','-i',str(video),'-frames:v','1',str(poster)],check=True)
   sl=slide(f'真实录像：{side} / {target}（左几何、中固定规则、右VLM）')
   sl.shapes.add_movie(str(video),Inches(.4),Inches(1.15),Inches(12.5),Inches(5.47),poster_frame_image=str(poster),mime_type='video/mp4')
   sl.notes_slide.notes_text_frame.text='原始相机上、同回合俯视下；按运动仿真时间取原帧，不插值合成运动。'
 file=out/'配送指令交叉实验_组会汇报.pptx';prs.save(file)
 html=out/'index.html';s=html.read_text();s=s.replace('</h1>','</h1><p><a href="'+file.name+'">组会PPTX（内嵌四段实际录像）</a></p>',1);html.write_text(s)
 print(file)
if __name__=='__main__':main()
