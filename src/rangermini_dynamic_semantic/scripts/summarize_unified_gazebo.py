#!/usr/bin/env python3
"""Independent geometry diagnostics and episode-level result tables."""
import argparse
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET


def rectangle(x,y,yaw,length,width):
    c,s = math.cos(yaw),math.sin(yaw)
    return [(x+c*a-s*b,y+s*a+c*b) for a,b in
            [(-length/2,-width/2),(length/2,-width/2),(length/2,width/2),(-length/2,width/2)]]


def segment_distance(point,a,b):
    dx,dy=b[0]-a[0],b[1]-a[1]
    projection=max(0,min(1,((point[0]-a[0])*dx+(point[1]-a[1])*dy)/(dx*dx+dy*dy)))
    return math.hypot(point[0]-a[0]-projection*dx,point[1]-a[1]-projection*dy)


def polygon_distance(a,b):
    separated=False
    for polygon in (a,b):
        for i in range(4):
            p,q=polygon[i],polygon[(i+1)%4]
            axis=(q[1]-p[1],p[0]-q[0])
            ia=[x*axis[0]+y*axis[1] for x,y in a]
            ib=[x*axis[0]+y*axis[1] for x,y in b]
            if max(ia)<min(ib) or max(ib)<min(ia):
                separated=True
    if not separated:
        return 0.
    return min(segment_distance(p,poly[i],poly[(i+1)%4])
               for points,poly in ((a,b),(b,a)) for p in points for i in range(4))


def static_rectangles(world):
    def pose(element):
        return list(map(float,(element.findtext('pose') or '0 0 0 0 0 0').split()))
    polygons=[]
    dynamic=None
    for m in ET.parse(world).getroot().find('world').findall('model'):
        if m.get('name')=='rangermini_2_0':
            continue
        pm=pose(m)
        for link in m.findall('link'):
            pl=pose(link)
            for collision in link.findall('collision'):
                size=collision.findtext('geometry/box/size')
                if size is None:
                    continue
                size=list(map(float,size.split()))
                pc=pose(collision)
                z=pm[2]+pl[2]+pc[2]
                if z-size[2]/2>.23 or z+size[2]/2<.03:
                    continue
                xl=pl[0]+math.cos(pl[5])*pc[0]-math.sin(pl[5])*pc[1]
                yl=pl[1]+math.sin(pl[5])*pc[0]+math.cos(pl[5])*pc[1]
                x=pm[0]+math.cos(pm[5])*xl-math.sin(pm[5])*yl
                y=pm[1]+math.sin(pm[5])*xl+math.cos(pm[5])*yl
                poly=rectangle(x,y,pm[5]+pl[5]+pc[5],*size[:2])
                if m.get('name')=='pedestrian_proxy':
                    dynamic=(poly,size[:2])
                else:
                    polygons.append(poly)
    return polygons,dynamic


def summarize(root):
    rows=[]
    for path in sorted(root.glob('*/episode/result.json')):
        episode=path.parent
        work=episode.parent
        result=json.loads(path.read_text())
        manifest=json.loads((episode/'manifest.json').read_text())
        scene=json.loads((work/'scene_manifest.json').read_text())
        trace=[json.loads(s) for s in (episode/'trace.jsonl').read_text().splitlines()]
        plans=[row['plan'] for row in trace if 'plan' in row]
        if not plans:
            continue
        times=sorted(p['planning_time_ms'] for p in plans)
        result['planning_median_ms']=statistics.median(times)
        result['planning_p95_ms']=times[math.ceil(.95*len(times))-1]
        result['planning_calls']=len(times)
        result['negative_vx_commands']=sum(p['action'][0]<-1e-9 for p in plans)
        result['condition']=scene['condition']
        result['sources']=manifest['sources']
        result['world_sha256']=manifest['world_sha256']
        polygons,dynamic=static_rectangles(work/'world.sdf')
        events=[]
        eventfile=work/'event.log'
        if eventfile.exists():
            for line in eventfile.read_text().splitlines():
                try:
                    event=json.loads(line)
                    if event.get('acknowledged'):
                        events.append(event)
                except ValueError:
                    pass
        events.sort(key=lambda e:e.get('ack_sim_time',e['sim_time']))
        clearance=float('inf')
        overlap_ticks=0
        for row in trace:
            x,y,yaw=row['pose']
            body=rectangle(x,y,yaw,.751,.500)
            current=list(polygons)
            if dynamic is not None:
                prior=[e for e in events if e.get('ack_sim_time',e['sim_time'])<=row['timestamp']]
                if prior:
                    event=prior[-1]
                    current.append(rectangle(event['x'],event['y'],0,*dynamic[1]))
            d=min(polygon_distance(body,p) for p in current)
            clearance=min(clearance,d)
            overlap_ticks+=int(d==0)
        result['minimum_static_geometry_clearance_m']=clearance if not events else None
        result['minimum_geometry_clearance_m']=clearance if dynamic is None or events else None
        result['geometry_overlap_ticks']=overlap_ticks if dynamic is None or events else None
        result['dynamic_pose_reconstruction']='acknowledged_set_pose_log' if events else ('missing' if dynamic else 'not_applicable')
        result['physical_contact_measurement']=False
        result['marker_visible_frames']=sum(row.get('pixels',0)>=80 for row in trace)
        result['directory']=str(work)
        rows.append(result)
    (root/'audited_results.json').write_text(json.dumps(rows,indent=2))
    table=['# Gazebo 闭环运行汇总','',
      '代码标记确认任务；不评测 VLM/LLM 识别质量。净空由独立 SDF 几何重建，重叠不是物理接触传感器测量。', '',
      '|任务|条件|seed|方法|结果|确认/终止(s)|路径(m)|模式切换|规划P95(ms)|最小几何净空(m)|',
      '|---|---|---:|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        clearance_text=f"{row['minimum_geometry_clearance_m']:.3f}" if row['minimum_geometry_clearance_m'] is not None else '未记录动态位姿'
        table.append(f"|{row['task']}|{row['condition']}|{row['seed']}|{row['method']}|{row['status']}|{row['elapsed_sim_s']:.2f}|{row['path_length_m']:.3f}|{row['active_mode_switches']}|{row['planning_p95_ms']:.2f}|{clearance_text}|")
    table+=['','每回合保留 manifest、源码 hash、world hash、trace、result 和最后一帧相机图像。']
    (root/'RESULTS_ZH.md').write_text('\n'.join(table)+'\n')
    print(json.dumps({'episodes':len(rows),'succeeded':sum(r['status']=='SUCCEEDED' for r in rows),
                      'geometry_overlap_episodes':sum((r['geometry_overlap_ticks'] or 0)>0 for r in rows)}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root')
    args=p.parse_args()
    summarize(Path(args.root))
