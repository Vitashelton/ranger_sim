"""Paired trajectories and service-pose outcomes from the frozen transition study."""
from pathlib import Path
import csv, hashlib, json, math, sys

import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src/rangermini_dynamic_semantic')]
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph

FIXED = '#62686C'
SEARCHED = '#9A7410'


def wrap_deg(angle):
    return abs(math.degrees(math.atan2(math.sin(angle), math.cos(angle))))


def main():
    summary_path = ROOT/'results/10_transition_target_pairs/summary.json'
    metrics_path = ROOT/'results/10_transition_target_pairs/metrics.csv'
    summary = json.loads(summary_path.read_text())
    records = list(csv.DictReader(metrics_path.open()))
    protocol = json.loads((ROOT/'artifacts/transition_target_pairs_20261027/task_pairs_v5/protocol.json').read_text())
    targets = {i: pose for i, pose in enumerate(protocol['goals'])}
    pairs = sorted({(int(r['target']), int(r['seed'])) for r in records})
    assert len(pairs) == 6 and len(records) == 12
    entry_error = {}
    traces = {}
    sources = [summary_path, metrics_path, ROOT/'artifacts/transition_target_pairs_20261027/task_pairs_v5/protocol.json']
    for r in records:
        key=(int(r['target']),r['method'],int(r['seed']))
        run=ROOT/r['run']
        trace_path=run/'episode/trace.jsonl'
        trace=[json.loads(line) for line in trace_path.read_text().splitlines()]
        traces[key]=trace
        goal=targets[key[0]]
        entry=next(row for row in trace if math.dist(row['pose'][:2],goal[:2]) <= .18)
        entry_error[key]=wrap_deg(entry['pose'][2]-goal[2])
        sources.extend([trace_path,run/'world.sdf'])

    fixed_run=ROOT/next(r['run'] for r in records if int(r['target'])==1 and r['method']=='fixed' and int(r['seed'])==53)
    graph=SpatialGraph(fixed_run/'world.sdf')
    limits=(10.7,13.35,-1.9,.95)
    frames=[]
    faster=sum(next(r for r in records if int(r['target'])==t and int(r['seed'])==s and r['method']=='hybrid')['elapsed_s'] <
               next(r for r in records if int(r['target'])==t and int(r['seed'])==s and r['method']=='fixed')['elapsed_s']
               for t,s in pairs)
    better_heading=sum(entry_error[(t,'hybrid',s)] < entry_error[(t,'fixed',s)] for t,s in pairs)
    rows_svg=[]
    for i,(target,seed) in enumerate(pairs,1):
        f=next(r for r in records if int(r['target'])==target and int(r['seed'])==seed and r['method']=='fixed')
        h=next(r for r in records if int(r['target'])==target and int(r['seed'])==seed and r['method']=='hybrid')
        rows_svg.append({'pair':i,'target':target,'seed':seed,'fixed_s':float(f['elapsed_s']),'hybrid_s':float(h['elapsed_s']),
                         'fixed_heading':entry_error[(target,'fixed',seed)],'hybrid_heading':entry_error[(target,'hybrid',seed)]})
    W,H=1450,680
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
         '<rect width="100%" height="100%" fill="white"/>',
         '<style>text{font-family:Arial,sans-serif;fill:#222}.small{font-size:17px}.label{font-size:20px}.head{font-size:24px;font-weight:600}.tick{font-size:16px}</style>']
    # Panel (a): actual recorded paths in the same map, with entry footprints.
    x0,y0,pw,ph=95,95,520,480
    svg += [f'<text x="{x0}" y="48" class="head">(a) Recorded paths and entry poses</text>',
            f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="#fff" stroke="#555" stroke-width="1.5"/>']
    def xy_map(x,y):
        return x0+(x-limits[0])/(limits[1]-limits[0])*pw, y0+ph-(y-limits[2])/(limits[3]-limits[2])*ph
    for box in graph.static_boxes:
        # geometry is already stored as axis-aligned boxes in world frame
        bx,by,bz,sx,sy=box
        xa,ya=xy_map(max(limits[0],bx-sx/2),min(limits[3],by+sy/2))
        xb,yb=xy_map(min(limits[1],bx+sx/2),max(limits[2],by-sy/2))
        if xb>xa and yb>ya:
            svg.append(f'<rect x="{xa:.1f}" y="{ya:.1f}" width="{xb-xa:.1f}" height="{yb-ya:.1f}" fill="#e1e3e4" stroke="#aeb3b5" stroke-width="1"/>')
    for method,color,dash in [('fixed',FIXED,'9,7'),('hybrid',SEARCHED,'')]:
        trace=traces[(1,method,53)]; pts=[xy_map(row['pose'][0],row['pose'][1]) for row in trace]
        path=' '.join((('M' if i==0 else 'L')+f'{x:.1f},{y:.1f}') for i,(x,y) in enumerate(pts))
        svg.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="4"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')
        entry=next(row for row in trace if math.dist(row['pose'][:2],targets[1][:2])<=.18)
        ex,ey=xy_map(entry['pose'][0],entry['pose'][1]); yaw=entry['pose'][2]
        # measured Ranger footprint rectangle and body heading.
        corners=[]
        for lx,ly in [(-.3755,-.25),(.3755,-.25),(.3755,.25),(-.3755,.25)]:
            wx=entry['pose'][0]+lx*math.cos(yaw)-ly*math.sin(yaw); wy=entry['pose'][1]+lx*math.sin(yaw)+ly*math.cos(yaw)
            corners.append(xy_map(wx,wy))
        pstr=' '.join(f'{x:.1f},{y:.1f}' for x,y in corners)
        svg.append(f'<polygon points="{pstr}" fill="white" fill-opacity=".78" stroke="{color}" stroke-width="3"/>')
        hx,hy=xy_map(entry['pose'][0]+.22*math.cos(yaw),entry['pose'][1]+.22*math.sin(yaw))
        svg.append(f'<line x1="{ex:.1f}" y1="{ey:.1f}" x2="{hx:.1f}" y2="{hy:.1f}" stroke="{color}" stroke-width="3"/>')
        frames.append({'method':method,'entry_pose':entry['pose'],'entry_time_s':entry['timestamp'],'entry_heading_deg':math.degrees(yaw),'entry_heading_error_deg':entry_error[(1,method,53)]})
    gx,gy=xy_map(*targets[1][:2]);svg.append(f'<path d="M{gx-9},{gy}h18m-9,-9v18" stroke="#222" stroke-width="2"/>')
    for i in range(5):
        xv=limits[0]+i*(limits[1]-limits[0])/4; px,_=xy_map(xv,limits[2]);svg.append(f'<line x1="{px:.1f}" y1="{y0+ph}" x2="{px:.1f}" y2="{y0+ph+6}" stroke="#444"/><text x="{px:.1f}" y="{y0+ph+27}" text-anchor="middle" class="tick">{xv:.1f}</text>')
        yv=limits[2]+i*(limits[3]-limits[2])/4;_,py=xy_map(limits[0],yv);svg.append(f'<line x1="{x0-6}" y1="{py:.1f}" x2="{x0}" y2="{py:.1f}" stroke="#444"/><text x="{x0-12}" y="{py+5:.1f}" text-anchor="end" class="tick">{yv:.1f}</text>')
    svg += [f'<text x="{x0+pw/2}" y="{y0+ph+57}" text-anchor="middle" class="label">x (m)</text>',
            f'<text x="{x0-63}" y="{y0+ph/2}" text-anchor="middle" class="label" transform="rotate(-90 {x0-63} {y0+ph/2})">y (m)</text>']
    # Matched outcome panels: all six actual trials, each pair joined.
    for panel,metric,title,x,y,w,h,xmin,xmax,xticks in [
        ('b','time','(b) Arrival time (s)',720,125,640,210,16,35,[16,20,24,28,32]),
        ('c','heading','(c) Heading error at target entry',720,420,640,210,0,28,[0,7,14,21,28])]:
        svg.append(f'<text x="{x}" y="{y-35}" class="head">{title}</text>')
        def yy(v):return y+h-(v-xmin)/(xmax-xmin)*h
        for tick in xticks:
            py=yy(tick);svg.append(f'<line x1="{x}" y1="{py:.1f}" x2="{x+w}" y2="{py:.1f}" stroke="#e4e4e4"/><text x="{x-12}" y="{py+5:.1f}" text-anchor="end" class="tick">{tick}</text>')
        svg.append(f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y+h}" stroke="#333"/><line x1="{x}" y1="{y+h}" x2="{x+w}" y2="{y+h}" stroke="#333"/>')
        for i,r in enumerate(rows_svg):
            px=x+38+i*(w-76)/5
            a=r['fixed_s'] if metric=='time' else r['fixed_heading']; b=r['hybrid_s'] if metric=='time' else r['hybrid_heading']
            svg.append(f'<line x1="{px}" y1="{yy(a):.1f}" x2="{px}" y2="{yy(b):.1f}" stroke="#b7b9b9" stroke-width="2"/>')
            svg.append(f'<circle cx="{px}" cy="{yy(a):.1f}" r="7" fill="{FIXED}"/><path d="M{px} {yy(b)-8}l8 8-8 8-8-8z" fill="{SEARCHED}"/>')
            svg.append(f'<text x="{px}" y="{y+h+24}" text-anchor="middle" class="tick">P{i+1}</text>')
    svg += [f'<circle cx="735" cy="665" r="6" fill="{FIXED}"/><text x="750" y="671" class="small">Fixed boundaries</text>',
            f'<path d="M950 657l7 7-7 7-7-7z" fill="{SEARCHED}"/><text x="967" y="671" class="small">Searched boundaries</text>',
            '</svg>']
    out=ROOT/'paper/figures/transition_pair_effects';out.parent.mkdir(exist_ok=True)
    svg_text='\n'.join(svg);out.with_suffix('.svg').write_text(svg_text)
    subprocess.run(['inkscape',str(out.with_suffix('.svg')),'--export-filename='+str(out.with_suffix('.pdf'))],check=True)
    subprocess.run(['inkscape',str(out.with_suffix('.svg')),'--export-filename='+str(out.with_suffix('.png')),'--export-width=2600'],check=True)
    prov={'kind':'Paired closed-loop transition-search outcomes from Gazebo logs',
          'pairing':'Three service-pose placements crossed with two sampler initializations; fixed and searched within each pair.',
          'panel_a':{'run_pair':'target1, initialization 53','pose_rule':'first logged pose within 0.18 m of target position',
                     'poses':frames},
          'panel_b':{'metric':'time to verified pose arrival','n_pairs':len(pairs),'searched_faster_pairs':faster,
                     'values':[{'pair':i,'fixed_s':float(next(r for r in records if int(r['target'])==t and int(r['seed'])==s and r['method']=='fixed')['elapsed_s']),
                                'searched_s':float(next(r for r in records if int(r['target'])==t and int(r['seed'])==s and r['method']=='hybrid')['elapsed_s'])} for i,(t,s) in enumerate(pairs,1)]},
          'panel_c':{'metric':'absolute heading error at first recorded pose within 0.18 m of target position','n_pairs':len(pairs),
                     'searched_lower_error_pairs':better_heading,
                     'values':[{'pair':i,'fixed_deg':entry_error[(t,'fixed',s)],'searched_deg':entry_error[(t,'hybrid',s)]} for i,(t,s) in enumerate(pairs,1)]},
          'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(sources))}}
    out.with_suffix('.provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
    print(json.dumps({'searched_faster_pairs':faster,'searched_lower_heading_error_pairs':better_heading,
                      'arrival_counts':summary['summary'],'entry_error':prov['panel_c']['values']},indent=2))

if __name__=='__main__':main()
