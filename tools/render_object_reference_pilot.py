"""Counterbalanced real observations and measured object-directed responses."""
import hashlib
import json
from pathlib import Path
import sys
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]

def main():
    batch=ROOT/'artifacts/object_reference_20261026/pilot_v1'
    out=ROOT/'paper/figures/object_reference_20261026';out.mkdir(exist_ok=True)
    metrics=json.loads((batch/'analysis/execution_metrics.json').read_text())
    probes=json.loads((batch/'analysis/grounding_probes.json').read_text())
    rows=[]
    for scene in ('object_reference_a','object_reference_b'):
        for request in ('red','blue'):
            row=next(r for r in metrics['metrics'] if r['scene']==scene and r['request']==request)
            other='blue' if request=='red' else 'red'
            rows.append({'scene':scene,'request':request,'target':row['gap_at_crossing_m'][request],
                         'other':row['gap_at_crossing_m'][other], 'delta':row['target_minus_other_gap_m'],
                         'id':row['bound_ids'][0]})
    W,H=1440,650; svg=[f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
      '<rect width="100%" height="100%" fill="white"/>',
      '<style>text{font-family:Arial,sans-serif;fill:#222}.small{font-size:20px}.label{font-size:24px}.head{font-size:29px;font-weight:600}.tick{font-size:19px}</style>']
    src=[]
    for i,scene in enumerate(('object_reference_a','object_reference_b')):
        image=batch/f'{scene}_red_s41/episode/proposal_001_input.jpg'
        src.append(image)
        from PIL import Image
        im=Image.open(image).crop((0,38,640,356)).resize((640,248))
        crop=out/f'object_reference_panel_{i+1}.png';src.append(crop)
        from PIL import ImageDraw, ImageFont
        draw=ImageDraw.Draw(im)
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',20)
        centers=[(185,38),(470,38)] if i==0 else [(470,38),(185,38)]
        targets=[(185,155),(470,155)] if i==0 else [(470,155),(185,155)]
        names=['Red cart · ID 1','Blue cart · ID 2'] if i==0 else ['Red cart · ID 2','Blue cart · ID 1']
        for (cx,cy), (tx,ty),name in zip(centers,targets,names):
            draw.line((cx,cy+18,tx,ty-20),fill='#555555',width=2)
            draw.rounded_rectangle((cx-86,cy-19,cx+86,cy+19),radius=3,fill='white',outline='#555555',width=1)
            draw.text((cx,cy),name,fill='#222222',font=font,anchor='mm')
        im.save(crop); embedded=base64.b64encode(crop.read_bytes()).decode('ascii')
        x=55+i*690
        svg.append(f'<text x="{x}" y="36" class="head">({"a" if i==0 else "b"}) {"Original appearance" if i==0 else "Appearance exchanged"}</text>')
        svg.append(f'<image xlink:href="{crop.resolve()}" x="{x}" y="52" width="640" height="248" preserveAspectRatio="none"/>')
        binds=[]
        for request in ('red','blue'):
            row=next(r for r in rows if r['scene']==scene and r['request']==request)
            binds.append(f'{request.title()} request → {row["id"]}')
        
    x,y,w,h=205,380,1040,190; xmin,xmax=.58,.79
    svg.append('<text x="55" y="365" class="head">(c) Measured clearance at trolley crossing</text>')
    def px(v):return x+(v-xmin)/(xmax-xmin)*w
    labels=['Original · red','Original · blue','Swapped · red','Swapped · blue']
    for tick in (.60,.65,.70,.75):
        xx=px(tick);svg.append(f'<line x1="{xx:.1f}" y1="{y-10}" x2="{xx:.1f}" y2="{y+h}" stroke="#e4e4e4"/><text x="{xx:.1f}" y="{y+h+25}" text-anchor="middle" class="tick">{tick:.2f}</text>')
    for i,row in enumerate(rows):
        yy=y+18+i*43
        svg.append(f'<text x="{x-18}" y="{yy+5}" text-anchor="end" class="small">{labels[i]}</text>')
        a,b=px(row['other']),px(row['target'])
        svg.append(f'<line x1="{a:.1f}" y1="{yy}" x2="{b:.1f}" y2="{yy}" stroke="#aeb3b5" stroke-width="3"/>')
        svg.append(f'<circle cx="{a:.1f}" cy="{yy}" r="7" fill="#6f767a"/><circle cx="{b:.1f}" cy="{yy}" r="8" fill="#9a7410"/>')
        svg.append(f'<text x="{min(a,b)-4:.1f}" y="{yy-11}" text-anchor="end" class="tick">{row["other"]:.3f}</text><text x="{max(a,b)+4:.1f}" y="{yy-11}" class="tick">{row["target"]:.3f}</text>')
        svg.append(f'<text x="{x+w+20}" y="{yy+5}" class="small">+{row["delta"]:.3f} m</text>')
    svg.append(f'<text x="{x+w/2}" y="{H-16}" text-anchor="middle" class="label">Chassis-to-trolley proxy clearance (m)</text>')
    svg.append(f'<circle cx="1000" cy="365" r="6" fill="#6f767a"/><text x="1014" y="370" class="small">Other trolley</text><circle cx="1140" cy="365" r="6" fill="#9a7410"/><text x="1154" y="370" class="small">Referenced trolley</text>')
    svg.append('</svg>');svg_text='\n'.join(svg)
    svg_path=out/'object_reference_evidence.svg';svg_path.write_text(svg_text)
    import subprocess
    subprocess.run(['inkscape',str(svg_path),'--export-filename='+str(out/'object_reference_evidence.pdf')],check=True)
    subprocess.run(['inkscape',str(svg_path),'--export-filename='+str(out/'object_reference_evidence.png'),'--export-width=2600'],check=True)
    for r in metrics['metrics']:
        if r['request'] in ('red','blue'):
            for name in ('trace.jsonl','proposal_001_input.jpg'):
                f=batch/r['name']/'episode'/name
                if f.exists():src.append(f)
    for f in (batch/'protocol.json',batch/'analysis/execution_metrics.json',batch/'analysis/grounding_probes.json'):
        src.append(f)
    (out/'object_reference_evidence.provenance.json').write_text(json.dumps({
      'kind':'Measured Gazebo object-referenced clearance with exchanged visual appearance',
      'episodes':rows,'image_binding_queries':probes['full_image_correct'],'no_image_binding_queries':probes['without_image_correct'],
      'clearance_definition':'recorded oriented vehicle proxy to trolley proxy separation at crossing pose',
      'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(set(src))},
      'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')


if __name__=='__main__':main()
