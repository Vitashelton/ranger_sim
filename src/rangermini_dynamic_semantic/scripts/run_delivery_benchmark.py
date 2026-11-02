#!/usr/bin/env python3
"""One command: two furnished scenes x three selectors x three seeds, then report."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    root=args.output.resolve()
    if root.exists() and not args.resume:raise SystemExit('Output already exists; use a new path or --resume.')
    root.mkdir(parents=True,exist_ok=True)
    here=Path(__file__).resolve().parent
    cells=[(side,method,seed) for side in ['north','south'] for seed in [17,23,31] for method in ['geometric','fixed','vlm']]
    missing=[]
    for side,method,seed in cells:
        result=root/('runs' if side=='north' else 'south_runs')/f'{method}_s{seed}'/'episode/result.json'
        if not result.exists() or json.loads(result.read_text()).get('status')!='ARRIVED':missing.append((side,method,seed))
    if any(method=='vlm' for _,method,_ in missing) and not os.environ.get('DEEPSEEK_API_KEY'):
        raise SystemExit('DEEPSEEK_API_KEY must be set before running the complete matrix.')
    config=dict(scenes=['north','south'],methods=['geometric','fixed','vlm'],seeds=[17,23,31],
        instruction='把车载资料送到前面的实验室。经过箱子时，避开站在旁边的同学，别从他身边挤过去。',
        horizon=36,dt=.12,samples_per_motion_group=16,max_speed=.4,person_proxy_clearance_requirement_m=.6,
        complete_handover_measured=False,source_sha256={str(f.name):hashlib.sha256(f.read_bytes()).hexdigest() for f in [here/'run_delivery_route_pilot.py',here/'summarize_delivery_route_pilot.py']})
    (root/'benchmark_manifest.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
    for side,method,seed in missing:
        work=root/('runs' if side=='north' else 'south_runs')/f'{method}_s{seed}'
        if work.exists():
            archive=root/'failed_attempts';archive.mkdir(exist_ok=True)
            work.rename(archive/f'{side}_{method}_s{seed}_{time.time_ns()}')
        with (root/f'{side}_{method}_s{seed}_launcher.log').open('w') as log:
            process=subprocess.run([sys.executable,str(here/'run_delivery_route_pilot.py'),'--person-side',side,'--selector',method,'--seed',str(seed),'--output',str(work)],stdout=log,stderr=subprocess.STDOUT)
        result=work/'episode/result.json';data=json.loads(result.read_text()) if result.exists() else {}
        status=dict(scene=side,method=method,seed=seed,exit_code=process.returncode,status=data.get('status','NO_RESULT'),route=data.get('route_id'))
        with (root/'batch_status.jsonl').open('a') as log:log.write(json.dumps(status)+'\n')
        print(json.dumps(status),flush=True)
        if process.returncode or data.get('status')!='ARRIVED':raise SystemExit('Episode failed; preserved logs. Resume after fixing the cause.')
    # Keep distro NumPy/Matplotlib consistent, without changing user's packages.
    env=dict(os.environ,PYTHONNOUSERSITE='1')
    for side in ['north','south']:
        subprocess.run(['/usr/bin/python3',str(here/'summarize_delivery_route_pilot.py'),'--root',str(root),'--person-side',side],env=env,check=True)
    subprocess.run([sys.executable,str(here/'build_delivery_meeting.py'),'--root',str(root)],check=True)
    print('COMPLETE: 18 terminal records, source data, figures, videos and meeting deck.',flush=True)


if __name__=='__main__':main()
