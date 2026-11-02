#!/usr/bin/env python3
"""Run public-solver Gazebo comparisons; preserve every attempt and source hash."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/rangermini_dynamic_semantic/scripts'))
from audit_grounded_delivery import audit

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',required=True)
    ap.add_argument('--seeds',type=int,nargs='+',default=[17,37,41])
    ap.add_argument('--scenarios',nargs='+',default=['offset_portal','encounter_entry'])
    ap.add_argument('--methods',nargs='+',default=['mppi-h','ranger-mppi'])
    ap.add_argument('--max-sim-time',type=float,default=100)
    ap.add_argument('--max-wall-time',type=float,default=320)
    ap.add_argument('--selector',choices=['vlm','cost'],default='cost')
    ap.add_argument('--samples-per-family',type=int,default=75)
    ap.add_argument('--upstream-samples',type=int,default=300)
    args=ap.parse_args()
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    files=list((ROOT/'integration/mppi_h_adapter').rglob('*.cpp'))+list((ROOT/'integration/mppi_h_adapter/compat').rglob('*.hpp'))
    files+=[ROOT/'integration/mppi_h_adapter/build.py',ROOT/'integration/mppi_h_adapter/build/libmppi_h.so']
    files+=list((ROOT/'src/rangermini_dynamic_semantic/rangermini_dynamic_semantic').glob('*.py'))
    files+=list((ROOT/'src/rangermini_dynamic_semantic/scripts').glob('*.py'))
    files.append(Path(__file__).resolve())
    hashes={str(p.relative_to(ROOT)):digest(p) for p in files}
    protocol=dict(methods=args.methods,scenarios=args.scenarios,seeds=args.seeds,
        full_selector=args.selector,source_sha256=hashes,upstream=json.loads((ROOT/'integration/mppi_h_adapter/provenance.json').read_text()),
        common=dict(speed_m_s=.35,finite_steering=True,world='furnished_lab',
            matched_total_rollouts=4*args.samples_per_family,
            planner_samples_per_motion_family=args.samples_per_family,
            upstream_samples=args.upstream_samples,
            max_sim_time=args.max_sim_time),
        intended_episodes=len(args.methods)*len(args.scenarios)*len(args.seeds))
    frozen=out/'protocol.json'
    if frozen.exists() and json.loads(frozen.read_text())!=protocol:
        raise RuntimeError('source or protocol changed: refuse mixing runs')
    frozen.write_text(json.dumps(protocol,indent=2,ensure_ascii=False))
    summaries=[]
    for scenario in args.scenarios:
        for seed in args.seeds:
            # Rotate order across seeds; physical Gazebo transport is isolated.
            methods=args.methods[seed%len(args.methods):]+args.methods[:seed%len(args.methods)]
            for method in methods:
                if any(digest(ROOT/k)!=v for k,v in hashes.items()):
                    raise RuntimeError('frozen source changed during campaign')
                name=f'{scenario}_{method}_s{seed}';run=out/name
                if (run/'interaction_audit.json').exists():
                    summaries.append(audit(run));continue
                if run.exists():
                    raise RuntimeError(f'partial attempt retained at {run}; inspect before restart')
                cmd=[sys.executable,str(ROOT/'src/rangermini_dynamic_semantic/scripts/run_grounded_delivery.py'),
                    '--output',str(run),'--planner',method,'--scenario',scenario,'--seed',str(seed),
                    '--policy','predictive',
                    '--proposal-selector','cost',
                    '--samples',str(args.samples_per_family),'--upstream-samples',str(args.upstream_samples),
                    '--stable-modes','--verified-recovery','--map-recovery',
                    '--max-sim-time',str(args.max_sim_time),'--max-wall-time',str(args.max_wall_time),
                    '--ros-domain-id','195','--ign-partition','public_'+name.replace('-','_')]
                print('START',name,flush=True)
                with (out/'campaign.log').open('a') as log:
                    rc=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
                if not (run/'episode/result.json').exists():
                    (out/'interruption.json').write_text(json.dumps(dict(run=name,return_code=rc)))
                    raise RuntimeError(f'{name}: no terminal result, see campaign.log')
                result=audit(run);result.update(method=method)
                summaries.append(result)
                (out/'results.json').write_text(json.dumps(summaries,indent=2,ensure_ascii=False))
                print('DONE',name,result['status'],'qualified',result['interaction_success'],flush=True)
    print('COMPLETE',len(summaries),flush=True)
if __name__=='__main__':main()
