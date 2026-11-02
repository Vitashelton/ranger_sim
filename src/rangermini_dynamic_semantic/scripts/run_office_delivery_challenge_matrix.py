#!/usr/bin/env python3
"""Run paired geometry MPPI / visual MPPI delivery challenge episodes."""
import argparse, csv, json, os, statistics, subprocess, sys, time
from pathlib import Path

PKG=Path(__file__).resolve().parents[1]
ROOT=PKG.parents[1]
RUNNER=PKG/'scripts/run_office_delivery_mppi.py'
LAYOUTS=('person_clearance','person_clearance_mirror','doorway_delivery')
SEEDS=(17,23,31)
CONDITIONS=('GEO_4WS','VLM_4WS')

def compare_initial_rollouts(left,right,tolerance=1e-4):
    """Compare the common moving candidates generated at the paired start."""
    a=json.loads(Path(left).read_text());b=json.loads(Path(right).read_text())
    a={x['candidate_id']:x for x in a if x.get('motion_behavior')!='STOP'}
    b={x['candidate_id']:x for x in b if x.get('motion_behavior')!='STOP'}
    if set(a)!=set(b):return {'match':False,'reason':'candidate IDs differ',
        'geometry_ids':sorted(a),'visual_ids':sorted(b)}
    maximum=0.0
    for cid in sorted(a):
        x=a[cid]['trajectory'];y=b[cid]['trajectory']
        if len(x)!=len(y):return {'match':False,'reason':f'{cid} rollout length differs'}
        for px,py in zip(x,y):
            if len(px)!=len(py):return {'match':False,'reason':f'{cid} pose dimension differs'}
            maximum=max(maximum,max(abs(float(u)-float(v)) for u,v in zip(px,py)))
    return {'match':maximum<=tolerance,'max_abs_pose_delta':maximum,
            'tolerance':tolerance,'candidate_count':len(a)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-root',default='artifacts/office_delivery_challenges_20261014')
    ap.add_argument('--condition',choices=CONDITIONS)
    ap.add_argument('--layout',choices=LAYOUTS)
    ap.add_argument('--seed',type=int,choices=SEEDS)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--clearance-threshold',type=float,default=.60)
    ap.add_argument('--trolley-clearance-threshold',type=float,default=.35)
    a=ap.parse_args()
    root=(ROOT/a.output_root).resolve();root.mkdir(parents=True,exist_ok=True)
    if (a.condition in (None,'VLM_4WS')) and not os.environ.get('DEEPSEEK_API_KEY'):
        raise SystemExit('VLM paired runs require DEEPSEEK_API_KEY in the parent process environment')
    selected=[c for c in CONDITIONS if not a.condition or c==a.condition]
    layouts=[l for l in LAYOUTS if not a.layout or l==a.layout]
    seeds=[s for s in SEEDS if a.seed is None or s==a.seed]
    logdir=root/'launcher_logs';logdir.mkdir(exist_ok=True)
    journal=(root/'challenge_runs.jsonl').open('a')
    idx=0
    for layout in layouts:
        for seed in seeds:
            # Paired runs are adjacent and share a world configuration, seed,
            # robot model and local controller. Existing results are immutable.
            for condition in selected:
                idx+=1;run=root/f'{condition}_{layout}_s{seed}'
                result=run/'episode/result.json'
                if a.resume and result.exists():
                    print(json.dumps({'index':idx,'condition':condition,'layout':layout,
                        'seed':seed,'status':'SKIP_EXISTING'}),flush=True);continue
                cmd=[sys.executable,str(RUNNER),'--condition',condition,'--layout',layout,
                     '--seed',str(seed),'--clearance-threshold',str(a.clearance_threshold),
                     '--trolley-clearance-threshold',str(a.trolley_clearance_threshold),
                     '--ros-domain-id',str(30+(sum(map(ord,layout))+seed+(0 if condition=='GEO_4WS' else 1))%190),
                     '--ign-partition',f'ranger_challenge_{condition}_{layout}_{seed}',
                     '--output',str(run)]
                start=time.time();status='FAILED';detail=''
                with (logdir/f'{condition}_{layout}_s{seed}.log').open('w') as out:
                    try:
                        proc=subprocess.run(cmd,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,
                                            timeout=720,env=os.environ.copy())
                        status='DONE' if proc.returncode==0 and result.exists() else 'FAILED'
                        if status=='FAILED':detail=f'exit={proc.returncode}'
                    except subprocess.TimeoutExpired:
                        status='WALL_TIMEOUT';detail='episode exceeded 720 s'
                row={'condition':condition,'layout':layout,'seed':seed,'status':status,
                     'wall_s':round(time.time()-start,2),'detail':detail}
                if result.exists():
                    try:row.update(json.loads(result.read_text()))
                    except Exception as exc:row['detail']=str(exc)
                journal.write(json.dumps(row,ensure_ascii=False)+'\n');journal.flush()
                print(json.dumps({k:row.get(k) for k in ('condition','layout','seed','status',
                    'task_success','person_proxy_min_clearance_m','path_length_m','elapsed_sim_s')},
                    ensure_ascii=False),flush=True)
    journal.close()
    if set(selected)==set(CONDITIONS):
        pair_checks=[]
        for layout in layouts:
            for seed in seeds:
                left=root/f'GEO_4WS_{layout}_s{seed}'/'episode'/'selection_01_rollouts.json'
                right=root/f'VLM_4WS_{layout}_s{seed}'/'episode'/'selection_01_rollouts.json'
                if left.exists() and right.exists():
                    check=compare_initial_rollouts(left,right)
                else:check={'match':False,'reason':'one or both rollout files missing'}
                pair_checks.append({'layout':layout,'seed':seed,**check})
        (root/'paired_initial_rollout_check.json').write_text(
            json.dumps(pair_checks,ensure_ascii=False,indent=2))
    results=[]
    for f in sorted(root.glob('*/episode/result.json')):
        try:results.append(json.loads(f.read_text()))
        except Exception:continue
    if results:
        fields=['condition','layout','seed','status','task_success','person_proxy_min_clearance_m',
                'trolley_proxy_min_clearance_m','elapsed_sim_s','path_length_m','mode_switches',
                'visual_selections','visual_selector_errors','visual_fallback_to_geometry']
        with (root/'episode_results.csv').open('w',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=fields);w.writeheader()
            for row in results:w.writerow({k:row.get(k) for k in fields})
        summaries=[]
        for condition in selected:
            group=[r for r in results if r.get('condition')==condition]
            summaries.append({'condition':condition,'episodes':len(group),
                'successes':sum(bool(r.get('task_success')) for r in group),
                'success_rate':(sum(bool(r.get('task_success')) for r in group)/len(group)) if group else None,
                'mean_person_clearance_m':statistics.mean(r['person_proxy_min_clearance_m'] for r in group
                    if isinstance(r.get('person_proxy_min_clearance_m'),(int,float))) if group else None,
                'mean_trolley_clearance_m':statistics.mean(r['trolley_proxy_min_clearance_m'] for r in group
                    if isinstance(r.get('trolley_proxy_min_clearance_m'),(int,float))) if group else None})
        (root/'condition_summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2))
    print(json.dumps({'output_root':str(root),'selected_conditions':selected,
        'layouts':layouts,'seeds':seeds,'clearance_threshold_m':a.clearance_threshold,
        'trolley_clearance_threshold_m':a.trolley_clearance_threshold},
        ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
