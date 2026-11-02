#!/usr/bin/env python3
"""Run/resume the preregistered 2x2 Office A delivery simulation matrix."""
import argparse, csv, json, os, statistics, subprocess, sys, time
from pathlib import Path

PKG=Path(__file__).resolve().parents[1]
ROOT=PKG.parents[1]
RUNNER=PKG/'scripts/run_office_delivery_mppi.py'
CONDITIONS=('GEO_ACK','GEO_4WS','VLM_ACK','VLM_4WS')
LAYOUTS=('east_weave','west_weave','door_approach')
SEEDS=(17,23,31)
METRICS=('task_success','elapsed_sim_s','path_length_m','delivery_position_error_m',
         'delivery_heading_error_deg','person_proxy_min_clearance_m',
         'trolley_proxy_min_clearance_m','mode_switches','visual_selections',
         'visual_selector_errors','visual_fallback_to_geometry')

def write_summary(root):
    rows=[]
    for p in sorted(root.glob('*/episode/result.json')):
        try: result=json.loads(p.read_text())
        except Exception: continue
        rows.append(result)
    raw=root/'episode_results.csv'
    if rows:
        with raw.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['condition','layout','seed',*METRICS,'status'])
            w.writeheader()
            for x in rows:w.writerow({k:x.get(k) for k in w.fieldnames})
    summaries=[]
    for c in CONDITIONS:
        group=[x for x in rows if x['condition']==c]
        line={'condition':c,'episodes':len(group),'successes':sum(x['task_success'] for x in group),
              'success_rate':sum(x['task_success'] for x in group)/len(group) if group else None}
        for metric in METRICS[1:]:
            vals=[x.get(metric) for x in group if isinstance(x.get(metric),(int,float))]
            line[metric+'_mean']=statistics.mean(vals) if vals else None
            line[metric+'_sd']=statistics.stdev(vals) if len(vals)>1 else (0. if vals else None)
        summaries.append(line)
    (root/'condition_summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2))
    if summaries:
        with (root/'condition_summary.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    return rows,summaries

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-root',default='artifacts/office_delivery_mppi_20261014/final_matrix_run9')
    p.add_argument('--resume',action='store_true');p.add_argument('--condition',choices=CONDITIONS)
    p.add_argument('--layout',choices=LAYOUTS);p.add_argument('--seed',type=int,choices=SEEDS)
    a=p.parse_args();root=(ROOT/a.output_root).resolve();root.mkdir(parents=True,exist_ok=True)
    if any(c.startswith('VLM') for c in (a.condition,) if c) and not os.environ.get('DEEPSEEK_API_KEY'):
        raise SystemExit('VLM runs require DEEPSEEK_API_KEY in the parent process environment')
    env=os.environ.copy();env['DEEPSEEK_VISION_MODEL']=env.get('DEEPSEEK_VISION_MODEL','deepseek-flash')
    journal=(root/'matrix_runs.jsonl').open('a')
    total=0;failures=0
    for condition in CONDITIONS:
        if a.condition and condition!=a.condition:continue
        for layout in LAYOUTS:
            if a.layout and layout!=a.layout:continue
            for seed in SEEDS:
                if a.seed and seed!=a.seed:continue
                total+=1
                run=root/f'{condition}_{layout}_s{seed}'
                result=run/'episode/result.json'
                if a.resume and result.exists():
                    print(f'SKIP {condition} {layout} seed={seed} (result exists)',flush=True);continue
                (root/'launcher_logs').mkdir(exist_ok=True)
                cmd=[sys.executable,str(RUNNER),'--condition',condition,'--layout',layout,
                     '--seed',str(seed),'--output',str(run)]
                start=time.time();status='FAILED';detail=''
                with (root/'launcher_logs'/f'{condition}_{layout}_s{seed}.log').open('w') as log:
                    try:
                        proc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=420)
                        status='DONE' if proc.returncode==0 and (run/'episode/result.json').exists() else 'FAILED'
                        if status=='FAILED':detail=f'exit={proc.returncode}'
                    except subprocess.TimeoutExpired:
                        status='WALL_TIMEOUT';detail='launcher exceeded 420 s'
                row={'condition':condition,'layout':layout,'seed':seed,'status':status,
                     'wall_s':round(time.time()-start,2),'detail':detail}
                if result.exists():
                    try:row.update(json.loads(result.read_text()))
                    except Exception as exc:row['detail']=str(exc)
                else:failures+=1
                journal.write(json.dumps(row,ensure_ascii=False)+'\n');journal.flush()
                print(json.dumps({'index':total,'condition':condition,'layout':layout,'seed':seed,
                    'status':status,'wall_s':row['wall_s'],'task_success':row.get('task_success')},ensure_ascii=False),flush=True)
                write_summary(root)
    journal.close();rows,summaries=write_summary(root)
    print(json.dumps({'selected_runs':total,'recorded_runs':len(rows),'runs_without_result':failures,
                      'summaries':summaries},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
