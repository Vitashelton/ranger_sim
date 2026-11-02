"""Audit the matched planning-side steering prediction experiment."""
from pathlib import Path
import json,sys,math,statistics
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src/rangermini_dynamic_semantic/scripts'),str(ROOT/'src/rangermini_dynamic_semantic')]
from audit_grounded_delivery import audit

def main():
 root=ROOT/'artifacts/steering_prediction_ablation_20261028'
 progress=json.loads((root/'progress.json').read_text())
 if progress.get('completed')!=4 or progress.get('running') is not None:raise SystemExit('Campaign not completed')
 rows=[]
 for seed in (37,53):
  for mode in ('finite','instant'):
   run=root/f'{mode}_s{seed}'
   result=json.loads((run/'episode/result.json').read_text());trace=[json.loads(x) for x in (run/'episode/trace.jsonl').read_text().splitlines() if x.strip()]
   ts=[x['timestamp'] for x in trace]
   if not ts or ts[0]>=10 or any(a>b for a,b in zip(ts,ts[1:])):raise RuntimeError('Invalid simulation clock: '+str(run))
   assessment=audit(run);changes=[];hold=0.
   for i,row in enumerate(trace[:-1]):
    a=row['actuation'];u=a.get('executed',{});dt=ts[i+1]-ts[i]
    if math.hypot(u.get('vx',0),u.get('vy',0))>.02 or abs(u.get('wz',0))>.02:changes.append(a['mode'])
    hold+=dt*bool(a.get('steering_transition_stop'))
   rows.append(dict(seed=seed,method=mode,status=result['status'],elapsed_s=result['elapsed_sim_s'],path_m=result.get('path_length_m'),proxy_gap_m=assessment['trolley_signed_SAT_min_gap_m'],proxy_overlap_samples=assessment['proxy_overlap_trace_count'],fixture_audit_valid=assessment['fixture_audit_valid'],qualified=assessment['interaction_success'],mode_changes=sum(a!=b for a,b in zip(changes,changes[1:])),alignment_s=hold,run=str(run.relative_to(ROOT))))
 summary={m:dict(n=2,arrived=sum(x['status']=='ARRIVED' for x in rows if x['method']==m),qualified=sum(x['qualified'] for x in rows if x['method']==m)) for m in ('finite','instant')}
 (root/'analysis.json').write_text(json.dumps(dict(kind='Planning-side model ablation, not a public baseline',rows=rows,summary=summary),indent=2)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
