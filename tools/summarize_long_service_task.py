#!/usr/bin/env python3
"""Summarize recorded long-task actions separately from physical handling."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();run=a.run
plan=json.loads((run/'long_task_plan.json').read_text());ep=run/'episode'
events=[json.loads(s) for s in (ep/'selection_events.jsonl').read_text().splitlines()]
trace=[json.loads(s) for s in (ep/'trace.jsonl').read_text().splitlines()]
result=json.loads((ep/'result.json').read_text()) if (ep/'result.json').exists() else None
expected=[(i,j,s['room'],action['type'],action['item']) for i,s in enumerate(plan['stops']) for j,action in enumerate(s['actions'])]
confirmed=[e for e in events if e['event']=='SERVICE_ACTION_CONFIRMED']
observed=[(e['stop_index'],e['action_index'],e['room'],e['action']['type'],e['action']['item']) for e in confirmed]
vision=[]
for f in sorted(ep.glob('proposal_*_response.json')):
 v=json.loads(f.read_text())
 if v.get('provider')=='deepseek_grounded_navigation':vision.append(f.name)
from collections import Counter
out={'status':result['status'] if result else 'RUNNING_OR_INTERRUPTED',
 'instruction':plan.get('instruction', ''),'plan_source':plan.get('source','unspecified'),
 'rooms':[s['room'] for s in plan['stops']], 'expected_actions':expected,'confirmed_actions':observed,
 'confirmed_action_count':len(observed),'dependencies_match_expected_prefix':observed==expected[:len(observed)],
 'task_complete':bool(result and result.get('task_success') and observed==expected),
 'confirmation_sources':dict(Counter(e['confirmation_source'] for e in confirmed)),
 'mode_observations':dict(Counter(r.get('actuation',{}).get('mode') for r in trace)),
 'visual_http_responses':len(vision),'real_language_parse_archived':any((run/name).exists() for name in ('task_parser_response.json','archived_task_parser_response.json')),
 'language_plan_reused':plan.get('source')=='archived_language_model_plan',
 'last_pose':trace[-1]['pose'] if trace else None,
 'last_stage':trace[-1]['task_stage'] if trace else None,
 'physical_pickup_delivery_validated':False,
 'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
  [run/'long_task_plan.json',ep/'selection_events.jsonl',ep/'trace.jsonl']+
  [run/name for name in ('task_parser_response.json','archived_task_parser_response.json','language_plan_provenance.json') if (run/name).exists()]}}
(run/'long_task_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!='source_sha256'},ensure_ascii=False,indent=2))
