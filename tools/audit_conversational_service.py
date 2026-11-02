"""Audit fresh conversational parse linked to one online service simulation."""
from pathlib import Path
import json,hashlib,sys,argparse,math
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src/rangermini_dynamic_semantic/scripts'),str(ROOT/'src/rangermini_dynamic_semantic')]
from audit_grounded_delivery import audit,rect_gap
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph

def main(campaign):
 campaign=campaign.resolve();run=campaign/'run';ep=run/'episode'
 raw=json.loads((campaign/'task_parser_response.json').read_text())
 request=json.loads((campaign/'request_context.json').read_text())
 model=json.loads(raw['response']['choices'][0]['message']['content'])
 plan=json.loads((run/'long_task_plan.json').read_text());summary=json.loads((run/'long_task_summary.json').read_text())
 trace=[json.loads(s) for s in (ep/'trace.jsonl').read_text().splitlines()]
 events=[json.loads(s) for s in (ep/'selection_events.jsonl').read_text().splitlines()]
 responses=sorted(ep.glob('proposal_*_response.json'))
 graph=SpatialGraph(run/'world.sdf');interaction=audit(run)
 gaps=[min(rect_gap(z['pose'],b[:3],bl=2*b[3],bw=2*b[4]) for b in graph.static_boxes) for z in trace]
 parsed_matches=model['stops']==plan['stops']
 expected=[('617','pickup','实验报告'),('614','deliver','实验报告'),('614','pickup','回执'),('619','deliver','回执')]
 interpreted=[(stop['room'],action['type'],action['item']) for stop in model['stops'] for action in stop['actions']]
 confirmed=[e for e in events if e['event']=='SERVICE_ACTION_CONFIRMED']
 actions=[(e['room'],e['action']['type'],e['action']['item']) for e in confirmed]
 manifest=json.loads((run/'scene_manifest.json').read_text());checks=[]
 for e in confirmed:
  z=min(trace,key=lambda z:abs(z['timestamp']-e['sim_time']));goal=manifest['service_poses'][e['room']]
  pos=math.dist(z['pose'][:2],goal[:2]);yaw=abs(math.atan2(math.sin(z['pose'][2]-goal[2]),math.cos(z['pose'][2]-goal[2])));u=z['actuation']['executed'];speed=math.hypot(u['vx'],u['vy'])
  checks.append(dict(room=e['room'],action=e['action'],nearest_sample_delta_s=z['timestamp']-e['sim_time'],position_error_m=pos,heading_error_deg=math.degrees(yaw),speed_mps=speed,within_registered_service_pose=pos<=.32 and yaw<=math.radians(18) and speed<.025))
 intent_matches=interpreted==expected and actions==expected
 instruction_matches=request['context']['instruction']==plan['instruction']==summary['instruction']
 sources=[campaign/'task_parser_response.json',campaign/'request_context.json',run/'long_task_plan.json',run/'long_task_summary.json',ep/'trace.jsonl',ep/'selection_events.jsonl',run/'world.sdf']+responses
 result=dict(kind='One conversational task parse followed by online VLM and MPPI service navigation',instruction=plan['instruction'],task_plan_origin='Real model parse of this user request, reused across the dynamic and static scenario attempts',parse_precedes_simulation=True,runtime_loaded_supplied_plan_file=True,language_plan_matches_raw_model=parsed_matches,instruction_matches=instruction_matches,people_room_source='explicit simulation service directory, not visual identity recognition',physical_handling_validated=False,status=interaction['status'],task_complete=summary['task_complete'],confirmed_action_count=summary['confirmed_action_count'],real_visual_responses=len(responses),minimum_static_signed_SAT_m=min(gaps),static_overlap_pose_count=sum(g<=0 for g in gaps),proxy_overlap_trace_count=interaction['proxy_overlap_trace_count'],fixture_audit_valid=interaction['fixture_audit_valid'],elapsed_sim_s=interaction['elapsed_sim_s'],path_length_m=interaction['path_length_m'],source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
 result.update(task_intent_reference=dict(source='Evaluation interpretation of the actual request and explicit service directory, not supplied as a plan to the model',expected_actions=expected,interpretation_matches=intent_matches),service_pose_checks=checks)
 result['qualified']=bool(intent_matches and all(c['within_registered_service_pose'] for c in checks) and parsed_matches and instruction_matches and summary['task_complete'] and len(responses)>0 and not result['static_overlap_pose_count'] and not result['proxy_overlap_trace_count'] and result['fixture_audit_valid'])
 (campaign/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},ensure_ascii=False,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);main(p.parse_args().campaign)
