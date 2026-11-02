"""Compile a conversational service request using registered facts and a real model."""
import argparse,json,sys,time
from pathlib import Path
from urllib.request import Request,urlopen
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/rangermini_dynamic_semantic'))
from rangermini_dynamic_semantic.service_task_context import task_compile_request
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph
from rangermini_dynamic_semantic.long_service_task import validate_task
from rangermini_dynamic_semantic.deepseek_vision_provider import DeepSeekVisionProvider

SYSTEM = ('Compile the user service request into an ordered service plan. '
 'Use registered rooms and the provided service directory to resolve names and locations. '
 'Directory facts are registration, not visual recognition. Do not claim to have recognized a person. '
 'Do not invent a person, room, service action, pickup, visual observation, clearance number, heading requirement, '
 'chassis mode, coordinate or path. Preserve item identity across pickup and delivery. '
 'When an identity or destination is missing or ambiguous, return '
 '{"status":"clarification_needed","question":"a concise question to the user"}. '
 'Otherwise return {"version":1,"stops":[{"room":"registered room number",'
 '"actions":[{"type":"pickup|deliver|notify","item":"canonical item ID"}]}]}. '
 'No additional navigation constraints are required from the user. JSON only.')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--world',type=Path,required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--instruction',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
 context=task_compile_request(a.instruction,SpatialGraph(a.world),a.directory)
 messages=[dict(role='system',content=SYSTEM),dict(role='user',content=json.dumps(context,ensure_ascii=False))]
 a.output.mkdir(parents=True,exist_ok=True)
 (a.output/'request_context.json').write_text(json.dumps(dict(context=context,messages=messages,executed_model_request=False),ensure_ascii=False,indent=2)+'\n')
 if a.prepare_only:
  print('Prepared registered task context; no model request or navigation executed.');return
 client=DeepSeekVisionProvider()
 if not client.configured:raise SystemExit('DEEPSEEK_API_KEY is required for a real model request; no plan was generated.')
 body=dict(model=client.model,temperature=0,response_format=dict(type='json_object'),messages=messages)
 req=Request(client.base_url+'/chat/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+client.api_key,'Content-Type':'application/json'})
 begin=time.monotonic()
 with urlopen(req,timeout=client.timeout_s) as response:raw=json.loads(response.read())
 content=json.loads(raw['choices'][0]['message']['content'])
 (a.output/'model_response.json').write_text(json.dumps(dict(response=raw,request=body,latency_s=time.monotonic()-begin),ensure_ascii=False,indent=2)+'\n')
 if content.get('status')=='clarification_needed':
  (a.output/'clarification.json').write_text(json.dumps(content,ensure_ascii=False,indent=2)+'\n');print(content['question']);return
 plan=validate_task(content,SpatialGraph(a.world));plan.update(instruction=a.instruction,source='real_language_model',registration_context=context.get('service_directory'))
 (a.output/'task_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n');print('Validated model plan saved to',a.output/'task_plan.json')
if __name__=='__main__':main()
