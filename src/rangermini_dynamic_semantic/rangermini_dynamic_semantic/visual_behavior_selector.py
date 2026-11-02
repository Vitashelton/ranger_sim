"""Bounded visual selection over planner-generated route candidates.

The model returns a route ID, never a chassis command. Route IDs are separate
from Ackermann/parallel/spinning modes, which remain the local planner's job.
"""
import base64
import json
import os
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


def select_direct_velocity(image_bytes, instruction, state, *, timeout_s=35):
    """Direct VLM-to-body-twist baseline; no trajectory candidates or MPPI."""
    key = os.environ.get('DEEPSEEK_API_KEY', '')
    if not key:
        raise RuntimeError('DEEPSEEK_API_KEY unavailable')
    model = os.environ.get('DEEPSEEK_VISION_MODEL', 'deepseek-flash')
    prompt = (
        'You directly control a simulated four-wheel-steered indoor delivery robot. '
        'Use the current RGB camera image, task instruction, relative goal and reported wheel state. '
        'Output the next body-velocity command to be held until your next observation. '
        'vx is forward/backward m/s, vy is left/right m/s, and wz is yaw rate rad/s. '
        'Make steady progress toward the subgoal while avoiding visible collisions. Do not output a route ID, '
        'trajectory, steering angle, or wheel mode. Return exactly JSON with numeric vx, vy, wz '
        'and a short reason. Use the same limits as the MPPI controller: sqrt(vx^2+vy^2)<=0.35 m/s '
        'and abs(wz)<=0.90 rad/s. '
        'Use zero velocity only when stopping is necessary.')
    content = [
        {'type': 'text', 'text': json.dumps({'instruction': instruction, 'state': state}, ensure_ascii=False)},
        {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' +
            base64.b64encode(image_bytes).decode(), 'detail': 'high'}}]
    payload = dict(model=model, stream=False, response_format={'type': 'json_object'}, messages=[
        {'role': 'system', 'content': prompt}, {'role': 'user', 'content': content}])
    base = os.environ.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com').rstrip('/')
    if not base.startswith('https://'):
        raise ValueError('HTTPS endpoint required')
    request = Request(base + '/chat/completions', data=json.dumps(payload).encode(), headers={
        'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
    start = time.monotonic()
    try:
        with urlopen(request, timeout=timeout_s) as response:
            raw = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f'DeepSeek HTTP {exc.code}') from None
    except (URLError, TimeoutError, OSError):
        raise RuntimeError('DeepSeek network/timeout failure') from None
    answer = json.loads(raw['choices'][0]['message']['content'])
    if set(answer) != {'vx', 'vy', 'wz', 'reason'}:
        raise ValueError('direct-control response schema mismatch')
    if any(isinstance(answer[k], bool) or not isinstance(answer[k], (int, float))
           or not __import__('math').isfinite(answer[k]) for k in ('vx', 'vy', 'wz')):
        raise ValueError('direct-control command must contain finite numbers')
    if (answer['vx']**2 + answer['vy']**2)**.5 > .35 or abs(answer['wz']) > .90:
        raise ValueError('direct-control command exceeds declared bounds')
    if not isinstance(answer['reason'], str) or len(answer['reason']) > 500:
        raise ValueError('invalid direct-control reason')
    return dict(provider='deepseek_direct_twist', requested_model=model,
        served_model=raw.get('model'), command=[float(answer['vx']), float(answer['vy']), float(answer['wz'])],
        reason=answer['reason'], latency_s=time.monotonic()-start, usage=raw.get('usage', {}))


def select_visual_route(image_bytes, instruction, candidates, *, timeout_s=30):
    key=os.environ.get('DEEPSEEK_API_KEY','')
    if not key:raise RuntimeError('DEEPSEEK_API_KEY unavailable')
    model=os.environ.get('DEEPSEEK_VISION_MODEL','deepseek-flash')
    available={c['candidate_id'] for c in candidates if c['feasible']}
    if not available:raise ValueError('no feasible candidate')
    prompt=("You select a route for a mobile robot delivering documents through a laboratory corridor. "
        "The supplied image is the actual forward robot camera with planner trajectories overlaid. "
        "Green is LEFT and orange is RIGHT, relative to the fixed forward corridor direction. "
        "LEFT means passing on the image-left side; RIGHT means image-right. These are routes, not wheel modes. "
        "Use the user's instruction and actual visual relationships. Do not invent identities or room labels. "
        "Choose only a feasible candidate listed in the request. Return exactly one JSON object: "
        '{"candidate_id":"LEFT|RIGHT", "reason":"short scene-grounded reason"}. '
        "No coordinates, velocities, joints, or alternative fields.")
    content=[{'type':'text','text':json.dumps(dict(instruction=instruction,candidates=candidates),ensure_ascii=False)}]
    if image_bytes is not None:
        content.append({'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(image_bytes).decode(),'detail':'high'}})
    else:
        content.append({'type':'text','text':'Image withheld for a controlled ablation. No object locations are available.'})
    payload=dict(model=model,stream=False,response_format={'type':'json_object'},
        messages=[{'role':'system','content':prompt},{'role':'user','content':content}])
    base=os.environ.get('DEEPSEEK_BASE_URL','https://api.deepseek.com').rstrip('/')
    if not base.startswith('https://'):raise ValueError('HTTPS endpoint required')
    request=Request(base+'/chat/completions',data=json.dumps(payload).encode(),headers={
        'Content-Type':'application/json','Authorization':'Bearer '+key})
    start=time.monotonic()
    try:
        with urlopen(request,timeout=timeout_s) as response:raw=json.load(response)
    except HTTPError as exc:raise RuntimeError(f'DeepSeek HTTP {exc.code}') from None
    except (URLError,TimeoutError,OSError):raise RuntimeError('DeepSeek network/timeout failure') from None
    answer=json.loads(raw['choices'][0]['message']['content'])
    if set(answer)!={'candidate_id','reason'} or answer['candidate_id'] not in available:
        raise ValueError('invalid or infeasible visual route selection')
    if not isinstance(answer['reason'],str) or len(answer['reason'])>1000:
        raise ValueError('invalid selection reason')
    return dict(provider='deepseek',image_supplied=image_bytes is not None,requested_model=model,served_model=raw.get('model'),
        decision=answer,latency_s=time.monotonic()-start,usage=raw.get('usage',{}))


def select_visual_trajectory(image_bytes, instruction, candidates, *, timeout_s=45):
    """Select among continuous MPPI rollouts projected into the live RGB view."""
    key=os.environ.get('DEEPSEEK_API_KEY','')
    if not key:raise RuntimeError('DEEPSEEK_API_KEY unavailable')
    model=os.environ.get('DEEPSEEK_VISION_MODEL','deepseek-flash')
    available={c['candidate_id'] for c in candidates if c.get('feasible')}
    if not available:raise ValueError('no feasible trajectory candidate')
    prompt=("Select the next short-horizon trajectory for a RangerMini performing the instructed indoor service "
        "mission. In the current task, the robot carries documents through a laboratory corridor and enters "
        "Office A / room 619 through its doorway to reach the indoor handoff zone. The image is the robot's "
        "current RGB view. Colored lines are feasible MPPI trajectories projected "
        "into the image; the legend maps colors to candidate IDs and motion behaviors. The JSON lists the same "
        "candidates and their geometric clearances. Follow the instruction, respect visible people, equipment, "
        "door geometry, and progress toward the current task subgoal. Do not infer hidden objects or create new "
        "routes. Return exactly one JSON object containing a listed candidate ID and short visual reason: "
        '{"candidate_id":"...","reason":"..."}. Do not output velocities or steering commands.')
    content=[{'type':'text','text':json.dumps({'instruction':instruction,'candidate_set':candidates},ensure_ascii=False)},
        {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(image_bytes).decode(),'detail':'high'}}]
    payload=dict(model=model,stream=False,response_format={'type':'json_object'},messages=[
        {'role':'system','content':prompt},{'role':'user','content':content}])
    base=os.environ.get('DEEPSEEK_BASE_URL','https://api.deepseek.com').rstrip('/')
    if not base.startswith('https://'):raise ValueError('HTTPS endpoint required')
    request=Request(base+'/chat/completions',data=json.dumps(payload).encode(),headers={
        'Content-Type':'application/json','Authorization':'Bearer '+key})
    start=time.monotonic()
    try:
        with urlopen(request,timeout=timeout_s) as response:raw=json.load(response)
    except HTTPError as exc:raise RuntimeError(f'DeepSeek HTTP {exc.code}') from None
    except (URLError,TimeoutError,OSError):raise RuntimeError('DeepSeek network/timeout failure') from None
    answer=json.loads(raw['choices'][0]['message']['content'])
    if set(answer)!={'candidate_id','reason'} or answer['candidate_id'] not in available:
        raise ValueError('invalid or infeasible trajectory selection')
    if not isinstance(answer['reason'],str) or len(answer['reason'])>500:
        raise ValueError('invalid trajectory selection reason')
    return dict(provider='deepseek_visual_trajectory',requested_model=model,
        served_model=raw.get('model'),decision=answer,latency_s=time.monotonic()-start,
        usage=raw.get('usage',{}))


def locate_visual_task_target(image_bytes, instruction, target_kind, *, timeout_s=90):
    """Return an image-space evidence box, not a metric goal or robot command."""
    if target_kind not in ('person','trolley'):raise ValueError('unsupported target kind')
    key=os.environ.get('DEEPSEEK_API_KEY','')
    if not key:raise RuntimeError('DEEPSEEK_API_KEY unavailable')
    model=os.environ.get('DEEPSEEK_VISION_MODEL','deepseek-flash')
    prompt=('Locate the explicitly requested object in the actual robot camera image. '
        'Separate a person from an equipment trolley: the trolley is the physical wheeled shelving cart, '
        'not a person near equipment and not an inferred equipment area. '
        'Visible objects may be clipped by the image boundary. Return its visible bounding box, '
        'normalized to image width/height, in [x_min,y_min,x_max,y_max]. '
        'If the requested object is not visible, return null. Do not choose a route or invent coordinates. '
        'Return exactly one JSON object: {"target_kind":"person|trolley","bbox": [numbers] or null,"reason":"short visual evidence"}.')
    payload=dict(model=model,stream=False,response_format={'type':'json_object'},messages=[
        dict(role='system',content=prompt),dict(role='user',content=[
            dict(type='text',text=json.dumps(dict(instruction=instruction,target_kind=target_kind),ensure_ascii=False)),
            dict(type='image_url',image_url=dict(url='data:image/jpeg;base64,'+base64.b64encode(image_bytes).decode(),detail='high'))])])
    base=os.environ.get('DEEPSEEK_BASE_URL','https://api.deepseek.com').rstrip('/')
    if not base.startswith('https://'):raise ValueError('HTTPS endpoint required')
    req=Request(base+'/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
    start=time.monotonic()
    try:
        with urlopen(req,timeout=timeout_s) as response:raw=json.load(response)
    except HTTPError as exc:raise RuntimeError(f'DeepSeek HTTP {exc.code}') from None
    except (URLError,TimeoutError,OSError):raise RuntimeError('DeepSeek network/timeout failure') from None
    answer=json.loads(raw['choices'][0]['message']['content'])
    if set(answer)!={'target_kind','bbox','reason'} or answer['target_kind']!=target_kind:raise ValueError('invalid target evidence schema')
    box=answer['bbox']
    if box is not None:
        if not isinstance(box,list) or len(box)!=4 or any(not isinstance(x,(int,float)) or not 0<=x<=1 for x in box):raise ValueError('invalid normalized target box')
        if box[2]-box[0]<.02 or box[3]-box[1]<.02:raise ValueError('target evidence box too small')
    if not isinstance(answer['reason'],str) or len(answer['reason'])>1000:raise ValueError('invalid target reason')
    return dict(provider='deepseek_target_grounding',requested_model=model,served_model=raw.get('model'),evidence=answer,latency_s=time.monotonic()-start,usage=raw.get('usage',{}))
