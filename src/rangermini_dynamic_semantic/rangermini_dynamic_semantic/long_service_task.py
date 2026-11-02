"""One instruction, validated service dependencies, and feedback-driven room visits.

The language model chooses semantic stops, never coordinates or wheel commands.
Service completion requires external acknowledgement or explicitly tagged
simulation acknowledgement. Arrival alone does not complete pickup/delivery.
"""
import hashlib
import json
import math
import time
import xml.etree.ElementTree as ET
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from .delivery_execution_recovery import DeliveryExecutionHarness


def prepare_service_layout(world_path):
    """Clear the 614 service/turning area in the new long-task world only."""
    tree = ET.parse(world_path);world = tree.getroot().find('world');changes=[]
    for name, xy in [('lab_table_614',(3.24,-7.0)),
                     ('lab_monitor_614',(3.24,-7.0)),('lab_chair_614',(4.25,-6.5)),
                     ('office_chair_617',(9.1,2.7))]:
        model = world.find(f"model[@name='{name}']")
        if model is None:
            continue
        pose = model.find('pose');values=list(map(float,pose.text.split()));before=list(values)
        values[0],values[1]=xy;pose.text=' '.join(map(str,values))
        changes.append({'model':name,'before':before,'after':values})
    tree.write(world_path,encoding='unicode',xml_declaration=True)
    return {'layout_revision':'multiroom_service_v2','furniture_changes':changes,
            'reason':'614 turning clearance and 617 receiving approach clearance'}


def validate_task(plan, graph):
    if not isinstance(plan, dict) or plan.get('version') != 1:
        raise ValueError('task plan requires version=1')
    stops = plan.get('stops')
    if not isinstance(stops, list) or not 1 <= len(stops) <= 12:
        raise ValueError('task requires 1..12 ordered stops')
    inventory = set()
    for stop in stops:
        if not isinstance(stop, dict):
            raise ValueError('stop must be an object')
        room = stop.get('room')
        if not isinstance(room, str) or 'D'+room not in graph.doors:
            raise ValueError('unknown room: '+str(room))
        if not graph.remaining_connected('D'+room):
            raise ValueError('disconnected room: '+room)
        actions = stop.get('actions')
        if not isinstance(actions, list) or not 1 <= len(actions) <= 8:
            raise ValueError('each stop requires 1..8 service actions')
        for action in actions:
            if not isinstance(action, dict):
                raise ValueError('action must be an object')
            kind, item = action.get('type'), action.get('item')
            if kind not in ('pickup', 'deliver', 'notify') or not isinstance(item, str) or not item.strip():
                raise ValueError('action requires pickup/deliver/notify and item ID')
            if kind == 'pickup':
                if item in inventory:
                    raise ValueError('item already carried: '+item)
                inventory.add(item)
            elif kind == 'deliver':
                if item not in inventory:
                    raise ValueError('delivery before pickup: '+item)
                inventory.remove(item)
    return json.loads(json.dumps(plan))


def compile_task(instruction, graph, output_path):
    """Single real text-model request, archived without credentials."""
    from .deepseek_vision_provider import DeepSeekVisionProvider
    client = DeepSeekVisionProvider()
    if not client.configured:
        raise RuntimeError('DEEPSEEK_API_KEY required for one-instruction task parsing')
    schema = {'version': 1, 'stops': [{'room': 'registered room number',
              'actions': [{'type': 'pickup|deliver|notify', 'item': 'canonical item ID'}]}]}
    body = {'model': client.model, 'temperature': 0,
            'response_format': {'type': 'json_object'}, 'messages': [
        {'role': 'system', 'content': 'Parse the complete indoor service instruction into an ordered task. '
         'Use only registered rooms. Preserve all requested stops, action order and item identity. '
         'Use a single canonical ID for an item across pickup and delivery. Inventory starts empty. '
         'Do not invent missing pickups, services, poses, paths, room numbers or wheel modes. '
         'If ambiguous return {"error":"clarification needed"}. Return JSON only. Schema: '+json.dumps(schema)},
        {'role': 'user', 'content': json.dumps({'instruction': instruction,
            'registered_rooms': sorted(d[1:] for d in graph.doors)}, ensure_ascii=False)}]}
    req = Request(client.base_url+'/chat/completions', data=json.dumps(body).encode(),
        headers={'Authorization': 'Bearer '+client.api_key, 'Content-Type': 'application/json'})
    started = time.monotonic()
    for attempt in range(2):
        try:
            with urlopen(req, timeout=client.timeout_s) as response:
                raw = json.loads(response.read())
            break
        except HTTPError:
            raise
        except (URLError, TimeoutError, OSError):
            if attempt == 1:
                raise
    archive = {'provider': 'deepseek_task_parser', 'instruction': instruction,
               'request': body, 'response': raw, 'http_latency_s': time.monotonic()-started,
               'http_attempts': attempt+1}
    output_path.write_text(json.dumps(archive, ensure_ascii=False, indent=2))
    plan = validate_task(json.loads(raw['choices'][0]['message']['content']), graph)
    plan['instruction'] = instruction
    plan['source'] = 'real_language_model'
    return plan


class LongServiceTaskHarness(DeliveryExecutionHarness):
    def __init__(self, graph, instruction, plan, emit=None, confirmation='external', **kwargs):
        self.task_plan = validate_task(plan, graph)
        self.service_poses = {}
        for stop_number, stop in enumerate(self.task_plan['stops']):
            room = stop['room'];door = graph.doors['D'+room]
            old = graph.nodes['H'+room]['pose']
            candidates = [old] + [[door['center_x']+.01,
                door['plane_y']+door['direction']*depth,door['direction']*math.pi/2]
                for depth in (.65,.55,.85,1.,1.25)]
            def supported_pose(q):
                if not graph.supports_pose(q, margin=.10):
                    return False
                if stop_number < len(self.task_plan['stops'])-1:
                    return all(graph.supports_pose((q[0],q[1],math.radians(deg)),margin=.10)
                               for deg in range(0,360,10))
                return True
            supported = next((q for q in candidates if supported_pose(q)), None)
            if supported is None:
                raise ValueError('no footprint-supported service pose at room '+room)
            graph.nodes['H'+room]['pose'] = list(supported)
            self.service_poses[room] = list(supported)
        self.stop_index = 0
        self.action_index = 0
        self.inventory = set()
        self.completed_actions = []
        self.receipt_tokens = set()
        self.confirmation = confirmation
        self.service_since = None
        self.task_id = hashlib.sha256((instruction+json.dumps(self.task_plan, sort_keys=True)).encode()).hexdigest()[:16]
        first = self.task_plan['stops'][0]['room']
        super().__init__(graph, instruction, emit, destination_room=first, **kwargs)
        self._event('LONG_TASK_REGISTERED', 0, task_id=self.task_id, task_plan=self.task_plan,
                    confirmation_mode=confirmation)

    @property
    def service_active(self):
        return self.stage == 'SERVICE_CONFIRM' or self.stage.startswith('EXIT_')

    @property
    def anchor_id(self):
        if self.stage == 'SERVICE_CONFIRM':
            return 'H'+self.portal[1:]
        if self.stage.startswith('EXIT_'):
            return self.portal
        return super().anchor_id

    def subgoal(self):
        if self.stage == 'CROSS':
            door = self.graph.doors[self.portal]
            pose = self.graph.nodes['H'+self.portal[1:]]['pose']
            depth = min(1., abs(pose[1]-door['plane_y']))
            return (door['center_x']+.01, door['plane_y']+door['direction']*depth), pose[2]
        if self.stage == 'SERVICE_CONFIRM':
            pose = self.graph.nodes['H'+self.portal[1:]]['pose']
            return tuple(pose[:2]), pose[2]
        if self.stage.startswith('EXIT_'):
            door = self.graph.doors[self.portal]
            x, y, d = door['center_x']+.01, door['plane_y'], door['direction']
            depth = abs(self.graph.nodes['H'+self.portal[1:]]['pose'][1]-y)
            if self.stage == 'EXIT_CENTER':
                return (x, y+d*depth), None
            if self.stage == 'EXIT_ALIGN':
                return (x, y+d*depth), -d*math.pi/2
            if self.stage == 'EXIT_CROSS':
                return (x, y-d*1.), -d*math.pi/2
            return (x, -2.1), None
        return super().subgoal()

    def at_service_pose(self, pose):
        if self.stage != 'SERVICE_CONFIRM':
            return False
        goal, yaw = self.subgoal()
        error = abs(math.atan2(math.sin(pose[2]-yaw), math.cos(pose[2]-yaw)))
        return math.dist(pose[:2], goal) <= .32 and error <= math.radians(18)

    def confirm(self, receipt, stamp):
        stop = self.task_plan['stops'][self.stop_index]
        if (self.stage != 'SERVICE_CONFIRM' or self.service_since is None or
                receipt.get('task_id') != self.task_id or receipt.get('stop_index') != self.stop_index or
                receipt.get('action_index') != self.action_index or receipt.get('room') != stop['room'] or
                receipt.get('confirmed') is not True or not receipt.get('token') or
                receipt['token'] in self.receipt_tokens):
            self._event('SERVICE_CONFIRMATION_REJECTED', stamp, stop_index=self.stop_index)
            return False
        action = stop['actions'][self.action_index]
        if action['type'] == 'deliver' and action['item'] not in self.inventory:
            raise RuntimeError('runtime inventory dependency violated')
        if action['type'] == 'pickup':
            self.inventory.add(action['item'])
        elif action['type'] == 'deliver':
            self.inventory.remove(action['item'])
        self.receipt_tokens.add(receipt['token'])
        event = self._event('SERVICE_ACTION_CONFIRMED', stamp, task_id=self.task_id,
            stop_index=self.stop_index, room=stop['room'], action_index=self.action_index,
            action=action, confirmation_source=receipt.get('source', 'external_service'),
            token=receipt['token'], inventory=sorted(self.inventory))
        self.stage_events.append(event)
        self.completed_actions.append(event)
        self.action_index += 1
        self.service_since = None
        if self.action_index == len(stop['actions']):
            self.stage = 'DONE' if self.stop_index == len(self.task_plan['stops'])-1 else 'EXIT_CENTER'
            self._release_binding('SERVICE_STOP_COMPLETED', stamp, stop_index=self.stop_index)
        return True

    def reassess_local_preference(self, pose, stamp):
        if self.selected_behavior not in ('YIELD_LEFT','YIELD_RIGHT','PASS_LEFT','PASS_RIGHT'):
            return
        if self.last_bound_stamp is None:
            return
        age = stamp-self.last_bound_stamp
        reached = (self.bound_behavior_goal is not None and
                   math.dist(pose[:2], self.bound_behavior_goal) <= .20)
        if age >= 8.0 or (age >= 2.0 and reached):
            self._release_binding('LOCAL_PREFERENCE_REASSESSMENT_REQUIRED', stamp,
                guide_reached=reached, preference_age_s=age,
                meaning='re-evaluate temporary preference; not evidence of obstacle passage')

    def feedback(self, pose, stamp, tracks, executed_speed=0):
        if self.stage == 'SERVICE_CONFIRM':
            ready = self.at_service_pose(pose) and executed_speed < .025
            self.service_since = ((self.service_since if self.service_since is not None else stamp)
                                  if ready else None)
            if (self.confirmation == 'simulation' and self.service_since is not None and
                    stamp-self.service_since >= .6):
                self.confirm({'task_id': self.task_id, 'room': self.portal[1:],
                    'stop_index': self.stop_index, 'action_index': self.action_index,
                    'confirmed': True, 'token': f'sim:{self.stop_index}:{self.action_index}',
                    'source': 'simulated_service_acknowledgement'}, stamp)
            return
        if self.stage.startswith('EXIT_'):
            previous = self.previous_pose
            # Preserve shared interaction leases and observation-based release.
            super().feedback(pose, stamp, tracks, executed_speed)
            self.reassess_local_preference(pose, stamp)
            goal, yaw = self.subgoal()
            dist = math.dist(pose[:2], goal)
            old = self.stage
            if old == 'EXIT_CENTER' and dist <= .12:
                self.stage = 'EXIT_ALIGN'
            elif old == 'EXIT_ALIGN':
                err = abs(math.atan2(math.sin(pose[2]-yaw), math.cos(pose[2]-yaw)))
                if dist <= .15 and err <= math.radians(10):
                    self.stage = 'EXIT_CROSS'
            elif old == 'EXIT_CROSS' and previous is not None:
                door = self.graph.doors[self.portal];d=door['direction']
                before=d*(previous[1]-door['plane_y']);after=d*(pose[1]-door['plane_y'])
                if before >= 0 > after:
                    f=before/(before-after);x=previous[0]+f*(pose[0]-previous[0])
                    if door['aperture'][0] <= x <= door['aperture'][1]:
                        self.stage='EXIT_CLEAR'
            elif old == 'EXIT_CLEAR' and dist <= .25:
                self.stop_index += 1;self.action_index=0
                room=self.task_plan['stops'][self.stop_index]['room']
                self.portal='D'+room;self.destination='R'+room
                self.stage='APPROACH';self.alignment_phase='POSITION';self.arrival_since=None
            if self.stage != old:
                self.stage_events.append(self._event('SERVICE_ROUTE_ADVANCED', stamp,
                    old_stage=old, stage=self.stage, stop_index=self.stop_index,
                    measured_pose=list(pose)))
                self._release_binding('LONG_TASK_CONTEXT_CHANGED', stamp)
            self.previous_pose=tuple(pose)
            return
        super().feedback(pose, stamp, tracks, executed_speed)
        self.reassess_local_preference(pose, stamp)
        if self.stage == 'DONE':
            self.stage='SERVICE_CONFIRM';self.service_since=None
            self._release_binding('SERVICE_CONFIRMATION_REQUIRED', stamp,
                stop_index=self.stop_index, room=self.portal[1:])

    def observed_interaction_clear(self, pose, relevant):
        goal, _ = self.subgoal()
        dx, dy = goal[0]-pose[0], goal[1]-pose[1]
        length = math.hypot(dx,dy)
        if length < .15:
            dx,dy = math.cos(pose[2]),math.sin(pose[2])
        else:
            dx,dy = dx/length,dy/length
        return bool(relevant) and all(
            (t.position[0]-pose[0])*dx+(t.position[1]-pose[1])*dy < -.75 or
            abs(-(t.position[0]-pose[0])*dy+(t.position[1]-pose[1])*dx) > 1.25
            for t in relevant)

    def context(self, pose, tracks):
        result=super().context(pose,tracks)
        result['long_task']={'task_id':self.task_id,'instruction':self.instruction,
            'stop_index':self.stop_index,'action_index':self.action_index,
            'current_stop':self.task_plan['stops'][self.stop_index],
            'remaining_stops':self.task_plan['stops'][self.stop_index:],
            'inventory':sorted(self.inventory),'confirmed_actions':len(self.completed_actions),
            'confirmation_mode':self.confirmation,
            'service_ready': self.stage == 'SERVICE_CONFIRM' and self.service_since is not None,
            'service_pose_source':'registered geometry with oriented-footprint support',
            'service_poses':self.service_poses}
        return result


from .footprint_route_guide import FootprintRouteGuide


class ObservedRouteGuide(FootprintRouteGuide):
    """Refresh the existing reference search with current observed occupancy.

    This is an optional long-task routing input, not a new motion optimizer.
    Dynamic prediction and final collision checks remain in MPPI.
    """
    def __init__(self, graph, **kwargs):
        super().__init__(graph, **kwargs)
        self.registered_boxes = list(graph.static_boxes)
        self.track_signature = None

    def update_tracks(self, tracks, stamp):
        import numpy as np
        boxes = [(float(t.position[0]),float(t.position[1]),0.,float(t.radius)+.20,float(t.radius)+.20)
                 for t in tracks if 0 <= stamp-t.stamp <= .65]
        signature = tuple(tuple(round(v,1) for v in box) for box in boxes)
        if signature == self.track_signature:
            return False
        self.track_signature=signature
        self.boxes=np.asarray(self.registered_boxes+boxes)
        self.cache.clear();self.pose_grid=None
        return True
