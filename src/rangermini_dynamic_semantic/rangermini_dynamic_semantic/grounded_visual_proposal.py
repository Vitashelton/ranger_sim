"""Real image-conditioned DeepSeek proposal with bounded navigation fields."""
import base64
import json
import os
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


def admissible_candidates(context, candidates):
    """The proposal domain must match the binding contract, not just geometry."""
    import math
    ids = [o['evidence_id'] for o in context['objects']]
    return [c for c in candidates if c['feasible'] and math.isfinite(c['cost'])
            and (c['behavior_id']=='CONTINUE' or ids or
                 (c.get('grounding_kind') == 'registered_station' and
                  c['behavior_id'] in context.get('station_choices', {})))]


def propose_navigation(jpeg, context, candidates, timeout=60):
    domain = admissible_candidates(context,candidates)
    if not domain:
        raise ValueError('no admissible proposal behavior')
    if len(domain)==1:
        response = propose_by_cost(jpeg,context,domain)
        response.update(provider='single_admissible_behavior',http_requests=0,
                        selection_source='binding_contract; no VLM call')
        return response
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY unavailable")
    model = os.environ.get("DEEPSEEK_VISION_MODEL", "deepseek-flash")
    prompt = (
        "You plan grounded navigation for an indoor document-delivery Ranger robot. "
        "Use the actual RGB image, current task stage, semantic graph, tracked-object evidence "
        "and recent execution events. The left panel is RGB with in-view trajectory segments; "
        "the right panel is a body-frame view of the SAME currently optimized MPPI candidates. "
        "C IDs link the legend and JSON. Footprints and arrows show body orientation; "
        "near-field lateral motions or rotation may be absent from the forward camera view. "
        "The legend gives each candidate's current motion family, not a mode you command directly. "
        "Decide how to continue the service task while respecting people, carts and doorway access. "
        "CONTINUE follows the active task anchor; PASS_LEFT/PASS_RIGHT generate a local lateral passing "
        "maneuver, not a predefined complete route; YIELD_LEFT/YIELD_RIGHT shift laterally "
        "without advancing into an approaching object; WAIT yields in the current position. "
        "Prefer progress when there is no relevant interaction; when a cart approaches head-on, "
        "do not wait directly in its path if a feasible lateral alternative exists. "
        "Select only an available feasible behavior. Use only listed object evidence IDs. "
        "Classify visible listed objects as person, trolley, or obstacle when supported by the image. "
        "Return JSON with exactly anchor_id, behavior_id, evidence_ids, object_labels, reason. "
        "anchor_id must equal the current task anchor. object_labels is a dict keyed by listed evidence IDs. "
        "Interaction choices require at least one evidence ID. No coordinates, velocities, or steering commands.")
    if context.get('station_choices'):
        prompt += (
            ' The current decision concerns a REGISTERED receiving desk, not a tracked moving object. '
            'Choose HANDOFF_FRONT or HANDOFF_LEFT from station_choices according to the instruction '
            'and desk/robot spatial relation in the observation. These labels denote the receiving '
            'side, NOT a chassis motion mode. The planner chooses the mode independently. '
            'The condition retains its selected heading throughout the final approach. '
            'First satisfy the requested receiving relation at the FINAL receiving pose. '
            'A lower trajectory cost or the CURRENT robot orientation must not override an explicit '
            'front/left receiving requirement. In HANDOFF_FRONT the robot nose points toward the '
            'desk at the receiving pose; HANDOFF_LEFT places the desk on the robot left. '
            'For these station choices return empty evidence_ids and object_labels; '
            'registered DESK619 is the spatial anchor, not a fabricated detection ID.')
    allowed_ids = [o['evidence_id'] for o in context['objects']]
    decision_input = {"context":context,"candidates":candidates,
        "decision_domain":{"anchor_id":context['anchor_id'],
                           "behavior_ids":[c['behavior_id'] for c in domain],
                           "evidence_ids":allowed_ids}}
    payload = {"model": model, "stream": False, "thinking": {"type": "disabled"},
        "max_tokens": 700, "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": [
            {"type": "text", "text": json.dumps(decision_input, ensure_ascii=False)},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"+base64.b64encode(jpeg).decode(),
                                                  "detail": "high"}}]}]}
    endpoint = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    if not endpoint.startswith("https://"):
        raise ValueError("HTTPS endpoint required")
    # One bounded repair, preserving the original observation and validation trail.
    # The caller still revalidates execution context after this worker returns.
    start = time.monotonic()
    attempts = []
    payload['messages'][0]['content'] += (
        ' Before answering, check all five required keys and all referenced IDs. '
        'reason must be one short sentence, at most 200 characters. '
        'Use only the decision_domain.behavior_ids and decision_domain.evidence_ids. '
        'Visible objects without a listed evidence ID cannot be referenced. '
        'Do not add fields or Markdown.')
    for attempt in range(2):
        remaining = timeout-(time.monotonic()-start)
        if remaining <= 0:
            raise NavigationProposalError('proposal deadline expired', attempts)
        req = Request(endpoint+'/chat/completions', data=json.dumps(payload).encode(),
            headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
        try:
            with urlopen(req, timeout=remaining) as response:
                raw = json.load(response)
        except HTTPError as exc:
            raise NavigationProposalError(f'DeepSeek HTTP {exc.code}', attempts, attempt+1) from None
        except (URLError, TimeoutError, OSError):
            raise NavigationProposalError('DeepSeek request transport failure', attempts, attempt+1) from None
        content = raw.get('choices',[{}])[0].get('message',{}).get('content','')
        record = {'attempt':attempt+1,'raw_content':content,'usage':raw.get('usage',{}),
                  'served_model':raw.get('model')}
        attempts.append(record)
        try:
            answer = validate_proposal(content, context, candidates)
        except (ValueError, TypeError, KeyError) as exc:
            record['validation_error'] = str(exc)
            if attempt:
                raise NavigationProposalError('proposal repair exhausted', attempts) from None
            payload['messages'].extend([
                {'role':'assistant','content':content if isinstance(content,str) else json.dumps(content)},
                {'role':'user','content': 'Validation failed: '+str(exc)+
                 '. Correct the JSON using only the original listed anchors, feasible behaviors and evidence IDs. '
                 'Return exactly anchor_id, behavior_id, evidence_ids, object_labels, reason. '
                 'Allowed domain: '+json.dumps(decision_input['decision_domain'])+
                 '. Keep reason under 200 characters.'}])
            continue
        record['accepted_schema'] = True
        usage = {name:sum(r['usage'].get(name,0) for r in attempts)
                 for name in ('prompt_tokens','completion_tokens','total_tokens')}
        return {'provider':'deepseek_grounded_navigation','requested_model':model,
                'served_model':raw.get('model'),'proposal':answer,
                'latency_s':time.monotonic()-start,'usage':usage,
                'http_requests':len(attempts),'attempts':attempts}


class NavigationProposalError(RuntimeError):
    def __init__(self, message, attempts, http_requests=None):
        super().__init__(message)
        self.attempts = attempts
        self.http_requests = len(attempts) if http_requests is None else http_requests


def replay_station_proposal(source, jpeg, context, candidates):
    """Control-mechanism pair: reuse one recorded real selector output exactly."""
    import hashlib
    from pathlib import Path
    raw = Path(source).read_bytes()
    record = json.loads(raw)
    if record.get('provider') != 'deepseek_grounded_navigation':
        raise ValueError('station replay requires an original real visual proposal')
    proposal = record['proposal']
    if proposal.get('anchor_id') != 'DESK619':
        raise ValueError('recorded proposal is not a receiving-station decision')
    validated = validate_proposal(json.dumps(proposal), context, candidates)
    return dict(provider='recorded_station_proposal', proposal=validated,
        source_sha256=hashlib.sha256(raw).hexdigest(), http_requests=0,
        latency_s=0., usage={}, selection_source='same recorded VLM output; no new inference')


def replay_task_condition(source, default_source, jpeg, context, candidates):
    """Apply recorded VLM conditions under the same live evidence gate."""
    import hashlib
    from pathlib import Path
    target = json.loads(Path(source).read_text())
    fallback = json.loads(Path(default_source).read_text())
    if target.get('provider') != 'deepseek_task_condition' or fallback.get('provider') != 'deepseek_task_condition':
        raise ValueError('task replay requires recorded visual task conditions')
    relevant = set(target['proposal']['relevant_object_ids'])
    available = {obj['evidence_id'] for obj in context.get('objects', [])}
    chosen_path = Path(source if relevant and relevant.issubset(available) else default_source)
    raw = chosen_path.read_bytes()
    proposal = dict(json.loads(raw)['proposal'])
    if context.get('spatial_conditions_only'):
        proposal.pop('motion_preference', None)
    validated = validate_task_condition(json.dumps(proposal), context)
    return dict(provider='recorded_task_condition', proposal=validated,
        source_sha256=hashlib.sha256(raw).hexdigest(), http_requests=0,
        latency_s=0., usage={}, selection_source='recorded VLM condition under live evidence gating')


def validate_proposal(content, context, candidates):
    answer = json.loads(content)
    required = {'anchor_id','behavior_id','evidence_ids','object_labels','reason'}
    if not isinstance(answer,dict) or set(answer)!=required:
        raise ValueError('expected exactly five navigation fields')
    if not isinstance(answer['reason'],str) or len(answer['reason'])>1000:
        raise ValueError('reason must be a string with at most 1000 characters')
    feasible = {c['behavior_id'] for c in admissible_candidates(context,candidates)}
    ids = {o['evidence_id'] for o in context['objects']}
    if answer['anchor_id'] != context['anchor_id']:
        raise ValueError('anchor_id must match current anchor')
    if not isinstance(answer['behavior_id'],str) or answer['behavior_id'] not in feasible:
        raise ValueError('behavior_id must be currently feasible')
    if not isinstance(answer['evidence_ids'],list) or any(not isinstance(e,str) or e not in ids for e in answer['evidence_ids']):
        raise ValueError('evidence_ids must reference listed objects')
    station = answer['behavior_id'] in context.get('station_choices', {})
    if answer['behavior_id']!='CONTINUE' and not station and not answer['evidence_ids']:
        raise ValueError('interaction behavior requires listed object evidence')
    labels = answer['object_labels']
    if not isinstance(labels,dict) or any(k not in ids or v not in ('person','trolley','obstacle') for k,v in labels.items()):
        raise ValueError('object_labels must use listed IDs and supported categories')
    return answer


def propose_by_cost(jpeg, context, candidates):
    """Matched selector ablation: same bank and proposal contract, no API."""
    import math
    start = time.monotonic()
    feasible = admissible_candidates(context,candidates)
    if not feasible:
        raise ValueError('no feasible candidate for cost selector')
    chosen = min(feasible, key=lambda c: (c['cost'], c['behavior_id']))
    ids = [o['evidence_id'] for o in context['objects']]
    answer = dict(anchor_id=context['anchor_id'], behavior_id=chosen['behavior_id'],
                  evidence_ids=ids if chosen['behavior_id'] != 'CONTINUE' and
                    chosen['behavior_id'] not in context.get('station_choices', {}) else [],
                  object_labels={}, reason='Minimum feasible MPPI bank cost')
    return dict(provider='matched_cost_selector', proposal=answer,
                latency_s=time.monotonic()-start, usage={}, api_request=False)


def propose_task_condition(jpeg, context, candidates, timeout=60, include_image=True):
    """Infer bounded task/spatial constraints; leave motion-family search to MPPI.

    The response can suggest a motion family as a soft objective preference.
    The hybrid planner evaluates all three Ranger families and switch times;
    it owns the executed sequence and continuous controls.
    """
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY unavailable")
    model = os.environ.get("DEEPSEEK_VISION_MODEL", "deepseek-flash")
    valid_ids = [o["evidence_id"] for o in context.get("objects", [])]
    spatial_only = bool(context.get('spatial_conditions_only', False))
    prompt = (
        "You ground an indoor service-navigation task for a four-wheel-steered Ranger robot. "
        "Interpret the complete instruction, current task stage, registered spatial anchor, "
        "RGB scene, tracked-object evidence, wheel state and recent execution feedback. "
        "The Ranger has three native motion families: dual-Ackermann curved driving, "
        "parallel translation (including oblique and lateral motion), and spinning. "
        "The local hybrid MPPI generates and compares sequences composed from these "
        "families, including their "
        "steering preparation and switching times. You may suggest a preferred motion family, "
        "but the optimizer checks all families and may override this soft preference. "
        "Do NOT output a velocity, steering command, or switching time. Return a bounded "
        "task condition: which listed objects matter to the current service objective, "
        "whether the task requires preserving the current robot heading during local motion "
        "or following the route/anchor heading, and a preferred clearance from the relevant "
        "objects. The planner decides whether and how any of its motion families realizes "
        "these requirements while retaining the registered task anchor. "
        "Use only listed evidence IDs. If the evidence list is empty, return relevant_object_ids as [] "
        "and object_labels as an empty JSON object {}; never invent an object ID. "
        "Track category 'obstacle' is a geometry-only placeholder, NOT a visual classification. "
        "Use the actual RGB appearance to classify each relevant listed object as trolley "
        "(wheeled cart, cabinet or shelving trolley), person, or obstacle (other/visually uncertain). "
        "object_labels must map every relevant_object_id to its visual class. "
        "Output exactly anchor_id, relevant_object_ids, object_labels, heading_policy, "
        "preferred_clearance_m, motion_preference, reason. "
        "heading_policy is route_heading or preserve_current_heading. "
        "motion_preference is auto, curved_driving, parallel_translation, or reorient_then_advance. "
        "Choose it from the instruction, visual context and actual candidate trajectories; "
        "parallel_translation includes oblique/lateral motion with unchanged body heading. "
        "Use auto when the scene provides no task reason to prefer a family. "
        "preferred_clearance_m is a real number in [0.55, 0.95]. "
        "The hard collision limit remains owned by the planner; this value is a soft "
        "preference. reason is at most 200 characters and briefly explains the visible object "
        "and task reason for the preference.")
    if spatial_only:
        prompt += (
            " Apply each preference only to the instruction's stated objects and execution context. "
            "A generic obstacle or a nearby wall, doorway or fixed furnishing does not automatically "
            "become a task-relevant interaction object. Keep such geometry in collision handling, "
            "but omit it from relevant_object_ids unless the instruction actually refers to it. "
            "Preserve heading during the requested encounter, not for every subsequent service stage. "
            "When the instruction permits heading adjustment for room entry/exit, use route_heading "
            "for that stage. A registered_static_map_geometry object is a known map anchor, not a "
            "new visual detection; use RGB to establish whether its appearance matches the requested object. ")
        prompt += (
            "Unspecified task preferences must not create extra requirements: for an instruction "
            "that only requests ordinary safe delivery, return relevant_object_ids=[], object_labels={}, "
            "heading_policy=route_heading and preferred_clearance_m=0.65. Collision avoidance still "
            "handles every observed obstacle. A request merely to navigate safely is not a request "
            "for additional object-specific clearance. When an explicit clearance value is requested, "
            "copy that value within the supported range; do not substitute a different preference. ")
        prompt = prompt.replace("The local hybrid MPPI generates and compares sequences composed from these "
            "families, including their ", "The downstream planner handles these capabilities, including ")
        prompt = prompt.replace(
            "steering preparation and switching times. You may suggest a preferred motion family, "
            "but the optimizer checks all families and may override this soft preference. ",
            "steering preparation and switching times. These capabilities belong to the optimizer; "
            "do not choose a motion family or candidate ID. ")
        prompt = prompt.replace("preferred_clearance_m, motion_preference, reason. ",
                                "preferred_clearance_m, reason. ")
        begin = prompt.index("motion_preference is auto,")
        end = prompt.index("preferred_clearance_m is a real", begin)
        prompt = prompt[:begin] + prompt[end:]
    prompt += " Return the result as a JSON object and no Markdown."
    candidate_fields = ("behavior_id", "feasible", "cost", "predicted_clearance_m",
                        "motion_behavior", "motion_sequence", "switch_times_s",
                        "candidate_id", "visible_camera_segments")
    compact_candidates = [{k: c[k] for k in candidate_fields if k in c}
                          for c in candidates]
    decision_input = {"context": context, "candidate_rollouts": compact_candidates,
        "motion_capabilities": ["curved_driving", "parallel_or_oblique_translation", "rotate_then_translate"],
        "decision_domain": {"anchor_id": context["anchor_id"],
                            "evidence_ids": valid_ids,
                            "heading_policy": ["route_heading", "preserve_current_heading"],
                            "preferred_clearance_range_m": [0.55, 0.95],
                            "motion_preference": ["auto", "curved_driving", "parallel_translation",
                                                  "reorient_then_advance"]}}
    if spatial_only:
        decision_input['decision_domain'].pop('motion_preference')
    payload = {"model": model, "stream": False, "temperature": 0, "thinking": {"type": "disabled"},
        "max_tokens": 500, "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": [
            {"type": "text", "text": json.dumps(decision_input, ensure_ascii=False)},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"+
                base64.b64encode(jpeg).decode(), "detail": "high"}}]}]}
    if not include_image:
        payload['messages'][1]['content'] = payload['messages'][1]['content'][:1]
    endpoint = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    if not endpoint.startswith("https://"):
        raise ValueError("HTTPS endpoint required")
    start = time.monotonic()
    attempts = []
    for attempt in range(2):
        remaining = timeout-(time.monotonic()-start)
        if remaining <= 0:
            raise NavigationProposalError("task-condition deadline expired", attempts)
        req = Request(endpoint+"/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer "+key})
        try:
            with urlopen(req, timeout=remaining) as response:
                raw = json.load(response)
        except HTTPError as exc:
            try:
                body = json.loads(exc.read().decode("utf-8", "replace"))
                detail = body.get("error", {}).get("message", "")
            except Exception:
                detail = ""
            detail = str(detail).replace("\n", " ")[:240]
            suffix = f": {detail}" if detail else ""
            raise NavigationProposalError(f"DeepSeek HTTP {exc.code}{suffix}", attempts, attempt+1) from None
        except (URLError, TimeoutError, OSError) as exc:
            attempts.append({"attempt": attempt+1, "transport_error": type(exc).__name__, "usage": {}})
            if attempt == 0 and timeout-(time.monotonic()-start) > 1.0:
                continue
            raise NavigationProposalError("DeepSeek request transport failure", attempts, attempt+1) from None
        content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        record = {"attempt": attempt+1, "raw_content": content, "usage": raw.get("usage", {}),
                  "served_model": raw.get("model")}
        attempts.append(record)
        try:
            proposal = validate_task_condition(content, context)
        except (ValueError, TypeError, KeyError) as exc:
            record["validation_error"] = str(exc)
            if attempt:
                raise NavigationProposalError("task-condition repair exhausted", attempts) from None
            payload["messages"].extend([
                {"role": "assistant", "content": content if isinstance(content, str) else json.dumps(content)},
                {"role": "user", "content": "Validation failed: "+str(exc)+
                 ". Return one corrected JSON object using exactly the required keys, current anchor, and only "
                 "listed evidence IDs. object_labels must be a JSON object (dictionary); when evidence is empty use {}. "+
                 ("Do not output a motion family, candidate ID, velocity or steering command."
                  if spatial_only else
                  "motion_preference is a soft family preference; do not output velocities or steering commands.")}])
            continue
        usage = {name: sum(r["usage"].get(name, 0) for r in attempts)
                 for name in ("prompt_tokens", "completion_tokens", "total_tokens")}
        return {"provider": "deepseek_task_condition", "requested_model": model,
            "served_model": raw.get("model"), "proposal": proposal,
            "latency_s": time.monotonic()-start, "http_requests": len(attempts),
            "attempts": attempts, "usage": usage}


def validate_task_condition(content, context):
    answer = json.loads(content)
    required = {"anchor_id", "relevant_object_ids", "object_labels",
                "heading_policy", "preferred_clearance_m", "motion_preference", "reason"}
    spatial_only = bool(context.get('spatial_conditions_only', False))
    if spatial_only:
        required.remove('motion_preference')
    if not isinstance(answer, dict) or set(answer) != required:
        raise ValueError("expected exactly the registered task-condition fields")
    if spatial_only:
        # Internal compatibility field, not a VLM-selected motion family.
        answer['motion_preference'] = 'auto'
    if answer["anchor_id"] != context.get("anchor_id"):
        raise ValueError("task condition anchor does not match active task")
    ids = {o["evidence_id"] for o in context.get("objects", [])}
    relevant = answer["relevant_object_ids"]
    if not isinstance(relevant, list) or any(not isinstance(i, str) or i not in ids for i in relevant):
        raise ValueError("task condition references unknown object evidence")
    if answer["heading_policy"] not in ("route_heading", "preserve_current_heading"):
        raise ValueError("unsupported heading policy")
    if answer["motion_preference"] not in ("auto", "curved_driving", "parallel_translation",
                                            "reorient_then_advance"):
        raise ValueError("unsupported motion preference")
    clearance = answer["preferred_clearance_m"]
    if isinstance(clearance, bool) or not isinstance(clearance, (int, float)) or not 0.55 <= clearance <= 0.95:
        raise ValueError("preferred clearance must lie in [0.55, 0.95] m")
    labels = answer["object_labels"]
    if labels == [] and not ids:
        labels = {}
        answer["object_labels"] = labels
    if not isinstance(labels, dict) or any(k not in ids or v not in ("person", "trolley", "obstacle") for k, v in labels.items()):
        raise ValueError("object labels must use listed IDs and supported classes")
    if set(labels) != set(relevant):
        raise ValueError("every relevant object must have exactly one visual class")
    if not isinstance(answer["reason"], str) or len(answer["reason"]) > 300:
        raise ValueError("reason must be a short string")
    return answer
