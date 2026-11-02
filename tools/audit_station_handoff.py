#!/usr/bin/env python3
"""Score the requested desk relation from measured poses, not selected labels."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src/rangermini_dynamic_semantic'),
               str(ROOT/'src/rangermini_dynamic_semantic/scripts')]
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph
from audit_grounded_delivery import rect_gap


def audit(work, requested):
    work = Path(work).resolve()
    manifest = json.loads((work/'episode/manifest.json').read_text())
    result = json.loads((work/'episode/result.json').read_text())
    if not manifest.get('station_handoff'):
        raise ValueError('not a receiving-station episode')
    if requested == 'front' and '正面' not in manifest['instruction']:
        raise ValueError('front relation was not requested in this registered instruction')
    if requested == 'left' and '左侧' not in manifest['instruction']:
        raise ValueError('left relation was not requested in this registered instruction')
    world = ET.parse(work/'world.sdf').getroot().find('world')
    desk = list(map(float, world.find("model[@name='office_desk_619']/pose").text.split()))
    graph = SpatialGraph(work/'world.sdf')
    scene=json.loads((work/'scene_manifest.json').read_text())
    if 'station_registration' not in scene:
        raise ValueError('legacy pilot lacks a frozen station registration; preserve its original audit')
    target=tuple(scene['station_registration']['target_position'])
    bearing = math.atan2(desk[1]-target[1], desk[0]-target[0])
    yaw = bearing-(math.pi/2 if requested == 'left' else 0.)
    rows = [json.loads(s) for s in (work/'episode/trace.jsonl').read_text().splitlines()]
    def error(a):
        return abs(math.atan2(math.sin(a-yaw), math.cos(a-yaw)))
    # Last approach to the receiver: use measured distance and the final entry
    # into the registered radius, not the controller's phase/selected label.
    radius=scene['station_registration']['approach_region']['last_entry_radius_m']
    outside=[i for i,r in enumerate(rows) if math.dist(r['pose'][:2],target)>radius]
    approach=rows[(outside[-1]+1) if outside else 0:]
    gaps = [min(rect_gap(r['pose'], (x,y,a), bl=2*hx, bw=2*hy)
                for x,y,a,hx,hy in graph.static_boxes) for r in rows]
    final = result['final_pose']
    position = math.dist(final[:2], target)
    heading = error(final[2])
    max_heading = max((error(r['pose'][2]) for r in approach), default=None)
    position_ok = position <= .10
    relation_ok = heading <= math.radians(10)
    process_ok = bool(approach) and max_heading <= math.radians(10)
    no_overlap = bool(gaps) and min(gaps) >= -.01
    arrived = result['status'] == 'ARRIVED' and result['task_stage'] == 'DONE'
    success = arrived and position_ok and relation_ok and process_ok and no_overlap
    answer = dict(protocol='office619_station_handoff_v1', requested_relation=requested,
        task_success=success, verified_stop=arrived, target_position=target,
        requested_heading_rad=yaw, final_position_error_m=position,
        final_heading_error_deg=math.degrees(heading),
        approach_max_heading_error_deg=None if max_heading is None else math.degrees(max_heading),
        approach_observation_count=len(approach), receiving_relation_satisfied=relation_ok,
        heading_retained_during_approach=process_ok, position_satisfied=position_ok,
        static_proxy_collision_free=no_overlap, static_min_signed_gap_m=min(gaps) if gaps else None,
        failure_categories=[name for name,ok in [('arrival',arrived),('position',position_ok),
            ('requested_receiving_relation',relation_ok),('approach_heading',process_ok),
            ('static_overlap',no_overlap)] if not ok],
        evaluation='logged physical pose and registered scene geometry; model wheel states are not measured actuator feedback')
    files=[work/'world.sdf',work/'scene_manifest.json',work/'episode/manifest.json',work/'episode/result.json',
           work/'episode/trace.jsonl',Path(__file__).resolve()]
    answer['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (work/'station_task_audit.json').write_text(json.dumps(answer,ensure_ascii=False,indent=2))
    return answer


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('run', type=Path)
    ap.add_argument('--requested', choices=['front','left'], required=True)
    args=ap.parse_args()
    print(json.dumps(audit(args.run,args.requested),ensure_ascii=False,indent=2))
