#!/usr/bin/env python3
"""Summarize measured task progress and motion, including unfinished attempts."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/rangermini_dynamic_semantic'))
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run', type=Path)
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    source = args.run/'episode/trace.jsonl'
    trace = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    groups = {}
    for row in trace:
        groups.setdefault(row['task_stage'], []).append(row)
    segments = []
    for stage, rr in groups.items():
        length = sum(math.dist(a['pose'][:2], b['pose'][:2]) for a,b in zip(rr, rr[1:]))
        yaw_change = sum(abs(math.atan2(math.sin(b['pose'][2]-a['pose'][2]),
                            math.cos(b['pose'][2]-a['pose'][2]))) for a,b in zip(rr,rr[1:]))
        lateral, reverse = 0., 0.
        for a,b in zip(rr,rr[1:]):
            dt = b['timestamp']-a['timestamp']
            velocity = a.get('actuation',{}).get('executed',{})
            lateral += abs(velocity.get('vy',0.))*dt
            reverse += max(0.,-velocity.get('vx',0.))*dt
        modes = collections.Counter(r.get('actuation',{}).get('mode','unknown') for r in rr)
        segments.append(dict(stage=stage, first_elapsed_s=rr[0]['elapsed'],
            last_elapsed_s=rr[-1]['elapsed'], path_m=length,
            absolute_heading_change_deg=math.degrees(yaw_change),
            integrated_lateral_m=lateral, integrated_reverse_m=reverse,
            mode_observations=dict(modes), first_pose=rr[0]['pose'], last_pose=rr[-1]['pose']))
    result_file = args.run/'episode/result.json'
    result = json.loads(result_file.read_text()) if result_file.exists() else None
    audit_file = args.run/'interaction_audit.json'
    actor_audit = json.loads(audit_file.read_text()) if audit_file.exists() else None
    geometry = SpatialGraph(args.run/'world.sdf')
    static_violations = sum(not geometry.supports_pose(row['pose'], margin=0.) for row in trace)
    summary = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        status=result['status'] if result else 'INCOMPLETE',
        complete_task=bool(result and result['task_success']),
        collection_verified=bool(result and any(e.get('stop_id') == 'COLLECTION'
            for e in result['verified_stage_events'])),
        static_support_violations=static_violations,
        qualified_task=bool(result and result['task_success'] and actor_audit
            and actor_audit['interaction_success'] and static_violations == 0),
        segments=segments,
        note='Registered service-pose navigation. No manipulation or new visual API inference is implied.')
    output = args.run/'multistop_motion_summary.json'
    if args.check:
        assert json.loads(output.read_text()) == summary, 'summary differs from recorded motion'
    else:
        output.write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
