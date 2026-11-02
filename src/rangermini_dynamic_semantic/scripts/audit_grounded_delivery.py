#!/usr/bin/env python3
"""Independent moving-proxy audit from acknowledged fixture poses and traces."""
import argparse
import json
import math
from pathlib import Path
import numpy as np


def rect_gap(a, b, al=.751, aw=.500, bl=.70, bw=.42):
    def axes(theta):
        return [(math.cos(theta), math.sin(theta)), (-math.sin(theta), math.cos(theta))]
    aa, ab = axes(a[2]), axes(b[2])
    separation = -float('inf')
    for ax, ay in aa+ab:
        ra = al/2*abs(ax*aa[0][0]+ay*aa[0][1])+aw/2*abs(ax*aa[1][0]+ay*aa[1][1])
        rb = bl/2*abs(ax*ab[0][0]+ay*ab[0][1])+bw/2*abs(ax*ab[1][0]+ay*ab[1][1])
        separation = max(separation, abs((b[0]-a[0])*ax+(b[1]-a[1])*ay)-ra-rb)
    return separation


def person_gap(pose, xy):
    x, y, yaw = pose
    dx, dy = xy[0]-x, xy[1]-y
    c, s = math.cos(yaw), math.sin(yaw)
    return math.hypot(max(abs(c*dx+s*dy)-.751/2, 0),
                      max(abs(-s*dx+c*dy)-.5/2, 0))-.18


def load_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def audit(path):
    root = Path(path)
    result = json.loads((root/'episode/result.json').read_text())
    scene = json.loads((root/'scene_manifest.json').read_text())
    rows = load_rows(root/'episode/trace.jsonl')
    fixture = load_rows(root/'fixture_truth.jsonl') if (root/'fixture_truth.jsonl').exists() else []
    acknowledged = [r for r in fixture if r.get('acknowledged')]
    groups = {name: [r for r in acknowledged if r['entity'] == name]
              for name in ('service_trolley', 'lab_student')}
    def actor(name, stamp):
        group = groups[name]
        initial = scene['trolley_initial' if name == 'service_trolley' else 'person_initial']
        if not group or stamp < group[0]['sim_time']:
            return [*initial, 0]
        # Acknowledged fixture states are held until the next application;
        # this matches the kinematic actor rather than inventing smooth motion.
        times = [g['sim_time'] for g in group]
        index = int(np.searchsorted(times, stamp, side='right'))-1
        return group[max(0, index)]['pose']
    records = []
    for row in rows:
        stamp = row['timestamp']
        trolley = actor('service_trolley', stamp)
        person = actor('lab_student', stamp)
        records.append({'timestamp': stamp, 'elapsed': row['elapsed'],
            'trolley_signed_SAT_gap_m': rect_gap(row['pose'], trolley),
            'person_signed_gap_m': person_gap(row['pose'], person),
            'trolley_pose': trolley, 'person_pose': person})
    trolley_min = min((r['trolley_signed_SAT_gap_m'] for r in records), default=None)
    person_min = min((r['person_signed_gap_m'] for r in records), default=None)
    overlap = sum(r['trolley_signed_SAT_gap_m'] <= 0 or r['person_signed_gap_m'] <= 0 for r in records)
    moving_expected = scene['scenario'] != 'plain'
    fixture_ok = not moving_expected or (len(acknowledged) >= 10 and
                    len(acknowledged)/max(1, len(fixture)) > .95)
    answer = {'navigation_success': result['task_success'],
        'interaction_success': bool(result['task_success'] and fixture_ok and records and overlap == 0),
        'trolley_signed_SAT_min_gap_m': trolley_min, 'person_signed_min_gap_m': person_min,
        'proxy_overlap_trace_count': overlap, 'fixture_acknowledged': len(acknowledged),
        'fixture_requests': len(fixture), 'fixture_audit_valid': fixture_ok,
        'trace_count': len(rows),
        'measurement': 'signed rectangle SAT separation and rectangle-circle gap; proxies, not complete meshes',
        'source': 'independent acknowledged scene fixture; never used by online policy'}
    (root/'interaction_audit.json').write_text(json.dumps(answer, ensure_ascii=False, indent=2))
    (root/'clearance_trace.json').write_text(json.dumps(records, separators=(',', ':')))
    return {**result, **answer, 'run_path': str(root)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+')
    args = parser.parse_args()
    for path in args.runs:
        print(json.dumps(audit(path), ensure_ascii=False))
