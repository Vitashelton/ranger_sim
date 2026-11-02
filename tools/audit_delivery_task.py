#!/usr/bin/env python3
"""Task outcomes from logged poses, independent actor acknowledgements and geometry.

No planner costs or behavior labels enter the pass/fail predicate. Priority
is a registered instruction condition, not an inferred success label.
"""
import argparse,bisect,hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src/rangermini_dynamic_semantic/scripts'),str(ROOT/'src/rangermini_dynamic_semantic')]
from audit_grounded_delivery import audit,rect_gap,person_gap
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph

def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def audit_task(work,priority_required):
    work=Path(work).resolve();basic=audit(work);scene=json.loads((work/'scene_manifest.json').read_text())
    protocol=scene['task_protocol']
    manifest=json.loads((work/'episode/manifest.json').read_text())
    if priority_required and '先在器材通道外让推车完全驶出通道' not in manifest.get('instruction',''):
        raise ValueError('Priority scoring requires the registered priority instruction; do not relabel ordinary trials')
    trace=rows(work/'episode/trace.jsonl');fixture=rows(work/'fixture_truth.jsonl')
    actor=[r for r in fixture if r.get('acknowledged') and r['entity']=='service_trolley'];times=[r['sim_time'] for r in actor]
    graph=SpatialGraph(work/'world.sdf');xmin,xmax=protocol['aisle_x'];ymin,ymax=protocol['aisle_y']
    def rectangle(pose):
        x,y,a=pose;c,s=math.cos(a),math.sin(a)
        return [(x+c*dx-s*dy,y+s*dx+c*dy) for dx,dy in [(-.751/2,-.5/2),(.751/2,-.5/2),(.751/2,.5/2),(-.751/2,.5/2)]]
    actor_cleared=None;entry=None;violations=0;static_min=float('inf');static_overlaps=0;first_fault=None;records=[]
    # A cart travels toward negative x; its rear is max world x.
    for r in actor:
        x,y,a=r['pose'];rear=x+.70/2*abs(math.cos(a))+.42/2*abs(math.sin(a))
        if rear<xmin-.02:actor_cleared=r['sim_time'];break
    for r in trace:
        poly=rectangle(r['pose']);xs=[p[0] for p in poly];ys=[p[1] for p in poly]
        # Positive overlap of the physical footprint with the registered aisle.
        inside=rect_gap(r['pose'],[(xmin+xmax)/2,(ymin+ymax)/2,0],bl=xmax-xmin,bw=ymax-ymin)<-.01
        if inside and entry is None:entry=r['timestamp']
        early=inside and (actor_cleared is None or r['timestamp']<actor_cleared)
        violations+=int(early)
        gap=min(rect_gap(r['pose'],[x,y,a],bl=2*hx,bw=2*hy) for x,y,a,hx,hy in graph.static_boxes)
        static_min=min(static_min,gap);static_overlaps+=int(gap<=0)
        if first_fault is None and gap<=0:first_fault=dict(kind='static_proxy_overlap',elapsed_s=r['elapsed'])
        records.append(dict(elapsed_s=r['elapsed'],in_aisle=inside,priority_violation=early,static_gap_m=gap))
    origin=trace[0]['timestamp']-trace[0]['elapsed']
    observed_passage=actor_cleared is not None
    compliant=bool(observed_passage and violations==0) if priority_required else None
    collision_free=bool(basic['fixture_audit_valid'] and basic['proxy_overlap_trace_count']==0 and static_overlaps==0)
    success=bool(basic['navigation_success'] and collision_free and (not priority_required or compliant))
    path=sum(math.dist(a['pose'][:2],b['pose'][:2]) for a,b in zip(trace,trace[1:]))
    if not basic['fixture_audit_valid']:failure='invalid_actor_record'
    elif not collision_free:failure='evaluated_collision'
    elif priority_required and not compliant:failure='priority_not_satisfied'
    elif not basic['navigation_success']:failure='delivery_not_completed'
    else:failure=None
    files=[work/'world.sdf',work/'scene_manifest.json',work/'episode/trace.jsonl',work/'episode/result.json',work/'episode/manifest.json',work/'fixture_truth.jsonl']
    result=dict(protocol=protocol['version'],instruction_condition='priority' if priority_required else 'ordinary',
        task_success=success,verified_arrival=bool(basic['navigation_success']),collision_free=collision_free,
        priority_satisfied=compliant,priority_violation_samples=violations if priority_required else None,
        observed_priority_rule_satisfied=bool(observed_passage and violations==0),
        actor_passage_observed=observed_passage,actor_cleared_elapsed_s=None if actor_cleared is None else actor_cleared-origin,
        first_aisle_entry_elapsed_s=None if entry is None else entry-origin,
        completion_time_s=basic['elapsed_sim_s'] if success else None,terminal_elapsed_s=basic['elapsed_sim_s'],
        path_length_m=path,failure_category=failure,static_min_signed_gap_m=static_min,
        actor_min_signed_gap_m=min(basic['trolley_signed_SAT_min_gap_m'],basic['person_signed_min_gap_m']),
        static_overlap_samples=static_overlaps,actor_overlap_samples=basic['proxy_overlap_trace_count'],
        qualification='logged-pose proxy collision checks; not continuous mesh or hardware safety validation',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (work/'task_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    (work/'task_event_trace.json').write_text(json.dumps(records,separators=(',',':')))
    return result
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('--priority',action='store_true');a=ap.parse_args()
    print(json.dumps(audit_task(a.run,a.priority),ensure_ascii=False,indent=2))
