"""Check same-run language/vision/motion evidence and independent geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src/rangermini_dynamic_semantic/scripts'),
               str(ROOT/'src/rangermini_dynamic_semantic')]
from audit_grounded_delivery import audit, rect_gap
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def main(run):
    run = run.resolve()
    interaction = audit(run)
    summary = json.loads((run/'long_task_summary.json').read_text())
    parser = json.loads((run/'task_parser_response.json').read_text())
    trace = rows(run/'episode/trace.jsonl')
    events = rows(run/'episode/selection_events.jsonl')
    graph = SpatialGraph(run/'world.sdf')
    protocol=json.loads((run.parent/'protocol.json').read_text())
    manifest=json.loads((run/'scene_manifest.json').read_text())
    unrestricted='--unrestricted-task-motion' in protocol['command']
    gaps = [min(rect_gap(r['pose'], box[:3], bl=2*box[3], bw=2*box[4])
                for box in graph.static_boxes) for r in trace]
    responses = sorted((run/'episode').glob('proposal_*_response.json'))
    bound = [e for e in events if e['event'] == 'TASK_CONDITION_BOUND']
    request_modes = []
    for path in responses:
        response = json.loads(path.read_text())
        assert response['provider'] == 'deepseek_task_condition'
        assert response['proposal']['motion_preference'] == 'auto'
        for attempt in response['attempts']:
            answer = json.loads(attempt['raw_content'])
            request_modes.append('motion_preference' in answer)
    frames = list((run/'episode').glob('proposal_*_raw_rgb.jpg'))
    source_files = [run/'task_parser_response.json', run/'long_task_plan.json',
                    run/'long_task_summary.json', run/'scene_manifest.json', run/'world.sdf',
                    run/'episode/result.json', run/'episode/trace.jsonl',
                    run/'episode/selection_events.jsonl', run/'fixture_truth.jsonl'] + responses + frames
    result = dict(kind='Integrated single-run evidence audit; not a comparative performance estimate',
        run=str(run.relative_to(ROOT)), fresh_language_parse=parser['provider']=='deepseek_task_parser',
        task_plan_replayed=summary['language_plan_reused'], task_complete=summary['task_complete'],
        confirmed_action_count=summary['confirmed_action_count'],
        confirmation_sources=summary['confirmation_sources'], physical_handling_validated=False,
        real_visual_responses=len(responses), accepted_spatial_conditions=len(bound),
        accepted_trolley_conditions=sum('trolley' in e['proposal']['object_labels'].values() for e in bound),
        raw_model_selected_motion_family=any(request_modes),
        task_stage_mode_filter_disabled=unrestricted,
        motion_executor=manifest.get('motion_executor',{}),
        minimum_static_signed_SAT_m=min(gaps), static_overlap_pose_count=sum(g<=0 for g in gaps),
        proxy_overlap_trace_count=interaction['proxy_overlap_trace_count'],
        fixture_audit_valid=interaction['fixture_audit_valid'],
        instruction=parser['instruction'], result=interaction['status'],
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
    result['integrated_qualified'] = bool(result['fresh_language_parse'] and not result['task_plan_replayed']
        and result['task_complete'] and result['real_visual_responses'] and result['accepted_trolley_conditions']
        and not result['raw_model_selected_motion_family'] and not result['static_overlap_pose_count']
        and not result['proxy_overlap_trace_count'] and result['fixture_audit_valid'])
    result['unrestricted_integrated_qualified']=bool(result['integrated_qualified'] and unrestricted)
    (run/'integrated_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    main(p.parse_args().run)
