"""Re-query an archived real observation; retain source hashes and raw outputs.

This is an interface audit, never a new closed-loop navigation episode.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/rangermini_dynamic_semantic'))
from rangermini_dynamic_semantic.candidate_view import render_candidate_view
from rangermini_dynamic_semantic.grounded_visual_proposal import propose_task_condition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--query', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--query-vlm', action='store_true')
    parser.add_argument('--instruction', help='Natural-language instruction contrast on the identical observation')
    args = parser.parse_args()
    args.episode = args.episode.resolve()
    args.output = args.output.resolve()
    stem = args.episode/f'proposal_{args.query:03d}'
    source = Path(str(stem)+'_input.json')
    raw_rgb = Path(str(stem)+'_raw_rgb.jpg')
    rollouts = Path(str(stem)+'_rollouts.json')
    context = json.loads(source.read_text())['context']
    if args.instruction:
        context['instruction'] = args.instruction
    rgb = cv2.cvtColor(cv2.imread(str(raw_rgb)), cv2.COLOR_BGR2RGB)
    candidates = []
    for parent in json.loads(rollouts.read_text()):
        for family in parent.get('native_family_candidates', []):
            candidates.append(dict(behavior_id=f'SEQ_{len(candidates)}',
                motion_behavior=family['behavior'], motion_sequence=[], switch_times_s=[],
                feasible=True, trajectory=family['trajectory'], cost=family['cost'],
                predicted_clearance_m=family['predicted_min_clearance_m']))
    objects = [dict(position=t['position'], radius=t['radius'], evidence_id=t['evidence_id'])
               for t in context['objects']]
    image, visibility = render_candidate_view(rgb, context['robot_pose'], candidates, objects)
    for candidate, view in zip(candidates, visibility):
        candidate.update(candidate_id=view['candidate_id'],
                         visible_camera_segments=view['visible_segments'])
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.imwrite(str(args.output/'vlm_input.jpg'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    (args.output/'input.json').write_text(json.dumps(dict(context=context, candidates=candidates),
                                                   ensure_ascii=False, indent=2)+'\n')
    inputs = [source, raw_rgb, rollouts, Path(__file__),
              ROOT/'src/rangermini_dynamic_semantic/rangermini_dynamic_semantic/grounded_visual_proposal.py',
              ROOT/'src/rangermini_dynamic_semantic/rangermini_dynamic_semantic/candidate_view.py']
    provenance = dict(kind='archived RGB/rollout interface audit; not a new executed episode',
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in inputs},
        candidate_count=len(candidates),
        visible_camera_segments=sum(v['visible_segments'] for v in visibility))
    if args.query_vlm:
        ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError('JPEG encoding failed')
        response = propose_task_condition(bytes(encoded), context, candidates)
        (args.output/'response.json').write_text(json.dumps(response, ensure_ascii=False, indent=2)+'\n')
        provenance['proposal'] = response['proposal']
    (args.output/'audit.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in provenance.items() if k != 'source_sha256'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
