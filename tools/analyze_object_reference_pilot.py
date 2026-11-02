"""Separate visual referent correctness, spatial response and navigation outcome."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import getpass
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src/rangermini_dynamic_semantic/scripts'),
              str(ROOT/'src/rangermini_dynamic_semantic')]
from audit_grounded_delivery import audit,rect_gap
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph
from rangermini_dynamic_semantic.grounded_visual_proposal import propose_task_condition


def rows(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def object_truth(world):
    found={}
    root=ET.parse(world).getroot()
    for i,name in enumerate(('entry_equipment_station','entry_equipment_station_2')):
        model=root.find(f"world/model[@name='{name}']")
        color=list(map(float,model.findtext("link[@name='reference_payload_link']/visual/material/diffuse").split()))
        label='red' if color[0]>color[2] else 'blue'
        found[label]=dict(evidence_id=f'map_object_{i+1}',pose=list(map(float,model.findtext('pose').split()))[0:2]+[0.])
    return found


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('batch',type=Path)
    p.add_argument('--probe',action='store_true')
    p.add_argument('--ask-key',action='store_true')
    a=p.parse_args();batch=a.batch.resolve();out=batch/'analysis';out.mkdir(exist_ok=True)
    protocol=json.loads((batch/'protocol.json').read_text());metrics=[];sources={}
    for trial in protocol['trials']:
        run=batch/trial['name']
        if not (run/'episode/result.json').exists():continue
        data=rows(run/'episode/trace.jsonl');events=rows(run/'episode/selection_events.jsonl')
        result=json.loads((run/'episode/result.json').read_text());gold=object_truth(run/'world.sdf')
        crossing=next((r for r in data if r['pose'][0]>=6.5),None)
        binding=next((e for e in events if e['event']=='TASK_CONDITION_BOUND'),None)
        all_static=SpatialGraph(run/'world.sdf').static_boxes
        minimum=min(rect_gap(r['pose'],b[:3],bl=2*b[3],bw=2*b[4]) for r in data for b in all_static)
        independent=audit(run)
        r=dict(trial,arrived=result['task_success'],elapsed_sim_s=result['elapsed_sim_s'],
            navigation_collision_free=bool(result['task_success'] and minimum>0 and not independent['proxy_overlap_trace_count']),
            minimum_static_SAT_m=minimum,crossed_object_station=crossing is not None,
            bound_ids=binding['proposal']['relevant_object_ids'] if binding else [],
            ground_truth=gold)
        if trial['request']!='geometry':
            expected=gold[trial['request']]['evidence_id']
            r['grounded_target_correct']=r['bound_ids']==[expected]
            r['preferred_clearance_m']=binding['proposal']['preferred_clearance_m'] if binding else None
        if crossing:
            gaps={label:rect_gap(crossing['pose'],truth['pose']) for label,truth in gold.items()}
            r.update(crossing_pose=crossing['pose'],gap_at_crossing_m=gaps)
            if trial['request']!='geometry':
                other='blue' if trial['request']=='red' else 'red'
                r['target_minus_other_gap_m']=gaps[trial['request']]-gaps[other]
        metrics.append(r)
        for f in (run/'world.sdf',run/'episode/trace.jsonl',run/'episode/selection_events.jsonl',run/'episode/result.json'):
            sources[str(f.relative_to(ROOT))]=sha(f)
    summary=dict(kind='Object identity pilot; one sampling seed and two appearance assignments',
        geometry_equivalence_verified=protocol['geometry_equivalence_verified'],episodes=len(metrics),
        language_episodes=sum(r['request']!='geometry' for r in metrics),metrics=metrics,
        definitions={'grounded_target_correct':'First accepted condition references exactly the requested appearance-grounded object',
                     'navigation_collision_free':'Arrival with no sampled registered-static or actor-proxy overlap',
                     'target_minus_other_gap_m':'At first measured crossing x>=6.5, requested-object gap minus other-object gap; positive denotes selective clearance'},
        source_sha256=sources)
    (out/'execution_metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    for r in metrics:print(r['name'],r.get('grounded_target_correct'),r['navigation_collision_free'],r.get('target_minus_other_gap_m'))
    if not a.probe:return
    if a.ask_key:os.environ['DEEPSEEK_API_KEY']=getpass.getpass('API credential (not saved): ')
    canonical=batch/'object_reference_a_red_s41/episode/proposal_001_input.json'
    packet=json.loads(canonical.read_text())
    requests=[]
    for scene in ('object_reference_a','object_reference_b'):
        image=batch/f'{scene}_red_s41/episode/proposal_001_input.jpg'
        gold=object_truth(batch/f'{scene}.sdf')
        for referent in ('red','blue'):
            context=copy.deepcopy(packet['context']);context['instruction']=protocol['phrases'][referent]
            for present in (True,False):
                requests.append((scene,referent,present,image,context,gold))
    probe_dir=out/'same_observation_probes';probe_dir.mkdir(exist_ok=True)
    def execute(item):
        scene,referent,present,image,context,gold=item
        name=f'{scene}_{referent}_'+('image' if present else 'no_image')
        path=probe_dir/f'{name}.json'
        if path.exists():return json.loads(path.read_text())
        response=propose_task_condition(image.read_bytes(),context,packet['candidates'],include_image=present)
        request=dict(context=context,candidates=packet['candidates'])
        record=dict(name=name,scene=scene,referent=referent,image_present=present,
            context_sha256=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest(),
            image_path=str(image.relative_to(ROOT)) if present else None,
            image_sha256=sha(image) if present else None,
            expected_id=gold[referent]['evidence_id'],response=response,
            correct=response['proposal']['relevant_object_ids']==[gold[referent]['evidence_id']])
        path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
        print(name,'correct',record['correct'],response['proposal']['relevant_object_ids'],flush=True)
        return record
    with ThreadPoolExecutor(max_workers=2) as pool:probes=list(pool.map(execute,requests))
    for referent in ('red','blue'):
        assert len({r['context_sha256'] for r in probes if r['referent']==referent})==1
    (out/'grounding_probes.json').write_text(json.dumps(dict(
        kind='Eight isolated same-context visual/without-image API queries, not eight navigation episodes',
        exact_nonimage_context_shared_across_appearances=True,
        canonical_context=str(canonical.relative_to(ROOT)),canonical_context_sha256=sha(canonical),
        full_image_correct=sum(r['correct'] for r in probes if r['image_present']),
        without_image_correct=sum(r['correct'] for r in probes if not r['image_present']),
        per_modality=4,records=probes),ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
