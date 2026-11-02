"""Export completed multiroom trials; room segments remain repeated measures."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def summarize(batch):
    protocol = json.loads((batch / 'protocol.json').read_text())
    missions, segments, sources = [], [], {}
    for trial in protocol['trials']:
        run = batch / trial['name']
        result_path = run / 'episode/result.json'
        if not result_path.exists():
            continue
        trace_path = run / 'episode/trace.jsonl'
        event_path = run / 'episode/selection_events.jsonl'
        result = json.loads(result_path.read_text())
        rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
        events = [json.loads(line) for line in event_path.read_text().splitlines()]
        for p in (result_path, trace_path, event_path):
            sources[str(p.relative_to(batch))] = hashlib.sha256(p.read_bytes()).hexdigest()
        active_mode = None
        switches = 0
        hold_s = 0.
        solver_errors = 0
        parts = []
        for i, row in enumerate(rows):
            dt = (rows[i + 1]['elapsed'] - row['elapsed']) if i + 1 < len(rows) else 0.
            if dt < 0:
                raise ValueError('Regressed simulation timestamps')
            act = row.get('actuation', {})
            vel = act.get('executed', {})
            moving = math.hypot(vel.get('vx', 0.), vel.get('vy', 0.)) > .02 or abs(vel.get('wz', 0.)) > .02
            mode = act.get('mode', 'UNKNOWN')
            change = int(moving and active_mode is not None and mode != active_mode)
            if moving:
                active_mode = mode
            switches += change
            hold = dt if act.get('steering_transition_stop', False) else 0.
            hold_s += hold
            error = int(bool(row.get('plan', {}).get('public_solver_error')))
            solver_errors += error
            key = (row.get('anchor_id'), row.get('task_stage'))
            if not parts or parts[-1]['key'] != key:
                parts.append(dict(key=key, start_s=row['elapsed'], end_s=row['elapsed'],
                                  path_m=0., alignment_hold_s=0., moving_mode_changes=0,
                                  public_error_records=0))
            part = parts[-1]
            part['end_s'] = row['elapsed'] + dt
            part['alignment_hold_s'] += hold
            part['moving_mode_changes'] += change
            part['public_error_records'] += error
            if i + 1 < len(rows):
                part['path_m'] += math.dist(row['pose'][:2], rows[i + 1]['pose'][:2])
        missions.append(dict(method=trial['method'], seed=trial['seed'], status=result['status'],
            arrived=result['task_success'], elapsed_sim_s=result['elapsed_sim_s'],
            path_m=result['path_length_m'], moving_mode_changes=switches,
            sampled_alignment_hold_s=hold_s, public_error_records=solver_errors,
            simulated_service_confirmations=sum(e.get('event') == 'SERVICE_ACTION_CONFIRMED' for e in events)))
        for index, part in enumerate(parts):
            anchor, stage = part.pop('key')
            segments.append(dict(method=trial['method'], seed=trial['seed'],
                                 segment=index, anchor=anchor, stage=stage, **part))
    out = batch / 'analysis'
    out.mkdir(exist_ok=True)
    for name, data in [('missions', missions), ('segments', segments)]:
        if not data:
            continue
        with (out / (name + '.csv')).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (out / 'provenance.json').write_text(json.dumps(dict(
        independent_unit=protocol['independent_unit'], completed_missions=len(missions),
        planned_missions=protocol['intended_missions'],
        scope='Segments are repeated measures, not additional independent trials. Public-controller qualification pending; no superiority claim.',
        alignment_definition='Left-endpoint trace integration of measured adapter alignment-stop flag; not actuator measurement.',
        moving_mode_definition='Changes between successive native modes with body speed or yaw rate >0.02; STOP excluded.',
        sources=sources), indent=2) + '\n')
    print(json.dumps(missions, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    summarize(parser.parse_args().batch)
