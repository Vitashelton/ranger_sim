#!/usr/bin/env python3
"""Summarize actual callback overlap; never infer DDS arrival from callback time."""
import argparse
import bisect
import csv
import json
from pathlib import Path
import statistics


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def audit(root):
    rows=read(root/'episode/runtime_timeline.jsonl')
    independent=read(root/'independent_runtime_timeline.jsonl')
    windows=[]
    pending={}
    for row in rows:
        event=row['event']
        if event in ('planner_call_begin','planner_rpc_begin'):
            pending[(event,row['method'])]=row
        elif event in ('planner_call_end','planner_rpc_end'):
            start=pending.pop((event.replace('_end','_begin'),row['method']),None)
            if start and row['method']=='bank':
                windows.append((start['wall_monotonic_s'],row['wall_monotonic_s']))
    # Old legacy captures have callback boundaries, but no planner-call spans.
    if not windows:
        starts=[r for r in rows if r['event']=='planning_callback_begin']
        ends=[r for r in rows if r['event']=='planning_callback_end']
        windows=[(a['wall_monotonic_s'],b['wall_monotonic_s']) for a,b in zip(starts,ends)]
        span_kind='planning_callback_including_route_and_full_bank'
    else:
        # Prefer call spans (RPC spans duplicate the same computation).
        call_ends=[r for r in rows if r['event']=='planner_call_end' and r['method']=='bank']
        if call_ends:
            windows=[(r['wall_monotonic_s']-r['duration_wall_s'],r['wall_monotonic_s']) for r in call_ends]
        span_kind='full_bank_call'
    near427=min(windows,key=lambda p:abs((p[1]-p[0])-.427))
    near_rows=[dict(source='controller',**r) for r in rows if near427[0]<=r['wall_monotonic_s']<=near427[1]]
    near_rows += [dict(source='independent_observer',**r) for r in independent if near427[0]<=r['wall_monotonic_s']<=near427[1]]
    near_rows.sort(key=lambda r:r['wall_monotonic_s'])
    with (root/'near_427ms_timeline.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['relative_wall_ms','source','event','topic','acquisition_ros_s','state_version'])
        for r in near_rows:
            writer.writerow([round(1000*(r['wall_monotonic_s']-near427[0]),3),r['source'],r['event'],r.get('topic'),r.get('acquisition_ros_s'),r.get('state_version')])
    near_counts={source:{topic:sum(r['source']==source and r['event']==event and r.get('topic')==topic for r in near_rows)
        for topic in ['/scan','/odom','/motion_mode_debug','/clock']}
        for source,event in [('controller','sensor_callback_begin'),('independent_observer','independent_receive')]}
    longest=max(windows,key=lambda p:p[1]-p[0])
    a,b=longest
    selected=[dict(source='controller',**r) for r in rows if a<=r['wall_monotonic_s']<=b]
    selected += [dict(source='independent_observer',**r) for r in independent if a<=r['wall_monotonic_s']<=b]
    selected.sort(key=lambda r:r['wall_monotonic_s'])
    with (root/'longest_compute_timeline.jsonl').open('w') as f:
        for r in selected:f.write(json.dumps(r)+'\n')
    counts={}
    for stream,event in [('controller','sensor_callback_begin'),('independent_observer','independent_receive')]:
        counts[stream]={topic:sum(r['source']==stream and r['event']==event and r.get('topic')==topic for r in selected)
                       for topic in ['/scan','/odom','/motion_mode_debug','/clock']}
    ends=[r for r in rows if r['event']=='sensor_callback_end']
    ages=[r['ros_time_s']-r['acquisition_ros_s'] for r in ends if r['topic']=='/scan']
    observed={(r['topic'],round(r['acquisition_ros_s'],6)):r['wall_monotonic_s'] for r in independent
              if r.get('acquisition_ros_s') is not None}
    deltas=[r['wall_monotonic_s']-r['callback_wall_s']-observed[(r['topic'],round(r['acquisition_ros_s'],6))]
            for r in ends if (r['topic'],round(r['acquisition_ros_s'],6)) in observed and r['topic']=='/scan']
    clock=[r for r in independent if r.get('topic')=='/clock' and r.get('acquisition_ros_s') is not None]
    clock_times=[r['wall_monotonic_s'] for r in clock]
    clock_lags=[]
    for row in rows:
        if row['event']=='planner_call_end' and row.get('ros_time_s') is not None:
            index=bisect.bisect_right(clock_times,row['wall_monotonic_s'])-1
            if index>=0:clock_lags.append(clock[index]['acquisition_ros_s']-row['ros_time_s'])
    heartbeat=[r for r in rows if r['event']=='command_heartbeat']
    intervals=[b['wall_monotonic_s']-a['wall_monotonic_s'] for a,b in zip(heartbeat,heartbeat[1:])]
    result=dict(span_kind=span_kind, near_427ms_compute_wall_s=near427[1]-near427[0],
                near_427ms_callback_counts=near_counts, longest_compute_wall_s=b-a, callback_counts_during_compute=counts,
                scan_ros_age_max_s=max(ages,default=None),
                ros_clock_lag_at_planner_return_max_sim_s=max(clock_lags,default=None),
                scan_callback_vs_independent_receive_median_s=statistics.median(deltas) if deltas else None,
                scan_callback_vs_independent_receive_max_s=max(deltas,default=None),
                command_heartbeat_interval_max_s=max(intervals,default=None),
                observation_holds=sum(json.loads(l).get('event')=='OBSERVATION_HOLD' for l in (root/'episode/selection_events.jsonl').read_text().splitlines()) if (root/'episode/selection_events.jsonl').exists() else None,
                dds_received_timestamps_available=any(r.get('dds_received_system_ns') for r in rows),
                qualification='Short scheduling diagnostic, not a navigation success experiment. Observer receipt is not publisher or controller DDS arrival.')
    (root/'executor_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(root=str(root),**result),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('roots',nargs='+',type=Path)
    for root in p.parse_args().roots:audit(root)
