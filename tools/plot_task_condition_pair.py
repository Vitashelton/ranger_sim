"""Measured task-condition evidence: RGB, local trajectories, heading and video.

Claim: a real VLM motion preference enters MPPI and changes executed motion.
One matched seed is a mechanism case, not a navigation superiority estimate.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/rangermini_dynamic_semantic/scripts'))
from audit_grounded_delivery import audit
sys.path.insert(0, '/home/USER/.codex/skills/nature-figure/scripts')
from audit_panel_alignment import require_matplotlib_panel_alignment


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pair', type=Path)
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()
    pair = args.pair.resolve()
    output = pair/'analysis'
    output.mkdir(exist_ok=True)
    protocol = json.loads((pair/'protocol.json').read_text())
    records, metrics, sources = {}, [], [pair/'protocol.json', Path(__file__)]
    for method in ('no_condition', 'vlm'):
        run = pair/method
        result = audit(run)
        rows = read_rows(run/'episode/trace.jsonl')
        records[method] = rows
        times = np.asarray([r['timestamp'] for r in rows])
        dt = np.diff(times, append=times[-1])
        poses = np.asarray([r['pose'] for r in rows])
        yaw = np.arctan2(np.sin(poses[:, 2]), np.cos(poses[:, 2]))
        moving, parallel, modes, alignment = [], [], [], []
        for row in rows:
            act = row.get('actuation') or {}
            u = act.get('executed') or {}
            active = math.hypot(u.get('vx', 0), u.get('vy', 0))+abs(u.get('wz', 0)) > .001
            moving.append(active)
            parallel.append(active and act.get('mode') == 'PARALLEL')
            alignment.append(bool(act.get('steering_transition_stop')))
            if active:
                modes.append(act.get('mode'))
        moving_time = float(dt@np.asarray(moving))
        active_dt = dt*np.asarray(moving)
        metric = dict(method=method, status=result['status'], seed=protocol['seed'],
            qualified=result['interaction_success'], path_m=result['path_length_m'],
            elapsed_s=result['elapsed_sim_s'],
            max_heading_error_deg=float(np.degrees(np.max(np.abs(yaw)))),
            moving_heading_rms_deg=float(np.degrees(np.sqrt(active_dt@(yaw*yaw)/max(moving_time, 1e-9)))),
            parallel_moving_time_fraction=float(dt@np.asarray(parallel)/max(moving_time, 1e-9)),
            nonzero_mode_switches=sum(a != b for a, b in zip(modes, modes[1:])),
            sampled_alignment_s=float(dt@np.asarray(alignment)),
            trolley_gap_m=result['trolley_signed_SAT_min_gap_m'],
            overlap_samples=result['proxy_overlap_trace_count'],
            semantic_queries=result['semantic_queries'], api_requests=result.get('vlm_api_requests', 0),
            motion_gate_overrides=result['motion_gate_overrides'])
        metrics.append(metric)
        sources += [run/'episode/result.json', run/'episode/trace.jsonl', run/'scene_manifest.json',
                    run/'fixture_truth.jsonl']
    # Both runs must freeze identical planner/adapter sources and world geometry.
    manifests = [json.loads((pair/m/'scene_manifest.json').read_text()) for m in ('no_condition', 'vlm')]
    assert manifests[0]['source_sha256'] == manifests[1]['source_sha256']
    assert (pair/'no_condition/world.sdf').read_bytes() == (pair/'vlm/world.sdf').read_bytes()
    for metric in metrics:
        assert metric['motion_gate_overrides'] == 0, 'Legacy behavior gate is not a matched task-condition path'
    with (output/'metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader(); writer.writerows(metrics)
    report = dict(kind='one-seed matched Gazebo mechanism case', metrics=metrics,
        n_per_method=1, inferential_statistics=False,
        all_attempts_qualified=all(m['qualified'] for m in metrics),
        common_source_verified=True, common_world_verified=True,
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (output/'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['DejaVu Sans'], 'font.size':7,
        'pdf.fonttype':42, 'svg.fonttype':'none', 'axes.spines.top':False,
        'axes.spines.right':False, 'axes.linewidth':.7, 'legend.frameon':False})
    fig = plt.figure(figsize=(7.16, 2.35))
    axes = [fig.add_axes(rect) for rect in ([.015,.24,.28,.64],[.385,.21,.25,.67],[.735,.21,.25,.67])]
    # Original RGB crop only; no contrast, replacement or synthetic trajectories.
    response_paths = sorted((pair/'vlm/episode').glob('proposal_*_response.json'))
    selected = next((p for p in response_paths if json.loads(p.read_text())['proposal']['relevant_object_ids']), response_paths[0])
    image_path = selected.with_name(selected.name.replace('_response.json', '_input.jpg'))
    raw_path = selected.with_name(selected.name.replace('_response.json', '_raw_rgb.jpg'))
    raw = plt.imread(raw_path)
    annotated = plt.imread(image_path)
    h, w = raw.shape[:2]
    crop = annotated[38:38+h, :w]
    axes[0].imshow(crop, aspect='auto'); axes[0].set_axis_off()
    axes[0].set_title('(a) Actual VLM observation', loc='left', fontsize=8, pad=7)
    colors = {'no_condition':'#ce7c35', 'vlm':'#168c89'}
    labels = {'no_condition':'Baseline', 'vlm':'Ours'}
    for method, rows in records.items():
        poses = np.asarray([r['pose'] for r in rows])
        local = (poses[:,0]>=4)&(poses[:,0]<=8.5)
        axes[1].plot(poses[local,0], poses[local,1], color=colors[method], lw=1.4, label=labels[method])
        clearance = json.loads((pair/method/'clearance_trace.json').read_text())
        k = min(range(len(clearance)), key=lambda i:clearance[i]['trolley_signed_SAT_gap_m'])
        pose = poses[k]
        corners = np.array([[-.3755,-.25],[.3755,-.25],[.3755,.25],[-.3755,.25]])
        c, s = math.cos(pose[2]), math.sin(pose[2])
        axes[1].add_patch(Polygon(corners@np.array([[c,s],[-s,c]])+pose[:2],
                                 closed=True, fill=False, edgecolor=colors[method], lw=.9))
        if method == 'vlm':
            cart = np.asarray([r['trolley_pose'] for r in clearance])
            local = (cart[:,0]>=4)&(cart[:,0]<=8.5)
            axes[1].plot(cart[local,0],cart[local,1],color='#74787b',ls='--',lw=.9)
        elapsed = np.asarray([r['elapsed'] for r in rows])
        axes[2].plot(elapsed, np.degrees(np.arctan2(np.sin(poses[:,2]),np.cos(poses[:,2]))),
                     color=colors[method],lw=1.2,label=labels[method])
    axes[1].set(xlim=(4,8.5),ylim=(-3.6,-.6),xlabel='$x$ (m)',ylabel='$y$ (m)')
    axes[1].set_title('(b) Measured interaction paths',loc='left',fontsize=8,pad=7)
    axes[2].axhline(0,color='#888888',lw=.5,ls=':')
    axes[2].set(xlabel='Mission time (s)',ylabel='Heading error (deg)')
    axes[2].set_title('(c) Executed heading',loc='left',fontsize=8,pad=7)
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.70,-.015),ncol=2,fontsize=7)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, axes=axes, panel_ids=['a','b','c'],
        row_groups=[['b','c']], json_out=output/'task_condition_execution.alignment.json')
    fig.savefig(output/'task_condition_execution.pdf',facecolor='white')
    fig.savefig(output/'task_condition_execution.svg',facecolor='white')
    fig.savefig(output/'task_condition_execution.png',dpi=600,facecolor='white')
    plt.close(fig)
    provenance = dict(source_sha256=report['source_sha256'],
        image_source=str(image_path.relative_to(ROOT)),image_crop=[0,38,w,38+h],
        image_adjustment='crop only; RGB appearance unchanged; overlay is the actual VLM input',
        actual_proposal=json.loads(selected.read_text())['proposal'],
        candidate_and_model_records=[str(p.relative_to(ROOT)) for p in response_paths],
        full_trace_points_used=True, xy_panel_region=dict(x=[4,8.5],y=[-3.6,-.6]),
        xy_exclusion='display crop to the interaction region; metrics and heading use the complete episode',
        image_panel_resized_to_axes=True,
        claim='A scene-grounded motion preference conditions MPPI and changes executed motion.',
        case_scope='one seed; safety/arrival qualification reported independently in metrics.json')
    (output/'task_condition_execution.provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    if args.video:
        make_video(pair, output, records)
    print(json.dumps(dict(metrics=metrics,qualified=report['all_attempts_qualified']),ensure_ascii=False))


def make_video(pair, output, records):
    frame_logs = {m:read_rows(pair/m/'episode/overview_frames.jsonl') for m in records}
    origin = {m:records[m][0]['timestamp']-records[m][0]['elapsed'] for m in records}
    end = max(r[-1]['elapsed'] for r in records.values())
    events = read_rows(pair/'vlm/episode/selection_events.jsonl')
    conditions = [r for r in events if r['event'] == 'TASK_CONDITION_BOUND']
    temporary = output/'comparison_raw.mp4'
    writer = cv2.VideoWriter(str(temporary),cv2.VideoWriter_fourcc(*'mp4v'),15,(1280,480))
    for elapsed in np.arange(0,end,2/15):
        panels=[]
        for method in ('no_condition','vlm'):
            stamp = origin[method]+elapsed
            log = min(frame_logs[method],key=lambda r:abs(r['sim_time']-stamp))
            frame = cv2.imread(str(pair/method/'episode/overview_frames'/f"{log['frame']:05d}.jpg"))
            frame = cv2.resize(frame,(640,480),interpolation=cv2.INTER_AREA)
            row=min(records[method],key=lambda r:abs(r['timestamp']-stamp))
            act=row.get('actuation') or {};mode=act.get('mode','STOPPED')
            overlay=frame.copy();cv2.rectangle(overlay,(0,0),(640,83),(15,22,28),-1)
            frame=cv2.addWeighted(overlay,.72,frame,.28,0)
            text=('Baseline' if method=='no_condition' else 'Ours')+' | MPPI: '+mode
            cv2.putText(frame,text,(12,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(250,250,250),1,cv2.LINE_AA)
            if method=='vlm':
                live=[e for e in conditions if e['sim_time']<=stamp]
                p=live[-1]['proposal'] if live else {}
                text='VLM: '+('trolley | ' if 'trolley' in p.get('object_labels',{}).values() else '')+p.get('motion_preference','pending')
            else:text='Same goal, route, sampler and wheel execution'
            cv2.putText(frame,text,(12,48),cv2.FONT_HERSHEY_SIMPLEX,.43,(235,235,235),1,cv2.LINE_AA)
            cv2.putText(frame,'2x playback',(12,71),cv2.FONT_HERSHEY_SIMPLEX,.40,(210,210,210),1,cv2.LINE_AA)
            panels.append(frame)
        writer.write(np.concatenate(panels,axis=1))
    writer.release()
    import subprocess
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(temporary),'-c:v','libx264',
                    '-pix_fmt','yuv420p','-movflags','+faststart',str(output/'comparison.mp4')],check=True)
    temporary.unlink()


if __name__ == '__main__':
    main()
