"""Export measured manuscript data for external plotting without changing records."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
BASE = ROOT / 'artifacts/hybrid_sequence_20261022'
sources = {}
export_checks = {}


def read(p):
    return json.loads(p.read_text())


def track(p):
    sources[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()


def dump(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def copy(src, dst):
    track(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def csv_out(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                        for k, v in r.items()} for r in rows)


def trace_export(run, folder):
    src = run / 'episode/trace.jsonl'
    track(src)
    states, plans, tracks = [], [], []
    with src.open() as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            a = r.get('actuation') or {}
            ex = a.get('executed') or {}
            pose = r['pose']
            wheels = a.get('wheel_steering_rad') or []
            state = dict(sample_index=i, sim_time_s=r['timestamp'], elapsed_s=r.get('elapsed'),
                         x_m=pose[0], y_m=pose[1], yaw_rad=pose[2],
                         mode=a.get('mode'), steering_alignment_stop=a.get('steering_transition_stop'),
                         executed_vx_m_s=ex.get('vx'), executed_vy_m_s=ex.get('vy'), executed_wz_rad_s=ex.get('wz'),
                         task_stage=r.get('task_stage'), anchor_id=r.get('anchor_id'))
            # Preserve original wheel array order; do not guess hardware wheel IDs.
            state.update({f'wheel_{j}_rad': v for j, v in enumerate(wheels)})
            states.append(state)
            plan = r.get('plan') or {}
            plans.append(dict(sample_index=i, sim_time_s=r['timestamp'],
                              selected_behavior=plan.get('selected_behavior'), command_mode=plan.get('command_mode'),
                              predicted_score=plan.get('predicted_score'), planning_time_ms=plan.get('planning_time_ms'),
                              candidate_count=plan.get('candidate_count'), rejected_candidates=plan.get('rejected_candidates'),
                              cost_terms=plan.get('cost_terms'), selected_schedule=plan.get('selected_schedule'),
                              motion_condition=r.get('motion_condition')))
            for obj in r.get('tracks', []):
                tracks.append(dict(sample_index=i, sim_time_s=r['timestamp'], **obj))
    csv_out(folder / 'executed_states.csv', states)
    restored=list(csv.DictReader((folder/'executed_states.csv').open()))
    assert len(restored)==len(states)
    for a,b in zip(restored,states):
        assert all(float(a[k])==b[k] for k in ('sim_time_s','x_m','y_m','yaw_rad'))
    export_checks[str(folder.relative_to(OUT))]=len(states)
    csv_out(folder / 'planner_diagnostics.csv', plans)
    if tracks:
        csv_out(folder / 'observed_tracks.csv', tracks)
    event = run / 'episode/selection_events.jsonl'
    if event.exists():
        track(event)
        csv_out(folder / 'events.csv', [json.loads(l) for l in event.open() if l.strip()])
    clearance = run / 'clearance_trace.json'
    if clearance.exists():
        track(clearance)
        csv_out(folder / 'clearance.csv', read(clearance))


def main():
    OUT.mkdir(exist_ok=True)
    batch = BASE / 'offset_portal_domain_isolated_matrix_v6'
    data = read(batch / 'analysis/metrics.json')
    track(batch / 'analysis/metrics.json')
    target = OUT / '01_planner_comparison'
    csv_out(target / 'episode_metrics.csv', data['metrics'])
    csv_out(target / 'summary.csv', data['summary'])
    dump(target / 'metric_definitions.json', data['definitions'])
    copy(batch / 'protocol.json', target / 'protocol.json')
    for r in data['metrics']:
        name = f"offset_pose_{r['method']}_s{r['seed']}"
        trace_export(batch / name, target / name)
    copy(batch / 'offset_pose_hybrid_s37/world.sdf', target / 'world.sdf')

    pair = BASE / 'task_condition_pair_v6'
    target = OUT / '02_visual_condition'
    data = read(pair / 'analysis/metrics.json')
    track(pair / 'analysis/metrics.json')
    csv_out(target / 'episode_metrics.csv', data['metrics'])
    copy(pair / 'protocol.json', target / 'protocol.json')
    for name in ('no_condition', 'vlm'):
        trace_export(pair / name, target / name)
    # Include real visual inputs and model responses, never provider credentials.
    for p in sorted((pair / 'vlm/episode').glob('proposal_*')):
        if p.suffix in ('.json', '.jpg') and any(p.name.endswith(s) for s in
                ('_input.json', '_response.json', '_input.jpg', '_raw_rgb.jpg')):
            copy(p, target / 'model_records' / p.name)
    copy(pair / 'vlm/world.sdf', target / 'world.sdf')
    first = read(pair / 'vlm/episode/proposal_002_input.json')
    dump(target / 'semantic_space.json', first['context']['space'])

    service = BASE / 'service_static_native_v3'
    target = OUT / '03_multistop_service'
    trace_export(service, target)
    for name in ('long_task_summary.json', 'long_task_plan.json', 'independent_audit.json',
                 'language_plan_provenance.json', 'world.sdf'):
        copy(service / name, target / name)
    provenance = read(ROOT / 'paper/figures/service_execution.provenance.json')
    for name in provenance['source_sha256']:
        p = ROOT / name
        if p.suffix == '.jpg':
            copy(p, target / 'rgb' / p.name)

    target = OUT / '04_predicted_mechanism'
    provenance_path = ROOT / 'paper/figures/task_grounded_mppi_mechanism.provenance.json'
    provenance = read(provenance_path)
    src = ROOT / provenance['source']
    track(src)
    with gzip.open(src, 'rt') as f:
        record = next(json.loads(line) for i, line in enumerate(f) if i == provenance['observation_index'])
    dump(target / 'recorded_optimizer_return.json', record)
    candidates = record['plan']['hybrid_schedule_candidates']
    csv_out(target / 'schedule_candidates.csv', candidates)
    curves = []
    for i, candidate in enumerate(record['plan']['candidate_population'] + candidates):
        for j, pose in enumerate(candidate.get('trajectory') or []):
            curves.append(dict(candidate_index=i, candidate_id=candidate.get('candidate_id'),
                               structure=candidate.get('structure'), cost=candidate.get('cost'),
                               prediction_step=j, x_m=pose[0], y_m=pose[1], yaw_rad=pose[2]))
    csv_out(target / 'predicted_candidate_trajectories.csv', curves)
    csv_out(target / 'selected_prediction.csv', [dict(prediction_step=i, x_m=p[0], y_m=p[1], yaw_rad=p[2])
            for i, p in enumerate(record['plan']['selected_trajectory'])])
    copy(provenance_path, target / 'provenance.json')

    # Keep the earlier public-controller study separate from the hybrid search.
    benchmark=ROOT/'artifacts/nav2_matched_cases_20261021/path_window_fix_v4_baseline'
    source=benchmark/'analysis/aggregate.json'
    context=read(source);track(source)
    target=OUT/'05_navigation_contexts_earlier_controller'
    csv_out(target/'episode_metrics.csv',context['all_attempts'])
    csv_out(target/'summary.csv',context['rows'])
    for r in context['all_attempts']:
        parent=benchmark if r['method']=='nav2-omni' else benchmark.parent/'frozen_routefix_v3'
        name=f"{r['scene']}_{r['method']}_s{r['seed']}"
        trace_export(parent/name,target/name)
    dump(target/'scope.json',dict(episodes=27,
        controller='Frozen family-based controller without within-horizon hybrid sequence search',
        public_method='Nav2 Omni, corrected path-window adapter; lateral sampling retained',
        separation='Do not pool with the 15 hybrid timing trials or label this controller as the latest hybrid optimizer',
        attempts='All attempts retained, including non-arrivals and negative sampled gap'))

    benchmark=ROOT/'artifacts/task_condition_benchmark_20261023/task_matrix_v8'
    if (benchmark/'analysis/metrics.json').exists():
        metrics=read(benchmark/'analysis/metrics.json')
        assert metrics['completed_episodes']==36
        group=OUT/'06_task_condition_benchmark'
        for name in ('protocol.json','analysis/metrics.json','analysis/episode_metrics.csv'):
            copy(benchmark/name,group/Path(name).name)
        for trial in read(benchmark/'protocol.json')['trials']:
            run=benchmark/trial['name']
            trace_export(run,group/trial['name'])
        for src in (ROOT/'paper/figures/task_condition_20261023').glob('*'):
            if src.suffix in ('.pdf','.svg','.json','.tex'):
                copy(src,group/'figures'/src.name)
    mission=ROOT/'artifacts/task_guided_online_20261023/mission_v6_cpp_s41/run'
    if (mission/'integrated_audit.json').exists():
        group=OUT/'07_integrated_online_service'
        trace_export(mission,group)
        for name in ('integrated_audit.json','long_task_plan.json','long_task_summary.json',
                     'task_parser_response.json','world.sdf','scene_manifest.json'):
            copy(mission/name,group/name)
        for src in (ROOT/'post/task_guided_online_20261023/mission_v6_cpp_s41').glob('online_service_*'):
            if src.suffix not in ('.pdf','.svg','.json'):continue
            copy(src,group/src.name)
        for name in ('proposal_002_input.jpg','proposal_002_response.json'):
            copy(mission/'episode'/name,group/name)

    pilot=ROOT/'artifacts/object_reference_20261026/pilot_v1'
    if (pilot/'analysis/grounding_probes.json').exists():
        group=OUT/'08_object_reference_pilot'
        for name in ('protocol.json','analysis/execution_metrics.json','analysis/grounding_probes.json'):
            copy(pilot/name,group/Path(name).name)
        for trial in read(pilot/'protocol.json')['trials']:
            trace_export(pilot/trial['name'],group/trial['name'])
        for scene in ('object_reference_a','object_reference_b'):
            for name in ('proposal_001_raw_rgb.jpg','proposal_001_input.jpg','proposal_001_input.json','proposal_001_response.json'):
                copy(pilot/f'{scene}_red_s41/episode'/name,group/scene/name)
        for src in (ROOT/'paper/figures/object_reference_20261026').glob('*'):
            if src.suffix in ('.pdf','.svg','.json'):
                copy(src,group/'figures'/src.name)

    target=OUT/'figure_exports'
    for stem in ('visual_candidate_input','task_grounded_overview','task_grounded_mppi_mechanism_compact','task_grounded_mppi_mechanism_motion','service_navigation','executed_heading_steering','service_transfer_maneuvers','service_mppi_choice','service_execution_state','local_planner_response','service_transfer_compact','paired_planner_effects'):
        for ext in ('pdf','svg','provenance.json','collision-audit.json'):
            copy(ROOT/f'paper/figures/{stem}.{ext}',target/f'{stem}.{ext}')
    for ext in ('pdf','svg','provenance.json'):
        copy(ROOT/f'paper/figures/task_grounded_mppi_mechanism.{ext}',
             target/f'task_grounded_mppi_mechanism.{ext}')
    for stem in ('transition_execution_evidence','transition_paired_outcomes',
                 'navigation_operating_contexts','visual_execution_trace'):
        for ext in ('pdf','svg','provenance.json'):
            old=target/f'{stem}.{ext}'
            if old.exists():old.unlink()  # Regenerated figure copies; originals remain in paper/figures.
    for stem in ('Fig02_Matched_offset_doorway_s37','Fig03_Paired_arrival_times',
                 'Fig04_Executed_mode_timeline_s37','Fig05_Online_VLM_RGB_and_paths',
                 'Fig06_Service_mission_spatial_trace'):
        for ext in ('pdf','svg','provenance.json'):
            copy(ROOT/f'paper/figures/user_revision/final/{stem}.{ext}',target/f'{stem}.{ext}')
    copy(ROOT/'paper/figures/hybrid_transition_outcomes.tex',target/'table_I.tex')
    copy(ROOT/'paper/figures/hybrid_transition_outcomes.provenance.json',target/'table_I.provenance.json')

    text = '''# ChatGPT 论文绘图数据包

本目录仅整理已有实测与预测记录；不新增实验，不插值、不平滑、不修改指标。

## 内容与用途

1. `01_planner_comparison/`：同一偏置门口位姿任务，primitive / fixed / hybrid，各五种子，共15回合。episode_metrics.csv 是逐回合指标，summary.csv 是原统计结果；每回合 executed_states.csv 可画配对 X–Y 轨迹、轮角、速度和模式时间线。这三个配置是内部机制对照，不是三个公开算法。公共直接终点参考，无专为绕障绘制的路线。
2. `02_visual_condition/`：同一种子两回合，真实 VLM 输入、返回、轨迹和完整指标。用于展示指令—视觉条件—运动执行链，不据此宣称 VLM 总体提高性能。保留全部指标，包括航向误差；不只挑选有利指标。
3. `03_multistop_service/`：617取资料→614交付并取回执→619交付。四项服务确认来自仿真；语言计划来自已保存的真实模型解析，本回合未重新请求视觉模型。可画整程模式着色轨迹、事件时间线和关键 RGB。
4. `04_predicted_mechanism/`：主批v6种子37第2次实际优化器返回的候选与选中结果，对应当前Fig.1；是模型预测，不是执行轨迹或新增独立回合。
5. `05_navigation_contexts_earlier_controller/`：既有普通绕障、朝向约束通行、偏置门口27回合公开Nav2／运动族控制器／消融对照，保留未到站回合。这里的运动族控制器未启用新版混合序列搜索，不得混入主实验或冒充最新版算法排名。
6. `figure_exports/`：当前服务路线与局部停靠图、执行航向与轮角曲线、方法流程及预测机理图。旧表与旧图作为历史资产保留。配对图仅连接同一种子；VLM图中车体与推车分别取各自回合的最近交互时刻。paired_transition_decomposition.csv 是完成时间差分解，不是新增实验。research_evidence_design.json 是待执行设计，不是实验结果。

## 字段与单位

- x_m / y_m 是实际记录的世界坐标，yaw_rad 为记录航向；所有 *_rad 是弧度，*_s 是秒，*_m 是米。
- sim_time_s 为仿真时钟，elapsed_s 为原日志经过时间；不同回合分别起计，不能按行号强行对齐。
- wheel_0_rad 至 wheel_3_rad 保留日志原数组顺序；为运动学模型轮角，不是实机舵机测量值。PARALLEL 包括斜向平移和纯横移。
- executed_* 是执行层记录的本体速度；planner_diagnostics 是预测诊断，两者不能当成同一测量。
- predicted_score / cost_terms 只能在共同目标和代价定义内比较，不能跨批次直接排名。
- sampled_alignment_s 是采样记录估计的累计停车对轮时间；nonzero_mode_changes 是非零执行运动模式变化，非候选族变化。
- signed SAT gap 为矩形代理/登记几何的有符号间距，正值分离、负值重叠；不得称完整mesh或实机安全距离。逐指标以 metric_definitions.json 和原协议为准。
- 空单元格表示该记录无此字段，不是零。CSV 中数组/对象为 JSON 字符串。events.csv 可按 event / sim_time 筛选，保留所有任务事件。

## 直接给 ChatGPT 的绘图要求

请从逐回合CSV绘制IEEE双栏风格矢量图：白底、深灰环境、蓝绿色本方法、橙色对照，统一字号线宽。先画同种子X–Y执行轨迹及局部放大，再画轮角/模式/间距对应曲线；五种子的指标展示全部散点及汇总。环境几何读取 world.sdf，注意SDF局部坐标变换，不能编造墙和障碍位置。VLM只展示真实输入、记录条件和执行关联。预测候选必须标为predicted；不同冻结版本不混算。不要编造缺失轨迹、候选代价或图片，也不要把内部消融称为公开基线。输出PDF/SVG及绘图脚本，图上仅保留必要文字。

source_manifest.json 提供来源与SHA256；原始大文件仍留在工作区，不随包携带。重建：python3 tools/export_plot_results.py。
'''
    text += '\n最新补充：`06_task_condition_benchmark/` 为36回合冻结任务条件对照，包含所有到站、重叠和超时回合；`07_integrated_online_service/` 为新语言解析、真实在线视觉和混合MPPI同一次完整任务。历史目录仍独立保留，不跨版本混算。当前主稿以服务路线、任务轨迹和执行状态组织结果，不包含排行式汇总表。06目录保留完整36回合统计，旧figure_exports/table_I为历史单门口机制表。\n'
    text += '\n`08_object_reference_pilot/`：六次新闭环导航（4次对象任务、2次几何参考）及8次独立同观测API查询；查询不是导航回合。颜色交换保持非材质几何完全一致。有图4/4、去图1/4仅用于本定向试验，不代表通用接地准确率。对象正确性、选择性空间响应和到站分开报告，原36回合不混算。\n'
    (OUT / 'README.md').write_text(text)
    dump(OUT / 'source_manifest.json', dict(source_sha256=sources, data_policy='original samples; no smoothing, imputation or metric changes'))
    dump(OUT/'export_validation.json',dict(source_hashes_recorded=len(sources),
        episodes_verified=len(export_checks),sample_rows_verified=sum(export_checks.values()),
        exact_pose_and_timestamp_match=True,rows_by_episode=export_checks))
    # Scan exported text for common credential forms before packaging.
    import re
    for p in OUT.rglob('*'):
        if p.is_file() and p.suffix in ('.json', '.csv', '.md'):
            assert not re.search(r'sk-[A-Za-z0-9]{20,}|Bearer\s+[A-Za-z0-9._-]{20,}', p.read_text()), f'credential-like text: {p}'
    archive = ROOT / 'results_chatgpt.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(ROOT))
    print(f'Exported {len(sources)} source records; archive {archive.stat().st_size / 1024**2:.2f} MiB')


if __name__ == '__main__':
    main()
