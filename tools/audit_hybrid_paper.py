"""Check published hybrid results against frozen raw records and figure sources."""
import gzip
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
BATCH=ROOT/'artifacts/hybrid_sequence_20261022/offset_portal_domain_isolated_matrix_v6'
ARCHIVE=ROOT/'artifacts/hybrid_sequence_20261022/frozen_native_v1'
FIG=ROOT/'paper/figures'

def read(path):return json.loads(path.read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    data=read(BATCH/'analysis/metrics.json');protocol=read(BATCH/'protocol.json')
    assert len(data['metrics'])==15
    assert {(r['seed'],r['method']) for r in data['metrics']}=={
        (s,m) for s in protocol['seeds'] for m in protocol['methods']}
    for source,expected in data['source_sha256'].items():assert digest(ROOT/source)==expected,source
    manifests=[]
    for r in data['metrics']:
        run=BATCH/f"offset_pose_{r['method']}_s{r['seed']}"
        manifest=read(run/'scene_manifest.json');manifests.append(manifest)
        for name,expected in manifest['source_sha256'].items():
            assert digest(run/'source_snapshot'/name)==expected,name
        assert digest(run/'source_snapshot/libranger_mppi_native.so')==manifest['numeric_backend']['library_sha256']
        assert r['max_rollout_evaluations']<=protocol['internal_rollout_cap']
        assert read(run/'episode/result.json')['goal']==protocol['goal'][:2]
    assert all(m['source_sha256']==manifests[0]['source_sha256'] for m in manifests)
    assert digest(FIG/'hybrid_local_table.tex')==digest(BATCH/'analysis/results_table.tex')
    assert protocol['seeds']==[37,41,53,59,61] and set(protocol['methods'])=={'primitive','fixed','hybrid'}
    for ext in ('pdf','svg','png'):
        assert digest(FIG/f'hybrid_local_trajectories.{ext}')==digest(BATCH/f'analysis/trajectories.{ext}')
    mechanism=read(FIG/'hybrid_sequence_mechanism.provenance.json')
    assert mechanism['actual_planner_return']
    assert digest(ROOT/mechanism['source'])==mechanism['source_sha256']
    with gzip.open(ROOT/mechanism['source'],'rt') as stream:
        record=[json.loads(line) for line in stream][mechanism['observation_index']]
    plan=record['plan'];assert plan['selected_schedule']['switch_steps']==[11]
    assert abs(plan['predicted_score']-17.734822730571434)<1e-9
    main_mechanism=read(FIG/'task_grounded_mppi_mechanism.provenance.json')
    assert digest(ROOT/main_mechanism['source'])==main_mechanism['source_sha256']
    with gzip.open(ROOT/main_mechanism['source'],'rt') as stream:
        observation=[json.loads(line) for line in stream][main_mechanism['observation_index']]
    assert observation['plan']['selected_schedule']==main_mechanism['selected_schedule']
    assert abs(observation['plan']['predicted_score']-main_mechanism['selected_cost'])<1e-9
    assert main_mechanism['selected_schedule']['structure']==['A','R']
    assert main_mechanism['selected_schedule']['switch_steps']==[30]
    assert read(FIG/'task_grounded_mppi_mechanism.collision-audit.json')['summary']['fail']==0
    service=read(FIG/'service_execution.provenance.json');run=ROOT/service['run']
    for source,expected in service['source_sha256'].items():assert digest(ROOT/source)==expected,source
    assert service['mission_completed'] and service['completed_actions']==4
    assert read(run/'scene_manifest.json')['scenario']=='multiroom_service_static'
    assert read(run/'long_task_summary.json')['task_complete']
    assert read(run/'long_task_summary.json')['language_plan_reused']
    independent=read(run/'independent_audit.json')
    assert independent['interaction_success'] and independent['proxy_overlap_trace_count']==0
    assert independent['static_overlap_pose_count']==0
    # Service-only centering/routing changes must not alter the optimizer.
    service_sources=read(run/'scene_manifest.json')['source_sha256']
    for name in ('hybrid_sequence_mppi.py','observation_mppi.py','interaction_mppi.py',
                 'ranger_4ws_kinematics.py','native_kernels.py','core.cpp'):
        assert service_sources[name]==manifests[0]['source_sha256'][name],name
    qa={}
    for name in ('hybrid_sequence_mechanism','hybrid_local_trajectories',
                 'instruction_execution_trace','multiroom_execution'):
        collision=read(FIG/f'{name}.collision-audit.json')
        assert collision['verdict']=='PASS',name
        qa[name]=collision['verdict']
    pair=ROOT/'artifacts/hybrid_sequence_20261022/task_condition_pair_v6'
    visual=read(pair/'analysis/metrics.json')
    assert visual['common_source_verified'] and visual['common_world_verified']
    assert visual['all_attempts_qualified'] and len(visual['metrics'])==2
    for source,expected in visual['source_sha256'].items():
        assert digest(ROOT/source)==expected,source
    for row in visual['metrics']:
        assert row['motion_gate_overrides']==0 and row['overlap_samples']==0
        trial=pair/row['method'];manifest=read(trial/'scene_manifest.json')
        for name,expected in manifest['source_sha256'].items():
            assert digest(trial/'source_snapshot'/name)==expected,name
    assert read(pair/'analysis/task_condition_execution.collision-audit.json')['verdict']=='PASS'
    for ext in ('pdf','svg','png','provenance.json'):
        assert digest(FIG/f'task_condition_execution.{ext}')==digest(pair/f'analysis/task_condition_execution.{ext}')
    for name in ('transition_execution_evidence','transition_paired_outcomes',
                 'navigation_operating_contexts','visual_execution_trace'):
        provenance=read(FIG/f'{name}.provenance.json')
        for source,expected in provenance['source_sha256'].items():
            assert digest(ROOT/source)==expected,source
        collision=read(FIG/f'{name}.collision-audit.json')
        assert collision['verdict']=='PASS',name
        qa[name]=collision['verdict']
    primary=read(FIG/'hybrid_transition_outcomes.provenance.json')
    assert primary['all_summary_seeds']==[37,41,53,59,61]
    for row in primary['statistics']:
        assert row['n']==5
    assert digest(FIG/'hybrid_transition_outcomes.tex')==primary['table_sha256']
    for source,expected in primary['source_sha256'].items():
        assert digest(ROOT/source)==expected,source
    revision=FIG/'user_revision/final'
    for name in ('Fig02_Matched_offset_doorway_s37','Fig03_Paired_arrival_times',
                 'Fig04_Executed_mode_timeline_s37','Fig05_Online_VLM_RGB_and_paths',
                 'Fig06_Service_mission_spatial_trace'):
        provenance=read(revision/f'{name}.provenance.json')
        for source,expected in provenance['source_sha256'].items():
            assert digest(ROOT/source)==expected,source
        assert read(revision/f'{name}.collision-audit.json')['verdict']=='PASS',name
        qa[name]='PASS'
    visual_pairs=read(revision/'Fig05_Online_VLM_RGB_and_paths.provenance.json')['interaction_pairs']
    assert len(visual_pairs)==2
    assert all(p['gap_m']>0 for p in visual_pairs)
    summary={r['method']:r for r in data['summary']}
    assert summary['hybrid']['arrived']==summary['fixed']['arrived']==summary['primitive']['arrived']==5
    assert summary['hybrid']['mean_sampled_alignment_s']<summary['fixed']['mean_sampled_alignment_s']
    assert summary['hybrid']['mean_nonzero_mode_changes']<summary['fixed']['mean_nonzero_mode_changes']
    report=read(FIG/'hybrid_evidence_audit.json')
    report.update(complete=True,completed_episodes=15,service_run=service['run'],
        visual_condition_pair=str(pair.relative_to(ROOT)),visual_pair_episodes=2,
        service_confirmations=4,service_proxy_overlap_count=0,figure_collision_qa=qa,
        all_source_hashes_checked=True,service_optimizer_matches_primary=True,
        primary_table_sha256=digest(FIG/'hybrid_local_table.tex'),
        manuscript_sha256=digest(ROOT/'paper/office_delivery_research_revision.tex'),
        method_sha256=digest(ROOT/'paper/grounded_continuation_research_method.tex'))
    validity=ARCHIVE/'analysis/baseline_input_validity.json'
    if validity.exists():
        report['public_baseline_comparison_valid']=read(validity)['ranking_usable']
        report['public_baseline_validity_audit']=str(validity.relative_to(ROOT))
        report['public_baseline_validity_sha256']=digest(validity)
        report['complete_scope']='Raw-record consistency only; not baseline qualification or submission readiness.'
    (FIG/'hybrid_evidence_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: 15 mechanism trials, 2 real visual-condition episodes, 4 service confirmations, frozen sources and figures.')

if __name__=='__main__':main()
