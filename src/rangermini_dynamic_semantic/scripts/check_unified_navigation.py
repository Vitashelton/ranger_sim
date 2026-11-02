#!/usr/bin/env python3
"""Small deterministic gates and desktop CPU probes; not Gazebo episodes."""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import platform
import random
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rangermini_dynamic_semantic.command_semantics_mppi import MPPIConfig, ranger_command
from rangermini_dynamic_semantic.observation_mppi import ObservationMPPI, ObservationConfig, project_control
from rangermini_dynamic_semantic.task_observation import Evidence, EvidenceStore, TaskScope, QueryBudget, view_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    checks = []
    def check(name, predicate):
        if not predicate:
            raise AssertionError(name)
        checks.append(name)
    scope = TaskScope('lab-map', 1, 'confirm614', 'lab614', 'marker')
    store = EvidenceStore(['lab614'], ttl_s=2)
    store.register_request('r1', scope, 1.0, (0., 0., 0.))
    e = Evidence('r1', scope, 1, 1.5, (0., 0., 0.), 'confirmed', True)
    check('valid evidence admitted', store.admit(e, scope, 1.5)[0])
    check('task version isolation', not store.admit(e, replace(scope, version=2), 1.5)[0])
    check('map isolation', not store.admit(e, replace(scope, map_id='other'), 1.5)[0])
    check('node isolation', store.state(replace(scope, node_id='confirm-other'), 1.5) == 'unresolved')
    check('expiry on query', store.state(scope, 3.1) == 'unresolved')
    check('capture envelope checked', not store.admit(replace(e, capture_pose=(1.,0.,0.)), scope, 1.5)[0])
    check('future evidence rejected', not store.admit(replace(e, captured_at=9), scope, 1.5)[0])
    store.register_request('r2', scope, 1.1, (0.,0.,0.))
    check('conflict blocks confirmation', store.admit(replace(e, request_id='r2', captured_at=1.1, value='absent'), scope, 1.5)[1] == 'conflicting')
    budget = QueryBudget(2, 1000, 2)
    check('query reservation', budget.reserve('a', 400, 1))
    check('pending reservations counted', not budget.reserve('b', 400, 1.1))
    check('settlement releases unused wait', budget.settle('a', .25) and budget.reserve('b', 400, 1))
    check('query budget exhaustion', not budget.reserve('c', 1, .1))
    hypotheses = ((2., 0., .43, 1.),)
    check('visibility range', 0 <= view_score((0,0,0), hypotheses, []) <= 1)
    check('behind camera visibility zero', view_score((0,0,math.pi), hypotheses, []) == 0)
    clear = view_score((0,0,0), hypotheses, [])
    blocked = view_score((0,0,0), hypotheses, [(1,0)])
    check('occupied ray reduces visibility', 0 <= blocked < clear)
    check('mixture normalized', abs(view_score((0,0,0), ((2,0,.43,2),), []) - clear) < 1e-12)
    rng = random.Random(7)
    cfg = MPPIConfig(allow_reverse=False, max_speed=.3)
    for behavior in ('ACKERMANN', 'PARALLEL', 'SPIN', 'DRIVE'):
        for _ in range(100):
            raw = tuple(rng.uniform(-2,2) for _ in range(3))
            u = project_control(raw, behavior, cfg)
            vx, vy, wz, mode, _ = ranger_command(*u)
            check(f'{behavior} feasible sample {len(checks)}',
                  vx >= 0 and math.hypot(vx,vy) <= .3+1e-9 and
                  (mode != 'DUAL_ACKERMANN' or abs(wz) <= vx/.4764+1e-9) and
                  (mode != 'PARALLEL' or wz == 0) and
                  (mode != 'SPINNING' or vx == vy == 0))
    low = ObservationMPPI(cfg, ObservationConfig(observation_weight=0))
    high = ObservationMPPI(cfg, ObservationConfig(observation_weight=6))
    p0 = low.plan((0,0,0), (.2,0), current_mode='STOPPED', hypotheses=((0,1,.43,1),))
    p1 = high.plan((0,0,0), (.2,0), current_mode='STOPPED', hypotheses=((0,1,.43,1),))
    check('view cost changes candidate scores', p0['candidate_summary'] != p1['candidate_summary'])
    check('view cost participates in action selection', p0['action'] != p1['action'])
    # Fixed input, no tuning from resulting CPU measurements.
    points = [(3*math.cos(i*.1), 3*math.sin(i*.1)) for i in range(60)]
    probes = []
    for n in (8,16,32,64):
        planner = ObservationMPPI(replace(cfg, samples_per_behavior=n))
        timings = []
        for k in range(5):
            result = planner.plan((0,0,0), (1,0), current_mode='STOPPED',
                                  hypotheses=hypotheses, obstacle_points=points)
            timings.append(result['planning_time_ms'])
        probes.append({'samples_per_group': n, 'candidate_count': result['candidate_count'],
                       'calls': len(timings), 'median_ms': statistics.median(timings),
                       'maximum_ms': max(timings), 'raw_ms': timings})
    out = {'evidence_type': 'pure_python_interface_checks_and_desktop_timing',
           'not_gazebo_or_hardware': True, 'checks_passed': len(checks),
           'check_names': checks, 'platform': platform.platform(),
           'python': platform.python_version(), 'cpu_probes': probes,
           'cost_effect_probe': {'without_view': p0, 'with_view': p1}}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(json.dumps({'checks_passed':len(checks), 'cpu_probes':probes}, indent=2))


if __name__ == '__main__':
    main()
