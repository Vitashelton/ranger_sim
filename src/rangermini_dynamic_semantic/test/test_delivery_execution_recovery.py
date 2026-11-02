"""Regression cases for failures recorded in the stable-v7 Gazebo batch."""
import math
from dataclasses import replace
from pathlib import Path

from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph, Track
from rangermini_dynamic_semantic.delivery_execution_recovery import DeliveryExecutionHarness
from rangermini_dynamic_semantic.command_semantics_mppi import MPPIConfig
from rangermini_dynamic_semantic.stable_interaction_mppi import StableInteractionMPPI


ROOT = Path(__file__).resolve().parents[3]
WORLD = ROOT/'artifacts/grounded_dynamic_delivery_20261014/matrix_v7_stable_extension/plain_grounded_s41/world.sdf'


def harness():
    events = []
    return DeliveryExecutionHarness(SpatialGraph(WORLD), 'Deliver to Office A 619', events.append), events


def test_parked_obstacle_wait_expires_without_claiming_encounter_success():
    h, events = harness()
    h.selected_behavior = h.interaction = 'WAIT'
    h.evidence_ids = ['track_1']
    h.decision_deadline = 4
    pose = (7.5018, -2.1900, -.6587)  # recorded seed31 stopped pose
    for tick in range(100):
        stamp = tick*.12
        t = Track('track_1', (8.6851, -1.1029), (0, 0), .48, stamp, 0)
        h.feedback(pose, stamp, [t])
    assert h.selected_behavior == 'CONTINUE'
    assert h.stage == 'APPROACH'
    assert sum(e['event'] == 'WAIT_REASSESSMENT_REQUIRED' for e in events) == 1
    assert not any(e['event'] == 'INTERACTION_CLEARED' for e in events)


def test_passed_trolley_is_not_reattached_to_a_kinematic_wait():
    h, events = harness()
    h.stage = 'CROSS'
    pose = (12.5296, -.9406, 1.3310)  # recorded seed41 doorway failure
    candidate = {'behavior_id': 'WAIT', 'goal': pose[:2]}
    for tick in range(100):
        stamp = tick*.12
        t = Track('track_4', (9.1399, -.9469), (0, 0), .42, stamp, 0)
        h.feedback(pose, stamp, [t])
        h.provisional(candidate, [t], pose, stamp)
    assert not h.evidence_ids
    assert not any(e['event'] == 'INTERACTION_CLEARED' for e in events)
    assert sum(e['event'] == 'EXECUTION_FEASIBILITY_CHANGED' for e in events) <= 4
    assert h.stage == 'CROSS'


def test_portal_capture_requires_lateral_alignment_and_heading_before_cross():
    h, events = harness()
    h.feedback((12.49, -1.24, .28), 1, [])
    assert h.stage == 'ALIGN'
    assert h.subgoal()[1] is None  # center before turning toward the portal
    h.feedback((12.53, -.94, 1.33), 2, [])
    assert h.stage == 'ALIGN'  # proximity to a jamb is insufficient
    h.feedback((12.80, -1.15, math.pi/2), 3, [])
    assert h.stage == 'CROSS'
    h.feedback((12.80, -.1, math.pi/2), 4, [])
    assert h.stage == 'ARRIVE'
    assert [e['event'] for e in h.stage_events] == [
        'APPROACH_VERIFIED', 'PORTAL_ALIGNED', 'PORTAL_CROSSED']


def test_expired_observation_revokes_binding_without_advancing_task():
    h, events = harness()
    h.selected_behavior = h.interaction = 'YIELD_LEFT'
    h.evidence_ids = ['missing_track']
    h.feedback((6, -2.1, 0), 10, [])
    assert h.selected_behavior == 'CONTINUE'
    assert h.stage == 'APPROACH'
    assert events[-1]['event'] == 'BINDING_EVIDENCE_EXPIRED'
    assert not any(e['event'] == 'INTERACTION_CLEARED' for e in events)


def test_portal_heading_requirement_breaks_zero_motion_local_minimum():
    graph = SpatialGraph(WORLD)
    selections = []
    for tolerance in (None, math.radians(10)):
        planner = StableInteractionMPPI(MPPIConfig(samples_per_behavior=6,
            horizon=40, dt=.12, max_speed=.35, random_seed=31, allow_reverse=False),
            static_boxes=graph.static_boxes)
        planner.obs = replace(planner.obs, arrival_radius=.06)
        planner.wait_action_penalty = 4
        planner.capture_heading_tolerance = tolerance
        planner.capture_spin_only = tolerance is not None
        candidates = planner.bank((12.7711, -1.1267, 1.2754), (12.8, -1.15),
            ([], 0, 0), [], 0, behavior_ids=('CONTINUE', 'WAIT'),
            goal_yaw=math.pi/2, current_mode='PARALLEL',
            current_steering_angles=(0, 0, 0, 0))
        selections.append(min((c for c in candidates if c['feasible']),
                              key=lambda c: c['plan']['predicted_score']))
    assert selections[0]['plan']['selected_behavior'] == 'STOP'
    assert abs(selections[1]['plan']['action'][2]) > .1
    endpoint = selections[1]['plan']['selected_trajectory'][-1]
    assert abs(endpoint[2]-math.pi/2) <= math.radians(10)
    assert math.dist(endpoint[:2], (12.7711, -1.1267)) < 1e-8
