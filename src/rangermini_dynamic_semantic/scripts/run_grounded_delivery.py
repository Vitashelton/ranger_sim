#!/usr/bin/env python3
"""Grounded task/evidence/behavior execution in the existing Gazebo laboratory."""
import argparse
import gzip
from functools import partial
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys
import time
import threading
import xml.etree.ElementTree as ET
import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = PKG.parents[1]
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT/'artifacts/group_meeting_20261013/scripts'))
import run_view_demo as demo
from run_office_delivery_mppi import OfficeDeliveryPilot, wrap, rect_circle_gap, rect_rect_gap
from grounded_delivery_scene import create_world, SCENARIOS
from motion_planning_research_scene import SCENARIOS as RESEARCH_SCENARIOS
from rangermini_dynamic_semantic.command_semantics_mppi import MPPIConfig
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph, LidarEvidenceTracker, GroundedNavigationHarness, Track
from rangermini_dynamic_semantic.delivery_execution_recovery import DeliveryExecutionHarness
from rangermini_dynamic_semantic.station_handoff import StationHandoffHarness
from rangermini_dynamic_semantic.interaction_mppi import InteractionMPPI
from rangermini_dynamic_semantic.stable_interaction_mppi import StableInteractionMPPI
from rangermini_dynamic_semantic.grounded_visual_proposal import (propose_navigation, propose_by_cost,
    propose_task_condition, admissible_candidates, replay_station_proposal, replay_task_condition)
from rangermini_dynamic_semantic.ranger_4ws_kinematics import Ranger4WSKinematics
from rangermini_dynamic_semantic.continuation_evaluator import ContinuationEvaluator
from rangermini_dynamic_semantic.execution_freshness import observation_fault
from rangermini_dynamic_semantic.footprint_route_guide import FootprintRouteGuide


INSTRUCTION = "把资料送到 Office A（619 室）门内，到站停稳。途中留意来往人员和器材推车，选择合适的通行或让行行为，保持门口出入空间。"
COLORS = {"CONTINUE": (40, 150, 250), "WAIT": (245, 185, 35),
          "PASS_LEFT": (75, 220, 95), "PASS_RIGHT": (245, 85, 100),
          "YIELD_LEFT": (80, 220, 220), "YIELD_RIGHT": (210, 110, 245)}
SPEED_BOUND = .35


def adapter_parameters(args):
    model = Ranger4WSKinematics(max_speed=SPEED_BOUND if args.verified_recovery else 1.5)
    return {'wheelbase': model.wheelbase, 'track': model.track,
            'min_turn_radius': .4764, 'max_linear_speed': model.max_speed,
            'max_steer_rate': model.max_steer_rate,
            'max_ackermann_angle': model.max_ackermann_angle,
            'max_parallel_angle': model.max_parallel_angle}


class GroundedDelivery(OfficeDeliveryPilot):
    def create_subscription(self, msg_type, topic, callback, qos_profile, **kwargs):
        if self.args.execution_architecture == 'process' and topic != '/harness/service_confirmation':
            from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
            group = MutuallyExclusiveCallbackGroup()
            self.io_groups.append(group)
            kwargs['callback_group'] = group
        # Keep newest physical measurements while MPPI computes. A deep sensor
        # queue makes a moving scene appear behind its actual acquisition time.
        if isinstance(qos_profile, int):
            qos_profile = 1
        stream = {'/odom': 'pose', '/scan': 'scan', '/motion_mode_debug': 'actuation', '/clock': 'clock'}.get(topic)
        if stream:
            original = callback
            def monitored(msg, info=None):
                received_wall = time.monotonic()
                if stream != 'clock':
                    self.state_version += 1
                if hasattr(self, 'runtime_timeline'):
                    self.runtime_timeline.emit(event='sensor_callback_begin', topic=topic,
                        state_version=self.state_version, ros_time_s=self.now(),
                        dds_received_system_ns=getattr(info, 'received_timestamp', None),
                        dds_source_system_ns=getattr(info, 'source_timestamp', None))
                window = self.args.scan_drop_window
                if stream == 'scan' and window and getattr(self, 'motion_started', False):
                    elapsed = self.now()-self.started
                    if window[0] <= elapsed < window[0]+window[1]:
                        return
                stamp = self.now()
                if stream == 'clock':
                    stamp = msg.clock.sec+msg.clock.nanosec*1e-9
                if hasattr(msg, 'header'):
                    stamp = msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
                valid = True
                if stream == 'pose':
                    p, q = msg.pose.pose.position, msg.pose.pose.orientation
                    valid = all(math.isfinite(v) for v in (p.x, p.y, p.z, q.x, q.y, q.z, q.w))
                    valid = valid and sum(v*v for v in (q.x, q.y, q.z, q.w)) > 1e-8
                elif stream == 'scan':
                    valid = (bool(msg.ranges) and math.isfinite(msg.angle_min) and
                             math.isfinite(msg.angle_increment) and msg.angle_increment > 0 and
                             any(not math.isnan(v) and v > 0 for v in msg.ranges))
                elif stream == 'actuation':
                    try:
                        payload = json.loads(msg.data)
                        angles = payload['wheel_steering_rad']
                        stamp = payload.get('stamp_ros_s', stamp)
                        valid = (payload.get('feedback_valid', True) and len(angles) == 4
                                 and all(math.isfinite(v) for v in angles))
                    except (ValueError, KeyError, TypeError):
                        valid = False
                if not valid:
                    self.sensor_observations[stream] = (float('nan'), time.monotonic())
                    return
                original(msg)
                self.sensor_observations[stream] = (stamp, received_wall)
                if hasattr(self, 'runtime_timeline'):
                    self.runtime_timeline.emit(event='sensor_callback_end', topic=topic,
                        acquisition_ros_s=stamp, ros_time_s=self.now(),
                        callback_wall_s=time.monotonic()-received_wall)
            callback = monitored
        return super().create_subscription(msg_type, topic, callback, qos_profile, **kwargs)

    def __init__(self, args):
        self.args = args
        self.active_task_condition = None
        self.active_condition_task_key = None
        self.condition_refresh_needed = False
        self.command_lock = threading.RLock()
        self.io_groups = []
        self.state_version = 0
        self.sensor_observations = {}
        self.last_sensor_fault = None
        super().__init__(args)
        from rangermini_dynamic_semantic.runtime_timeline import RuntimeTimeline
        self.runtime_timeline = RuntimeTimeline(self.path/'runtime_timeline.jsonl')
        self.runtime_timeline.emit(event='executor_configuration',
            architecture=args.execution_architecture, executor_threads=4 if args.execution_architecture == 'process' else 1,
            planning_callback_group='default_mutually_exclusive',
            sensor_callback_groups='per_topic' if args.execution_architecture == 'process' else 'default',
            ros_clock_domain='simulation', duration_clock_domain='monotonic')
        self.graph = SpatialGraph(args.world)
        self.registered_objects = []
        if args.registered_object_evidence:
            world_root = ET.parse(args.world).getroot().find('world')
            for model in world_root.findall('model'):
                if not (model.get('name') == 'entry_equipment_station' or
                        model.get('name', '').startswith('entry_equipment_station_')):
                    continue
                p = list(map(float, model.findtext('pose').split()))
                self.registered_objects.append(dict(evidence_id=f'map_object_{len(self.registered_objects)+1}',
                    position=(p[0],p[1]), radius=math.hypot(.35,.21),
                    registered_box=(p[0],p[1],p[5],.35,.21),
                    source='registered_static_map_geometry'))
        self.tracker = LidarEvidenceTracker(self.graph)
        if args.long_task:
            from rangermini_dynamic_semantic.long_service_task import LongServiceTaskHarness
            task_plan = json.loads(Path(args.task_plan).read_text())
            self.harness = LongServiceTaskHarness(self.graph, args.instruction, task_plan,
                self.record_event, confirmation=args.service_confirmation,
                binding_recovery=True)
        elif args.scenario in ('collection_delivery', 'collection_return_delivery'):
            from rangermini_dynamic_semantic.multiwaypoint_delivery import MultiwaypointDeliveryHarness
            self.harness = MultiwaypointDeliveryHarness(self.graph, args.instruction, self.record_event,
                return_to_collection=args.scenario == 'collection_return_delivery',
                binding_recovery=args.policy == 'grounded')
        elif args.station_handoff:
            self.harness = StationHandoffHarness(self.graph, args.instruction, args.world,
                self.record_event, binding_recovery=args.policy == 'grounded',
                terminal_pose_only=args.station_terminal_pose_only)
        elif args.verified_recovery:
            self.harness = DeliveryExecutionHarness(self.graph, args.instruction,
                self.record_event, binding_recovery=args.policy == 'grounded')
        else:
            self.harness = GroundedNavigationHarness(self.graph, args.instruction, self.record_event)
        if args.local_pose_goal:
            from rangermini_dynamic_semantic.local_pose_navigation import LocalPoseNavigationHarness
            self.harness = LocalPoseNavigationHarness(self.graph,args.instruction,args.local_pose_goal,
                self.record_event,binding_recovery=args.policy=='grounded')
        self.cfg = MPPIConfig(samples_per_behavior=args.samples, horizon=40, dt=.12,
            max_speed=SPEED_BOUND, lidar_pitch=0, random_seed=args.seed, allow_reverse=False,
            signed_parallel_velocity=args.station_handoff)
        planner_class = StableInteractionMPPI if args.stable_modes else InteractionMPPI
        self.planner = planner_class(self.cfg, prediction=args.policy != "geometric",
            static_boxes=self.graph.static_boxes)
        if args.hybrid_sequences:
            from rangermini_dynamic_semantic.hybrid_sequence_mppi import HybridSequenceMPPI
            self.planner = HybridSequenceMPPI(self.cfg, prediction=args.policy != 'geometric',
                static_boxes=self.graph.static_boxes)
            self.planner.maximum_switches=args.hybrid_max_switches
            self.planner.fixed_switch_timing=args.hybrid_fixed_timing
        if args.planner == 'nav2-omni':
            from rangermini_dynamic_semantic.nav2_omni_mppi import Nav2OmniMPPI
            self.planner = Nav2OmniMPPI(self.cfg, static_boxes=self.graph.static_boxes, samples=args.upstream_samples)
        elif args.planner not in ('conditioned','ranger-mppi'):
            from rangermini_dynamic_semantic.upstream_mppi import UpstreamMPPI
            self.planner = UpstreamMPPI(self.cfg, static_boxes=self.graph.static_boxes,
                variant=args.planner, samples=args.upstream_samples)
        self.planner.ideal_steering_rollout = args.ideal_steering_rollout
        self.planner.guide_support = (None if args.unprojected_behavior_guides else
            partial(self.graph.supports_pose, margin=.10))
        if args.verified_recovery:
            self.planner.wait_action_penalty = self.planner.obs.stop_action_penalty
        if args.execution_architecture == 'process':
            from rangermini_dynamic_semantic.planner_process import PlannerProcess
            self.planner = PlannerProcess(self.planner, self.runtime_timeline.emit)
        self.observation_log = gzip.open(self.path/'planner_observations.jsonl.gz', 'wt')
        self.continuation = ContinuationEvaluator(self.cfg, self.graph.static_boxes) if args.continuation_lookahead else None
        self.continuation_log = (self.path/'continuation.jsonl').open('w')
        self.route_guide = FootprintRouteGuide(self.graph) if args.map_recovery else None
        self.frozen_global_reference = None
        if args.frozen_global_reference:
            reference_path = Path(args.frozen_global_reference).resolve()
            reference = json.loads(reference_path.read_text())
            static_geometry = np.asarray(self.graph.static_boxes, dtype='<f8')
            static_digest = hashlib.sha256(static_geometry.tobytes()).hexdigest()
            if reference.get('static_geometry_sha256') != static_digest:
                raise RuntimeError('Frozen global reference was generated from a different static map')
            points = [tuple(map(float, p)) for p in reference['path_xy']]
            goal = tuple(self.harness.subgoal()[0])
            if math.dist(points[-1], goal) > .08:
                raise RuntimeError(f'Frozen global reference endpoint {points[-1]} does not match task goal {goal}')
            checker = FootprintRouteGuide(self.graph)
            for a, b in zip(points, points[1:]):
                yaw = math.atan2(b[1]-a[1], b[0]-a[0])
                if not checker.segment(a, b, yaw):
                    raise RuntimeError('Frozen global reference is not footprint-feasible in the static map')
            self.frozen_global_reference = dict(path=points, source=str(reference_path),
                sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                start_pose=reference['start_pose'], goal_pose=reference['goal_pose'])
            self.route_guide = checker
        if args.long_task:
            from rangermini_dynamic_semantic.long_service_task import ObservedRouteGuide
            self.route_guide = ObservedRouteGuide(self.graph)
        self.route_log = (self.path/'route_guidance.jsonl').open('w')
        self.goal = tuple(self.graph.nodes['H'+self.harness.portal[1:]]["pose"][:2])
        self.goal_yaw = self.graph.nodes['H'+self.harness.portal[1:]]["pose"][2]
        self.scan_stamp = None
        self.tracks = []
        self.bank = []
        self.last_bank = -1e9
        self.pending_stage = None
        self.last_query = -1e9
        self.query_epoch = -1
        self.task_epoch = 0
        self.task_key = self.semantic_task_key()
        self.pending_task_key = None
        self.last_query_task_epoch = -1
        self.sensor_ready_time = None
        self.query_reason = None
        self.actual_queries = 0
        self.vlm_http_attempts = 0
        self.accepted_proposals = 0
        self.overrides = 0
        self.rejections = 0
        self.wait_time = 0
        self.last_action = (0, 0, 0)
        self.command_lease_until = 0.
        self.trajectory_version = 0
        self.action_held_until = 0
        self.plan_times = []
        self.motion_started = False
        self.timing_log = (self.path/'control_timing.jsonl').open('w')
        self.command_timing_log = (self.path/'command_timing.jsonl').open('w')
        self.bank_diagnostic_log = (self.path/'behavior_bank.jsonl').open('w')
        self.previous_command_wall = None
        self.previous_tick_wall = None
        self.proposal_deferred = False
        self.active_proposal_query = None
        self.active_proposal_request_wall = None
        self.direct_last_query = -1e9
        self.direct_command_count = 0
        from std_msgs.msg import String
        from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.start_pub = self.create_publisher(String, "/harness/mission_start", qos)
        if args.long_task:
            self.task_state_pub = self.create_publisher(String, '/harness/task_state', qos)
            self.create_subscription(String, '/harness/service_confirmation', self.service_receipt, 10)
            (self.path/'long_task_plan.json').write_text(json.dumps(self.harness.task_plan, ensure_ascii=False, indent=2))
        for timer in list(self.timers):
            timer.cancel()
        self.create_timer(.12, self.tick)
        if args.execution_architecture == 'process':
            from rclpy.clock import Clock, ClockType
            from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
            self.command_group = MutuallyExclusiveCallbackGroup()
            self.create_timer(.05, self.publish_heartbeat, callback_group=self.command_group,
                              clock=Clock(clock_type=ClockType.STEADY_TIME))
        manifest = {"instruction": args.instruction, "scenario": args.scenario, "policy": args.policy,
            "long_task": args.long_task, "task_plan_path": args.task_plan,
            "service_confirmation": args.service_confirmation,
            "requested_real_time_factor": args.real_time_factor, "max_wall_time_s": args.max_wall_time,
            "execution_architecture": args.execution_architecture,
            "planner": args.planner, "upstream_samples": args.upstream_samples,
            "retain_corridor_heading": args.retain_corridor_heading,
            "station_handoff": args.station_handoff,
            "station_terminal_pose_only": args.station_terminal_pose_only,
            "station_proposal_replay": args.station_proposal_replay,
            "seed": args.seed, "mppi_config": asdict(self.cfg),
            "stable_motion_family_execution": args.stable_modes,
            "ideal_steering_rollout_ablation": args.ideal_steering_rollout,
            "hybrid_sequence_planning": args.hybrid_sequences,
            "hybrid_max_switches": args.hybrid_max_switches,
            "hybrid_fixed_timing": args.hybrid_fixed_timing,
            "local_pose_goal": args.local_pose_goal,
            "observation_max_age_s": .7, "observation_max_wall_age_s": 1.5,
            "scan_drop_window_sim_s": args.scan_drop_window,
            "footprint_supported_behavior_guides": not args.unprojected_behavior_guides,
            "timing_contract": {"timer_period_sim_s": .12,
                "reference_wall_budget_s": .12,
                "wall_budget_is_reference_not_real_time_guarantee": True},
            "execution_profile": "portal_capture_and_bounded_binding_v4" if args.verified_recovery else "frozen_v7",
            "decision_policy": args.policy,
            "proposal_selector": args.proposal_selector,
            "proposal_interface": args.proposal_interface,
            "proposal_domain": ("task_condition_no_motion_command; Ranger hybrid MPPI selects family and switching time"
                if args.proposal_interface == 'task_condition' else
                "feasible_and_evidence_supported; sole_behavior_requires_no_VLM"),
            "freeze_object_labels": args.freeze_object_labels,
            "registered_object_evidence": self.registered_objects,
            "continuation_lookahead": args.continuation_lookahead,
            "footprint_map_recovery": args.map_recovery,
            "obstacle_seeded_sampling": args.obstacle_seeded_sampling,
            "route_reference_update": ("frozen_static_global_reference; dynamic actors excluded from route map"
                if self.frozen_global_reference else "geometric_progress_independent_of_proposal_queries"),
            "frozen_global_reference": ({"path": self.frozen_global_reference['path'],
                "source_sha256": self.frozen_global_reference['sha256'],
                "static_geometry_sha256": hashlib.sha256(np.asarray(self.graph.static_boxes,dtype='<f8').tobytes()).hexdigest(),
                "planner": "FootprintRouteGuide static-map A* with line-of-sight shortcut",
                "dynamic_obstacles_used_for_global_route": False}
                if self.frozen_global_reference else None),
            "local_pose_completion": {"route_position_m":.07,"heading_rad":math.radians(5),"rotation_family_when_position_reached":True},
            "continuation_parameters": {"weight": .35, "horizon": 32, "samples_per_mode": 2, "activation_distance_m": 4.0},
            "rollout_4ws_parameters": asdict(Ranger4WSKinematics(
                wheelbase=self.cfg.wheelbase, max_speed=self.cfg.max_speed)),
            "adapter_4ws_parameters": adapter_parameters(args),
            "world_sha256": hashlib.sha256(Path(args.world).read_bytes()).hexdigest(),
            "spatial_graph": self.graph.snapshot(), "motion_model": "shared Ranger 4WS rigid-body pure-rolling model",
            "dynamic_state_source": "registered-static-map-filtered actual 2D lidar; temporal association and velocity fit",
            "object_semantics_source": "fixed tracker labels" if args.freeze_object_labels else "actual RGB-conditioned DeepSeek labels for issued track evidence IDs",
            "scene_future_or_entity_poses_consumed_by_policy": False,
            "goal_generation": "room/portal graph geometry and verified stage events",
            "semantic_provider": ("deepseek_direct_velocity" if args.policy == 'direct_vlm' else
                ("deepseek" if args.proposal_selector == "vlm" else "matched_cost_selector")) if args.policy in ('grounded', 'visual', 'direct_vlm') else "none",
            "direct_vlm_control": ({"output": "body_twist_vx_vy_wz", "mppi_used": False,
                "query_period_sim_s": args.direct_query_period,
                "linear_speed_limit_mps": self.cfg.max_speed,
                "angular_rate_limit_radps": .90,
                "command_hold": "latest VLM twist until replacement or wall-clock lease expiry"}
                if args.policy == 'direct_vlm' else None),
            "localization_source": "Gazebo model odometry; not a SLAM experiment"}
        (self.path/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    def record_event(self, row):
        self.events.write(json.dumps(row, ensure_ascii=False)+"\n")
        self.events.flush()

    def on_rgb(self, msg):
        if msg.encoding.lower() not in ('rgb8', 'bgr8'):
            return
        data = demo.np.frombuffer(bytes(msg.data), demo.np.uint8).reshape(msg.height, msg.step)
        self.rgb = data[:, :msg.width*3].reshape(msg.height, msg.width, 3).copy()
        if msg.encoding.lower() == 'bgr8':
            self.rgb = self.rgb[:, :, ::-1].copy()
        self.frame_time = msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        if self.frame_time-self.last_saved_rgb >= .20:
            demo.PILImage.fromarray(self.rgb).save(self.path/'frames'/f'{self.frame_index:05d}.jpg', quality=90)
            self.frame_log.write(json.dumps({'frame': self.frame_index, 'sim_time': self.frame_time,
                'pose': self.pose})+'\n'); self.frame_log.flush()
            self.frame_index += 1
            self.last_saved_rgb = self.frame_time

    def on_overview(self, msg):
        if msg.encoding.lower() not in ('rgb8', 'bgr8'):
            return
        stamp = msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        if stamp-self.last_saved_overview < .20:
            return
        data = demo.np.frombuffer(bytes(msg.data), demo.np.uint8).reshape(msg.height, msg.step)
        image = data[:, :msg.width*3].reshape(msg.height, msg.width, 3)
        if msg.encoding.lower() == 'bgr8':
            image = image[:, :, ::-1]
        demo.PILImage.fromarray(image).save(self.path/'overview_frames'/f'{self.overview_index:05d}.jpg', quality=88)
        self.overview_log.write(json.dumps({'frame': self.overview_index, 'sim_time': stamp,
            'pose': self.pose})+'\n'); self.overview_log.flush()
        self.overview_index += 1
        self.last_saved_overview = stamp

    def service_receipt(self, msg):
        if not self.args.long_task:
            return
        try:
            receipt = json.loads(msg.data)
            executed = self.mode_payload.get('executed', {})
            moving = max(math.hypot(executed.get('vx', 0), executed.get('vy', 0)),
                         abs(executed.get('wz', 0))) >= .025
            if (self.pose is None or moving or self.sensor_fault() or
                    not self.harness.at_service_pose(self.pose)):
                self.record_event({'event': 'SERVICE_CONFIRMATION_NOT_AT_REST', 'sim_time': self.now()})
                return
            self.harness.confirm(receipt, self.now())
        except (ValueError, TypeError, KeyError) as exc:
            self.record_event({'event': 'SERVICE_CONFIRMATION_MALFORMED', 'sim_time': self.now(), 'error': str(exc)})

    def semantic_task_key(self):
        return (self.harness.destination, self.harness.stage, self.harness.anchor_id,
                getattr(self.harness, 'stop_index', None))

    def current_kwargs(self):
        self.planner.obstacle_seeded_sampling = self.args.obstacle_seeded_sampling
        if getattr(self.args, 'proposal_interface', 'behavior') == 'task_condition':
            self.planner.active_motion_condition = self.active_task_condition
        self.planner.require_navigation_progress = bool(self.args.local_pose_goal) or self.args.long_task or self.args.scenario in (
            'collection_delivery', 'collection_return_delivery')
        self.planner.service_pose_sampling = (
            bool(self.args.local_pose_goal) or
            ((self.args.long_task or self.args.scenario in ('collection_delivery', 'collection_return_delivery')) and self.harness.service_active))
        if self.args.verified_recovery:
            radius = .04 if self.harness.stage in ('ALIGN', 'CROSS') else .24
            self.planner.obs = replace(self.planner.obs, arrival_radius=radius)
            heading_capture = ((self.harness.stage == 'ALIGN' and self.harness.alignment_phase == 'HEADING') or
                               (self.args.long_task and self.harness.stage == 'EXIT_ALIGN'))
            spin_only = heading_capture and not self.args.unrestricted_task_motion
            self.planner.capture_spin_only = spin_only
            self.planner.capture_heading_tolerance = math.radians(10) if heading_capture else None
        if self.planner.service_pose_sampling:
            # The exit-centering verifier requires <=0.12 m. Its planning
            # capture radius must be tighter, otherwise STOP can be optimal
            # before the harness is permitted to advance.
            radius = .08 if self.args.long_task and self.harness.stage == 'EXIT_CENTER' else .16
            self.planner.obs = replace(self.planner.obs, arrival_radius=radius,
                arrival_heading_tolerance=math.radians(8))
        if self.args.station_handoff:
            stage = self.harness.stage
            self.planner.active_motion_condition = None
            if stage.startswith(('STATION_', 'HANDOFF')):
                self.planner.obs = replace(self.planner.obs, arrival_radius=.04)
                self.planner.capture_spin_only = stage == 'HANDOFF_ALIGN'
                self.planner.capture_heading_tolerance = math.radians(10) if stage == 'HANDOFF_ALIGN' else None
                condition = self.harness.motion_condition
                if condition:
                    self.goal = tuple(condition['target_pose'][:2])
                    self.goal_yaw = condition['target_pose'][2]
                if stage == 'HANDOFF_APPROACH' and not self.args.station_terminal_pose_only:
                    self.planner.active_motion_condition = dict(retain_heading=True,
                        heading=condition['target_pose'][2],
                        heading_tolerance_rad=condition['heading_tolerance_rad'])
        if getattr(self.args, 'retain_corridor_heading', False):
            self.planner.active_motion_condition = (dict(retain_heading=True, heading=0., heading_tolerance_rad=math.radians(18))
                if self.harness.stage == 'APPROACH' and
                   (self.args.scenario != 'heading_constrained_passage' or self.pose[0] <= 10.) else None)
        feedback = self.mode_payload
        extra = {}
        if self.args.planner == 'nav2-omni':
            velocity = feedback.get('executed', {})
            extra['current_body_velocity'] = tuple(velocity.get(k,0.) for k in ('vx','vy','wz'))
        return {**extra,"current_mode": feedback.get('mode', self.mode), "goal_yaw": self.harness.subgoal()[1],
                "current_steering_angles": feedback.get('wheel_steering_rad'),
                "current_steering_transition": feedback.get('steering_transition_stop', False)}

    def refresh_route_reference(self, now):
        """Advance the geometric reference from motion, independently of VLM events."""
        if self.frozen_global_reference:
            goal = tuple(self.harness.subgoal()[0])
            route = self.frozen_global_reference['path']
            if self.harness.stage != 'APPROACH':
                self.planner.route_goal = self.planner.route_yaw = None
                self.planner.reference_path = None
                self.planner.reference_pose_path = None
                self._route_reference_state = None
                return 0.
            if math.dist(route[-1], goal) > .08:
                raise RuntimeError('Task goal changed; refusing to mutate the frozen global route')
            nearest = min(range(len(route)), key=lambda i: math.dist(self.pose[:2], route[i]))
            lookahead = min(1.0, max(.25, math.dist(self.pose[:2], route[-1])))
            index = nearest
            remaining = lookahead
            while index+1 < len(route) and remaining > 0:
                edge = math.dist(route[index], route[index+1])
                if edge >= remaining:
                    alpha = remaining/max(edge, 1e-9)
                    guide = (route[index][0]+alpha*(route[index+1][0]-route[index][0]),
                             route[index][1]+alpha*(route[index+1][1]-route[index][1]))
                    break
                remaining -= edge
                index += 1
            else:
                guide = route[-1]
            tangent_index = min(index, len(route)-2)
            yaw = math.atan2(route[tangent_index+1][1]-route[tangent_index][1],
                             route[tangent_index+1][0]-route[tangent_index][0])
            self.planner.route_goal = guide
            self.planner.route_yaw = yaw
            self.planner.reference_path = route
            self.planner.reference_pose_path = None
            self._route_reference_state = {'pose':tuple(self.pose),'goal':goal}
            self.route_log.write(json.dumps({'sim_time':now,'pose':self.pose,'task_goal':goal,
                'guide':guide,'guide_yaw':yaw,'path':route,'wall_ms':0.,
                'trigger':'progress_on_frozen_static_global_reference',
                'semantic_query_required':False,'dynamic_obstacles_in_reference':False})+'\n')
            self.route_log.flush()
            return 0.
        if not self.route_guide:
            return 0.
        if (self.args.long_task and not self.args.static_global_map and
                self.route_guide.update_tracks(self.tracks, now)):
            self._route_reference_state = None
        goal = tuple(self.harness.subgoal()[0])
        previous = getattr(self, '_route_reference_state', None)
        guide = getattr(self.planner, 'route_goal', None)
        if self.harness.stage != 'APPROACH':
            self.planner.route_goal = self.planner.route_yaw = None
            self.planner.reference_path = None
            self.planner.reference_pose_path = None
            self._route_reference_state = None
            return 0.
        if previous is not None and previous['goal'] == goal:
            moved = math.dist(self.pose[:2], previous['pose'][:2])
            turned = abs(wrap(self.pose[2]-previous['pose'][2]))
            reached = guide is not None and math.dist(self.pose[:2],guide) <= self.cfg.dt*self.cfg.max_speed+4*self.route_guide.resolution
            if moved < 4*self.route_guide.resolution and turned < .10 and not reached:
                return 0.
        started = time.perf_counter()
        if self.args.local_pose_goal:
            guide,path = self.route_guide.guide(self.pose,goal,target_yaw=self.goal_yaw)
        else:
            guide,path = self.route_guide.guide(self.pose,goal)
        self.planner.route_goal = guide
        self.planner.route_yaw = self.route_guide.guide_yaw
        self.planner.reference_path = path
        self.planner.reference_pose_path = list(self.route_guide.pose_path)
        self._route_reference_state = {'pose':tuple(self.pose),'goal':goal}
        elapsed = (time.perf_counter()-started)*1000
        self.route_log.write(json.dumps({'sim_time':now,'pose':self.pose,'task_goal':goal,
            'guide':guide,'guide_yaw':self.route_guide.guide_yaw,'path':path,
            'pose_path':self.route_guide.pose_path,'wall_ms':elapsed,
            'trigger':'geometric_progress','semantic_query_required':False})+'\n')
        self.route_log.flush()
        return elapsed

    def call_planner(self, method, *args, **kwargs):
        started = time.monotonic()
        self.runtime_timeline.emit(event='planner_call_begin', method=method,
            state_version=self.state_version, ros_time_s=self.now())
        try:
            value=getattr(self.planner, method)(*args, **kwargs)
            if self.args.hybrid_sequences and method in ('bank','behavior_plan'):
                candidates=value if isinstance(value,list) else [value]
                for candidate in candidates:
                    plan=candidate.get('plan',{})
                    schedule=plan.get('selected_schedule')
                    key=(candidate.get('behavior_id'),str(schedule))
                    if key!=getattr(self,'_last_hybrid_record_key',None):
                        with gzip.open(self.path/'hybrid_rollouts.jsonl.gz','at') as stream:
                            stream.write(json.dumps(dict(kind='actual planner return, not executed future motion',
                                request_sim_time=args[4],pose=list(args[0]),main_goal=list(args[1]),
                                current_mode=kwargs.get('current_mode'),
                                wheel_steering_rad=kwargs.get('current_steering_angles'),
                                behavior_id=candidate.get('behavior_id'),plan=plan))+'\n')
                        self._last_hybrid_record_key=key
            return value
        finally:
            self.runtime_timeline.emit(event='planner_call_end', method=method,
                state_version=self.state_version, duration_wall_s=time.monotonic()-started,
                ros_time_s=self.now())

    def refresh_bank(self, now):
        route_ms = self.refresh_route_reference(now)
        self.bank = self.call_planner('bank', self.pose, self.harness.subgoal()[0],
            (self.scan.ranges, self.scan.angle_min, self.scan.angle_increment),
            self.tracks, now,
            behavior_ids=(('CONTINUE',) if (self.args.planner in ('ranger-mppi','nav2-omni') or
                getattr(self.args, 'proposal_interface', 'behavior') == 'task_condition') else
                ('CONTINUE', 'WAIT') if self.args.verified_recovery and self.harness.stage in ('ALIGN', 'CROSS') else None),
            **self.current_kwargs())
        if self.args.station_handoff and self.harness.stage == 'HANDOFF_SELECT':
            # Both selectors receive the same spatial relations and native-family
            # rollouts. The instruction is NOT parsed into a yaw by the controller.
            self.bank = []
            for name, condition in self.harness.station_choices.items():
                kwargs = self.current_kwargs()
                kwargs['goal_yaw'] = condition['target_pose'][2]
                candidate = self.call_planner('behavior_plan', self.pose, self.harness.staging,
                    (self.scan.ranges, self.scan.angle_min, self.scan.angle_increment),
                    self.tracks, now, 'CONTINUE', **kwargs)
                candidate.update(behavior_id=name, grounding_kind='registered_station',
                                 motion_condition=condition)
                self.bank.append(candidate)
        for candidate in self.bank:
            yaw = self.harness.subgoal()[1] if candidate['behavior_id'] == 'CONTINUE' else self.pose[2]
            if candidate.get('motion_condition'):
                yaw = candidate['motion_condition']['target_pose'][2]
            if candidate['behavior_id'] == 'CONTINUE' and getattr(self.planner, 'route_goal', None) is not None:
                yaw = self.planner.route_yaw
            # Public path tracking retains the terminal pose while the native
            # planner exposes its local guide. Check the orientation belonging
            # to this candidate's goal, not the unrelated staging guide's yaw.
            yaw = candidate.get('goal_yaw', yaw)
            supported = self.graph.supports_pose((*candidate['goal'], self.pose[2] if yaw is None else yaw))
            candidate['goal_supported'] = supported
            if not supported:
                candidate['feasible'] = False
                candidate['rejection_reason'] = 'goal_footprint_overlaps_registered_static_geometry'
        self.last_bank = now
        self.planner.bank_time_ms += route_ms
        if self.continuation and self.harness.stage == 'APPROACH':
            door = self.graph.doors[self.harness.portal]
            target = (door['center_x']+.01, door['plane_y']-door['direction']*.85)
            if math.dist(self.pose[:2], target) <= 4.0:
                record = self.continuation.evaluate(self.bank, self.pose,
                    (self.scan.ranges, self.scan.angle_min, self.scan.angle_increment),
                    self.tracks, now, target, door['direction']*math.pi/2)
                self.continuation_log.write(json.dumps(record)+'\n')
                self.continuation_log.flush()
                self.planner.bank_time_ms += record['wall_ms']
        self.plan_times.append(self.planner.bank_time_ms)
        self.bank_diagnostic_log.write(json.dumps({'sim_time': now,
            'wall_monotonic_s': time.monotonic(), 'stage': self.harness.stage,
            'pose': list(self.pose), 'active_behavior': self.harness.selected_behavior,
            'bank_wall_ms': self.planner.bank_time_ms,
            'candidates': [{'behavior': c['behavior_id'], 'feasible': c['feasible'],
                'goal': c['goal'], 'raw_goal': c.get('raw_goal'),
                'guide_projected': c.get('guide_projected', False), 'cost': c['plan']['predicted_score'],
                'motion_family': c['plan']['selected_behavior'],
                'cost_terms': c['plan']['cost_terms']} for c in self.bank]})+'\n')
        self.bank_diagnostic_log.flush()

    def proposal_image(self):
        from rangermini_dynamic_semantic.candidate_view import render_candidate_view
        candidates = []
        if self.args.proposal_interface == 'task_condition':
            # Show the actual optimized hybrid schedules to the model, while
            # making clear that it returns task constraints rather than
            # selecting one schedule or prescribing a movement mode.
            for parent in self.bank:
                plan = parent.get('plan', {})
                for family in plan.get('candidate_summary', []):
                    if not family.get('trajectory'):
                        continue
                    candidates.append({
                        'behavior_id': f"SEQ_{len(candidates)}", 'feasible': True,
                        'grounding_kind': 'optimizer_generated_primitive',
                        'goal': parent['goal'], 'cost': family['cost'],
                        'predicted_clearance_m': family.get('predicted_min_clearance_m'),
                        'motion_behavior': family['behavior'],
                        'motion_sequence': [], 'switch_times_s': [],
                        'trajectory': family['trajectory'],
                    })
                schedules = [s for s in plan.get('hybrid_schedule_candidates', [])
                             if s.get('feasible') and s.get('trajectory')]
                schedules = sorted(schedules, key=lambda s: s.get('cost', float('inf')))[:max(0, 6-len(candidates))]
                for index, schedule in enumerate(schedules):
                    structure = schedule.get('structure', [])
                    motion_labels = {'A': 'CURVE', 'P': 'PARALLEL', 'R': 'SPIN'}
                    motion_sequence = [motion_labels.get(family, family) for family in structure]
                    candidates.append({
                        'behavior_id': f"SEQ_{len(candidates)}",
                        'feasible': True,
                        'grounding_kind': 'optimizer_generated_hybrid_sequence',
                        'goal': parent['goal'],
                        'cost': schedule.get('cost'),
                        'predicted_clearance_m': schedule.get('predicted_min_clearance_m'),
                        'motion_behavior': '-'.join(motion_sequence),
                        'motion_sequence': motion_sequence,
                        'switch_times_s': schedule.get('switch_times_s', []),
                        'trajectory': schedule['trajectory'],
                        'motion_family_options': [{'family': f} for f in structure],
                    })
        if not candidates:
            candidates = None
        if candidates is None:
            candidates = []
            for index, candidate in enumerate(self.bank):
                behavior = candidate["behavior_id"]
                plan = candidate.get('plan', {})
                clearance = plan.get('predicted_min_clearance_m',
                                     plan.get('min_clearance_m', plan.get('minimum_clearance_m')))
                candidates.append({"behavior_id": behavior, "feasible": candidate['feasible'],
                    "grounding_kind": candidate.get('grounding_kind'),
                    "motion_condition": candidate.get('motion_condition'),
                    "continuation": candidate.get('continuation'),
                    "goal": candidate['goal'], "cost": plan.get('predicted_score'),
                    "trajectory": plan.get('selected_trajectory', []),
                    "predicted_clearance_m": clearance,
                    "motion_behavior": plan.get('selected_behavior'),
                    "motion_sequence": (plan.get('selected_schedule') or {}).get('structure', []),
                    "switch_times_s": (plan.get('selected_schedule') or {}).get('switch_times_s', []),
                    "motion_family_options": [
                        {"family": g['behavior'], "cost": g['cost'],
                         "predicted_clearance_m": g.get('predicted_min_clearance_m')}
                        for g in plan.get('candidate_summary', [])]})
        views=[dict(c,trajectory=c.get('trajectory', [])) for c in candidates]
        objects=[dict(position=list(t.position),radius=t.radius,evidence_id=t.evidence_id) for t in self.tracks]
        image,visibility=render_candidate_view(self.rgb,self.pose,views,objects)
        for c,v in zip(candidates,visibility):
            c.update(candidate_id=v['candidate_id'],visible_camera_points=v['visible_points'],
                     visible_camera_segments=v['visible_segments'],color_rgb=v['color_rgb'])
        ok, encoded = demo.cv2.imencode('.jpg', image[:, :, ::-1], [demo.cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise ValueError("candidate image encoding failed")
        return bytes(encoded), candidates, image

    def start_semantic_query(self, now, reason):
        if self.pending is not None:
            return
        if self.args.station_handoff and self.harness.stage == 'HANDOFF_SELECT':
            stopped_since = self.harness.station_stop_since
            if stopped_since is None or now-stopped_since < .35:
                return
        if not any(c['feasible'] for c in self.bank):
            return
        if (self.args.verified_recovery and self.motion_started and
                not any(c['feasible'] and c['behavior_id'] != 'WAIT' for c in self.bank)):
            return
        self.actual_queries += 1
        jpg, candidates, image = self.proposal_image()
        context = self.harness.context(self.pose, self.tracks)
        context['semantic_task_epoch'] = self.task_epoch
        context['motion_state_version'] = self.state_version
        context['chassis_state'] = {
            'mode': self.mode,
            'wheel_steering_rad': self.mode_payload.get('wheel_steering_rad'),
            'steering_alignment_active': self.mode_payload.get('steering_transition_stop', False),
            'state_source': 'kinematic simulation; not hardware actuator feedback'}
        if self.args.proposal_interface == 'task_condition':
            context['spatial_conditions_only'] = self.args.spatial_conditions_only
            context['local_planning_reference'] = {
                'registered_task_goal': list(self.harness.subgoal()[0]),
                'route_guide': list(self.planner.route_goal) if self.planner.route_goal is not None else None,
                'route_heading_rad': self.planner.route_yaw,
                'reference_source': ('frozen static-map global route' if self.frozen_global_reference
                                     else 'registered-map local route guidance'),
                'dynamic_objects_in_global_route': False if self.frozen_global_reference else None}
        if self.args.policy == 'visual':
            # Direct visual-selector ablation: same task goal, current sensor
            # objects, candidates, model and safety gate; no event retrieval.
            context = {k: v for k, v in context.items()
                       if k not in ('recent_events', 'space', 'execution_contract')}
        if self.args.proposal_interface == 'task_condition':
            context['decision_domain'] = {
                'anchor_id': context['anchor_id'],
                'evidence_ids':[o['evidence_id'] for o in context['objects']],
                'motion_families': ['curved_driving', 'parallel_or_oblique_translation',
                                    'rotate_then_translate'],
                'switching_time_search': bool(self.args.hybrid_sequences),
                'soft_motion_preference_output': not self.args.spatial_conditions_only,
                'explicit_motion_command_output': False}
        else:
            context['decision_domain'] = {
                'behavior_ids':[c['behavior_id'] for c in admissible_candidates(context,candidates)],
                'evidence_ids':[o['evidence_id'] for o in context['objects']]}
        self.pending_stage = self.harness.stage
        self.pending_observation_heading = float(self.pose[2])
        self.pending_task_key = self.semantic_task_key()
        self.last_query_task_epoch = self.task_epoch
        self.pending_epoch = self.harness.query_epoch
        self.last_query = now
        self.query_epoch = self.harness.query_epoch
        self.query_reason = reason
        demo.PILImage.fromarray(image).save(self.path/f'proposal_{self.actual_queries:03d}_input.jpg')
        demo.PILImage.fromarray(self.rgb).save(self.path/f'proposal_{self.actual_queries:03d}_raw_rgb.jpg')
        (self.path/f'proposal_{self.actual_queries:03d}_input.json').write_text(
            json.dumps({"context": context, "candidates": candidates, "trigger": reason}, ensure_ascii=False, indent=2))
        (self.path/f'proposal_{self.actual_queries:03d}_rollouts.json').write_text(json.dumps([
            {"behavior_id": c['behavior_id'], "goal": c['goal'], "feasible": c['feasible'],
             "trajectory": c['plan']['selected_trajectory'],
             "motion_behavior": c['plan']['selected_behavior'],
             "native_family_candidates": c['plan'].get('candidate_summary', [])}
             for c in self.bank]))
        selector = (propose_task_condition if self.args.proposal_interface == 'task_condition' else
                    propose_navigation if self.args.proposal_selector == 'vlm' else propose_by_cost)
        replay = self.args.station_proposal_replay and self.harness.stage == 'HANDOFF_SELECT'
        task_replay = bool(self.args.task_condition_replay)
        if task_replay:
            self.pending = self.pool.submit(replay_task_condition, self.args.task_condition_replay,
                self.args.task_condition_default_replay, jpg, context, candidates)
        elif replay:
            self.pending = self.pool.submit(replay_station_proposal, self.args.station_proposal_replay,
                jpg, context, candidates)
        else:
            self.pending = self.pool.submit(selector, jpg, context, candidates)
        self.pending_started_wall = time.monotonic()
        self.proposal_deferred = False
        self.record_event({"event": "SEMANTIC_QUERY_STARTED", "sim_time": now,
            "query": self.actual_queries, "trigger": reason, "stage": self.harness.stage,
            "selector": self.args.proposal_selector,
            "requires_vlm": not replay and not task_replay and self.args.proposal_selector == "vlm" and
                (self.args.proposal_interface == 'task_condition' or
                 len(context["decision_domain"].get("behavior_ids", [])) > 1),
            "proposal_interface": self.args.proposal_interface,
            "recorded_station_proposal": bool(replay), "recorded_task_condition": task_replay})

    def relevant_encounter(self):
        relevant = []
        goal, _ = self.harness.subgoal()
        delta = demo.np.asarray(goal)-demo.np.asarray(self.pose[:2])
        velocity = delta/max(.01, float(demo.np.linalg.norm(delta)))*.25
        for t in self.tracks:
            p = demo.np.asarray(t.position)-demo.np.asarray(self.pose[:2])
            rel = demo.np.asarray(t.velocity)-velocity
            closest_time = max(0., min(8., -float(p@rel)/max(.01, float(rel@rel))))
            miss = float(demo.np.linalg.norm(p+closest_time*rel))
            if float(demo.np.linalg.norm(t.velocity)) > .055 and miss < 1.1 and 0 < closest_time < 8:
                relevant.append(t.evidence_id)
        return relevant

    def start_direct_query(self, now):
        from rangermini_dynamic_semantic.visual_behavior_selector import select_direct_velocity
        ok, encoded = demo.cv2.imencode('.jpg', self.rgb[:, :, ::-1],
            [demo.cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError('could not encode current RGB observation')
        target, target_yaw = self.harness.subgoal()
        dx, dy = target[0]-self.pose[0], target[1]-self.pose[1]
        c, s = math.cos(self.pose[2]), math.sin(self.pose[2])
        state = {
            'robot_pose_map': [round(v, 3) for v in self.pose],
            'subgoal_relative_body_m': [round(c*dx+s*dy, 3), round(-s*dx+c*dy, 3)],
            'subgoal_heading_error_rad': round(wrap(target_yaw-self.pose[2]), 3) if target_yaw is not None else None,
            'executed_body_velocity_mps_radps': self.mode_payload.get('executed', {}),
            'simulated_wheel_angles_rad': self.mode_payload.get('wheel_steering_rad'),
            'simulated_chassis_mode': self.mode_payload.get('mode'),
            'note': 'wheel state is from the Gazebo kinematic adapter, not actuator feedback'}
        self.actual_queries += 1
        self.direct_last_query = now
        query = self.actual_queries
        jpg = bytes(encoded)
        demo.PILImage.fromarray(self.rgb).save(self.path/f'direct_{query:03d}_rgb.jpg', quality=90)
        (self.path/f'direct_{query:03d}_input.json').write_text(json.dumps(
            {'instruction': self.args.instruction, 'state': state, 'query_sim_time': now},
            ensure_ascii=False, indent=2))
        self.pending = self.pool.submit(select_direct_velocity, jpg, self.args.instruction, state)
        self.pending_started_wall = time.monotonic()
        self.record_event({'event': 'DIRECT_VLM_QUERY_STARTED', 'sim_time': now,
            'query': query, 'input': f'direct_{query:03d}_input.json'})

    def tick_direct_vlm(self, now):
        if self.pending is not None and self.pending.done():
            query = self.actual_queries
            try:
                response = self.pending.result()
                (self.path/f'direct_{query:03d}_response.json').write_text(
                    json.dumps(response, ensure_ascii=False, indent=2))
                self.last_action = tuple(response['command'])
                self.vlm_http_attempts += 1
                self.accepted_proposals += 1
                self.direct_command_count += 1
                # Bound stale open-loop commands. A new VLM response renews this
                # wall-clock lease; ordinary timer heartbeats do not.
                lease_s = max(3., self.args.direct_query_period /
                    max(.1, self.args.real_time_factor) + 2.)
                self.command_lease_until = time.monotonic()+lease_s
                if not self.motion_started:
                    self.activate_motion(now)
                self.record_event({'event': 'DIRECT_VLM_COMMAND_ACCEPTED',
                    'sim_time': now, 'query': query, 'command': list(self.last_action),
                    'latency_s': response['latency_s'], 'lease_wall_s': lease_s})
            except Exception as exc:
                self.vlm_errors += 1
                self.vlm_http_attempts += 1
                self.last_action = (0., 0., 0.)
                self.command_lease_until = 0.
                (self.path/f'direct_{query:03d}_error.json').write_text(
                    json.dumps({'error': str(exc)}, ensure_ascii=False, indent=2))
                self.record_event({'event': 'DIRECT_VLM_COMMAND_FAILED',
                    'sim_time': now, 'query': query, 'error': str(exc)})
            finally:
                self.pending = None
        if self.pending is None and (not self.motion_started or
                now-self.direct_last_query >= self.args.direct_query_period):
            try:
                self.start_direct_query(now)
            except Exception as exc:
                self.vlm_errors += 1
                self.record_event({'event': 'DIRECT_VLM_QUERY_START_FAILED',
                    'sim_time': now, 'error': str(exc)})
        action = self.last_action if time.monotonic() < self.command_lease_until else (0., 0., 0.)
        self.publish(action)
        self.last_action = action
        self.trace.write(json.dumps({'timestamp': now, 'elapsed': now-self.started if self.motion_started else 0.,
            'pose': self.pose, 'task_stage': self.harness.stage, 'interaction': 'DIRECT_VLM_TWIST',
            'anchor_id': self.harness.anchor_id, 'local_subgoal': self.harness.subgoal()[0],
            'behavior_id': 'DIRECT_VLM_TWIST', 'semantic_behavior': 'DIRECT_VLM_TWIST',
            'tracks': [t.to_dict() for t in self.tracks], 'encounter_ids': [],
            'command': action, 'actuation': self.mode_payload,
            'plan': {'controller': 'direct_vlm_body_twist', 'mppi_used': False,
                     'query_pending': self.pending is not None,
                     'command_lease_active': time.monotonic() < self.command_lease_until}})+'\n')
        self.trace.flush()

    def activate_motion(self, now):
        if self.motion_started:
            return
        from std_msgs.msg import String
        self.started = now
        self.motion_started = True
        self.start_pub.publish(String(data=json.dumps({"sim_time": now})))
        self.record_event({"event": "MISSION_EXECUTION_STARTED", "sim_time": now,
            "stage": self.harness.stage, "policy": self.args.policy})

    def publish_heartbeat(self):
        # A heartbeat is NOT a new trajectory or a renewed trajectory lease.
        with self.command_lock:
            fault = self.sensor_fault()
            expired = time.monotonic() >= self.command_lease_until
            action = (self.last_action if not self.done and not fault and not expired
                      else (0., 0., 0.))
            super().publish(action)
        self.runtime_timeline.emit(event='command_heartbeat', ros_time_s=self.now(),
            command=list(action), observation_fault=fault, trajectory_lease_expired=expired,
            trajectory_version=self.trajectory_version, state_version=self.state_version)

    def sensor_fault(self):
        return observation_fault(self.now(), time.monotonic(), self.sensor_observations)

    def publish(self, action):
        fault = self.sensor_fault()
        if fault:
            action = (0., 0., 0.)
            self.last_action = action
        if hasattr(self, 'harness') and fault != self.last_sensor_fault:
            self.record_event({'event': 'OBSERVATION_HOLD' if fault else 'OBSERVATION_RECOVERED',
                'sim_time': self.now(), 'reason': fault,
                'observations': self.sensor_observations})
            # Observation faults invalidate motion, not semantic task identity.
            self.last_sensor_fault = fault
        if hasattr(self, 'runtime_timeline'):
            self.runtime_timeline.emit(event='command_publish', ros_time_s=self.now(),
                state_version=self.state_version, trajectory_version=self.trajectory_version,
                command=list(action), observation_fault=fault,
                semantic_identity={'task_epoch': self.task_epoch,
                    'object_id': list(self.harness.evidence_ids),
                    'behavior_id': self.harness.selected_behavior} if hasattr(self, 'harness') else None)
        with self.command_lock:
            self.last_action = action
            super().publish(action)
        if hasattr(self, 'command_timing_log'):
            wall = time.monotonic()
            self.command_timing_log.write(json.dumps({
                'wall_monotonic_s': wall, 'sim_time': self.now(),
                'interval_wall_s': None if self.previous_command_wall is None else wall-self.previous_command_wall,
                'command': list(action), 'observation_fault': fault, 'vlm_pending': self.pending is not None,
                'active_proposal_query': self.active_proposal_query,
                'active_proposal_request_wall_s': self.active_proposal_request_wall,
                'motion_started': self.motion_started})+'\n')
            self.command_timing_log.flush()
            self.previous_command_wall = wall

    def tick(self):
        if not hasattr(self, 'timing_log'):
            return self._tick_impl()
        wall = time.monotonic()
        self.runtime_timeline.emit(event='planning_callback_begin', state_version=self.state_version)
        sim = self.now()
        count = len(self.plan_times)
        before = self.harness.selected_behavior
        pending = self.pending is not None
        try:
            return self._tick_impl()
        finally:
            duration = time.monotonic()-wall
            self.runtime_timeline.emit(event='planning_callback_end', state_version=self.state_version,
                                       duration_wall_s=duration)
            self.timing_log.write(json.dumps({
                'wall_monotonic_s': wall, 'sim_time': sim,
                'callback_wall_s': duration,
                'start_interval_wall_s': None if self.previous_tick_wall is None else wall-self.previous_tick_wall,
                'reference_wall_budget_s': .12, 'reference_budget_overrun': duration>.12,
                'bank_wall_ms': self.plan_times[count:],
                'vlm_pending_at_start': pending, 'vlm_pending_at_end': self.pending is not None,
                'behavior_before': before, 'behavior_after': self.harness.selected_behavior,
                'steering_alignment': self.mode_payload.get('steering_transition_stop', False),
                'motion_started': self.motion_started})+'\n')
            self.timing_log.flush()
            self.previous_tick_wall = wall

    def _tick_impl(self):
        if self.done or not hasattr(self, 'harness'):
            return
        now = self.now()
        if self.pose is None or self.scan is None or self.rgb is None or self.depth is None:
            self.publish((0., 0., 0.))
            return
        if self.motion_started and now-self.started > self.args.max_sim_time:
            self.finish('TIMEOUT')
            return
        if self.sensor_fault():
            self.publish((0., 0., 0.))
            return
        if self.sensor_ready_time is None:
            self.sensor_ready_time = now
        planning_pose = tuple(self.pose)
        planning_angles = tuple(self.mode_payload.get('wheel_steering_rad', ()))
        planning_wall = time.monotonic()
        planning_state_version = self.state_version
        scan_time = self.scan.header.stamp.sec+self.scan.header.stamp.nanosec*1e-9
        if scan_time != self.scan_stamp:
            capture_pose = min(list(self.history), key=lambda r: abs(r[0]-scan_time))[1] if self.history else self.pose
            if self.args.station_handoff:
                self.planner.scan_observation_pose = tuple(capture_pose)
            self.tracks = self.tracker.update(capture_pose,
                (self.scan.ranges, self.scan.angle_min, self.scan.angle_increment), scan_time)
            for obj in self.registered_objects:
                if math.dist(capture_pose[:2],obj['position']) <= 5.:
                    self.tracks.append(Track(obj['evidence_id'],obj['position'],(0.,0.),
                        obj['radius'],scan_time,0.,observations=0,
                        source=obj['source'],registered_box=obj['registered_box']))
            self.scan_stamp = scan_time
        executed = self.mode_payload.get('executed', {})
        speed = math.hypot(executed.get('vx', 0), executed.get('vy', 0))
        if self.args.station_handoff:
            self.harness.refresh_staging_from_scan(self.pose,
                self.planner._scan_points_world(self.pose,
                    (self.scan.ranges,self.scan.angle_min,self.scan.angle_increment)),now)
        previous_stage = self.harness.stage
        if (self.args.stable_modes and self.harness.selected_behavior.startswith(('PASS', 'YIELD'))
                and not self.harness.evidence_ids and self.harness.bound_behavior_goal is not None
                and math.dist(self.pose[:2], self.harness.bound_behavior_goal) < .28):
            self.harness._event('LOCAL_RECOVERY_VERIFIED', now,
                measured_pose=list(self.pose), bound_goal=list(self.harness.bound_behavior_goal))
            self.harness.interaction = self.harness.selected_behavior = 'CONTINUE'
            self.harness.bound_behavior_goal = None
            self.harness.query_epoch += 1
        if self.args.long_task or self.args.local_pose_goal:
            speed = max(speed, abs(executed.get('wz', 0)))
        self.harness.feedback(self.pose, now, self.tracks, speed)
        task_key = self.semantic_task_key()
        if task_key != self.task_key:
            self.task_epoch += 1
            self.task_key = task_key
            if getattr(self.args, 'proposal_interface', 'behavior') == 'task_condition':
                self.active_task_condition = None
                self.active_condition_task_key = None
                self.harness.evidence_ids = []
                self.planner.active_motion_condition = None
                self.condition_refresh_needed = True
        if (getattr(self.args, 'proposal_interface', 'behavior') == 'task_condition' and
                self.active_task_condition):
            live_ids = {t.evidence_id for t in self.tracks if now-t.stamp <= .65}
            bound_ids = set(self.active_task_condition.get('relevant_object_ids', ()))
            if bound_ids-live_ids:
                self.record_event({'event': 'TASK_CONDITION_EVIDENCE_EXPIRED',
                    'sim_time': now, 'expired_object_ids': sorted(bound_ids-live_ids),
                    'anchor_id': self.active_task_condition.get('anchor_id')})
                # Retain instruction-level preferences when object evidence expires.
                remaining = sorted(bound_ids & live_ids)
                self.active_task_condition = dict(self.active_task_condition,
                    relevant_object_ids=remaining,
                    object_labels={k: v for k, v in self.active_task_condition.get('object_labels', {}).items()
                                   if k in live_ids})
                self.harness.evidence_ids = remaining
                self.planner.active_motion_condition = self.active_task_condition
                self.condition_refresh_needed = True
        if self.args.long_task:
            from std_msgs.msg import String
            self.task_state_pub.publish(String(data=json.dumps({
                'stage': self.harness.stage, **self.harness.context(self.pose, self.tracks)['long_task']}, ensure_ascii=False)))
            arrival = self.graph.nodes['H'+self.harness.portal[1:]]['pose']
            self.goal, self.goal_yaw = tuple(arrival[:2]), arrival[2]
            if self.harness.stage == 'SERVICE_CONFIRM':
                self.publish((0., 0., 0.))
                self.last_action = (0., 0., 0.)
                return
        if self.harness.stage == "DONE":
            self.finish('ARRIVED')
            return
        if self.motion_started and now-self.started > self.args.max_sim_time:
            self.finish('TIMEOUT')
            return
        if now-self.last_plan < .12:
            # Refresh the held command so the actuator watchdog reflects
            # controller liveness rather than the timer's planning interval.
            self.publish(self.last_action)
            return
        dt = now-self.last_plan if self.last_plan > 0 else .12
        self.last_plan = now
        self.observation_log.write(json.dumps({'sim_time': now, 'pose': self.pose,
            'scan_stamp': scan_time, 'scan': {'ranges': [float(v) if math.isfinite(v) else None for v in self.scan.ranges],
                'angle_min': self.scan.angle_min, 'angle_increment': self.scan.angle_increment},
            'tracks': [t.to_dict() for t in self.tracks], 'actuation': self.mode_payload,
            'task_stage': self.harness.stage, 'goal': self.harness.subgoal(),
            'rng_state': self.planner.rng.getstate()}, separators=(',', ':'))+'\n')
        self.observation_log.flush()
        if self.args.policy == 'direct_vlm':
            self.tick_direct_vlm(now)
            return
        # The active trajectory and the full bank use the same current reference.
        self.refresh_route_reference(now)
        encounter = self.relevant_encounter()
        query_needed = self.args.policy in ('grounded', 'visual') and self.pending is None and (
            not self.motion_started or self.last_query_task_epoch != self.task_epoch or
            self.condition_refresh_needed or
            ((encounter or self.args.policy == 'visual') and now-self.last_query > 5))
        # During HTTP inference only the active behavior needs continuous
        # optimization. Rebuild all candidates when a result is ready so its
        # acceptance still uses current feasibility, not the request snapshot.
        response_ready = self.pending is not None and self.pending.done()
        if (response_ready and self.args.stable_modes and
                self.mode_payload.get('steering_transition_stop', False)):
            if not self.proposal_deferred:
                self.record_event({'event': 'PROPOSAL_DEFERRED_FOR_ALIGNMENT',
                    'sim_time': now, 'query': self.actual_queries,
                    'request_epoch': self.pending_epoch,
                    'current_epoch': self.harness.query_epoch})
                self.proposal_deferred = True
            response_ready = False
        if (query_needed or response_ready or self.args.proposal_interface == 'task_condition'
                or self.args.policy not in ('grounded', 'visual') or not self.bank):
            self.refresh_bank(now)
        if query_needed:
            trigger = 'initial_task' if not self.motion_started else ('task_event' if self.last_query_task_epoch != self.task_epoch else 'dynamic_encounter')
            self.start_semantic_query(now, trigger)
            self.condition_refresh_needed = False
        if response_ready:
            accepted_before = self.accepted_proposals
            api_counted = False
            try:
                response = self.pending.result()
                if self.args.proposal_selector == 'vlm':
                    self.vlm_http_attempts += response.get('http_requests',1)
                    api_counted = True
                (self.path/f'proposal_{self.actual_queries:03d}_response.json').write_text(
                    json.dumps(response, ensure_ascii=False, indent=2))
                if self.pending_task_key != self.semantic_task_key():
                    self.record_event({'event': 'PROPOSAL_REJECTED', 'sim_time': now,
                        'proposal': response['proposal'], 'reasons': ['semantic_task_changed']})
                    self.rejections += 1
                    self.query_epoch = -1
                elif self.args.proposal_interface == 'task_condition':
                    proposal = response['proposal']
                    live_ids = {t.evidence_id for t in self.tracks if now-t.stamp <= .65}
                    if any(e not in live_ids for e in proposal['relevant_object_ids']):
                        self.rejections += 1
                        self.query_epoch = -1
                        self.condition_refresh_needed = True
                        self.record_event({'event': 'TASK_CONDITION_REJECTED', 'sim_time': now,
                            'proposal': proposal, 'reasons': ['object_evidence_not_live_at_binding']})
                    else:
                        condition = {
                            'anchor_id': proposal['anchor_id'],
                            'relevant_object_ids': list(proposal['relevant_object_ids']),
                            'heading_policy': proposal['heading_policy'],
                            'preferred_clearance_m': float(proposal['preferred_clearance_m']),
                            'motion_preference': proposal['motion_preference'],
                            'heading_scale_rad': math.radians(self.args.task_heading_scale_deg),
                            'clearance_scale_m': self.args.task_clearance_scale_m,
                            'object_labels': dict(proposal['object_labels']),
                            'heading': (self.active_task_condition['heading']
                                if self.active_task_condition and
                                   self.active_task_condition.get('heading_policy') == proposal['heading_policy'] and
                                   self.active_condition_task_key == self.semantic_task_key()
                                else self.pending_observation_heading),
                        }
                        self.active_task_condition = condition
                        self.active_condition_task_key = self.semantic_task_key()
                        self.planner.active_motion_condition = condition
                        self.harness.evidence_ids = list(proposal['relevant_object_ids'])
                        self.accepted_proposals += 1
                        if not self.args.freeze_object_labels:
                            self.tracker.label(proposal['object_labels'])
                        self.record_event({'event': 'TASK_CONDITION_BOUND', 'sim_time': now,
                            'proposal': proposal, 'task_condition': condition,
                            'planner_owns_motion_family_and_switch_time': True})
                elif self.args.policy == 'visual':
                    behavior = response['proposal']['behavior_id']
                    selected = next((c for c in self.bank if c['behavior_id'] == behavior and c['feasible']), None)
                    if selected is None:
                        self.rejections += 1
                        self.query_epoch = -1
                        self.record_event({'event': 'PROPOSAL_REJECTED', 'sim_time': now,
                            'proposal': response['proposal'], 'reasons': ['current_rollout_infeasible']})
                    else:
                        self.harness.selected_behavior = self.harness.interaction = behavior
                        self.harness.bound_behavior_goal = tuple(selected['goal'])
                        self.harness.evidence_ids = response['proposal']['evidence_ids']
                        self.accepted_proposals += 1
                        
                        if not self.args.freeze_object_labels:
                            self.tracker.label(response['proposal']['object_labels'])
                        self.record_event({'event': 'VISUAL_SELECTOR_ACCEPTED', 'sim_time': now,
                            'proposal': response['proposal'], 'bound_goal': selected['goal']})
                elif self.harness.bind(response['proposal'], self.bank, self.tracks, now, self.pending_stage):
                    self.accepted_proposals += 1
                    if not self.args.freeze_object_labels:
                        self.tracker.label(response['proposal']['object_labels'])
                else:
                    self.rejections += 1
                    self.query_epoch = -1
            except Exception as exc:
                if self.args.proposal_selector == 'vlm' and not api_counted:
                    self.vlm_http_attempts += getattr(exc,'http_requests',0)
                (self.path/f'proposal_{self.actual_queries:03d}_error.json').write_text(
                    json.dumps({'error':str(exc),'attempts':getattr(exc,'attempts',[]),'http_requests':getattr(exc,'http_requests',0)},ensure_ascii=False,indent=2))
                self.vlm_errors += 1
                self.record_event({"event": "SEMANTIC_REQUEST_FAILED", "sim_time": now,
                    "query": self.actual_queries, "error": str(exc)})
                if not self.motion_started:
                    self.finish('VLM_REQUEST_FAILED')
                    return
                self.query_epoch = -1
            finally:
                accepted = self.accepted_proposals > accepted_before
                if accepted:
                    self.runtime_timeline.emit(event='semantic_condition_bound',
                        task_epoch=self.task_epoch, object_id=list(self.harness.evidence_ids),
                        behavior_id=self.harness.selected_behavior,
                        task_condition=self.active_task_condition,
                        state_version=self.state_version, trajectory_version=self.trajectory_version)
                    self.active_proposal_query = self.actual_queries
                    self.active_proposal_request_wall = self.pending_started_wall
                self.record_event({'event': 'SEMANTIC_RESPONSE_HANDLED',
                    'sim_time': now, 'query': self.actual_queries, 'accepted': accepted,
                    'request_to_handling_wall_s': time.monotonic()-self.pending_started_wall,
                    'deferred_for_alignment': self.proposal_deferred,
                    'request_epoch': self.pending_epoch,
                    'current_epoch': self.harness.query_epoch})
                self.pending = None
        if not self.motion_started:
            self.publish((0., 0., 0.))
            self.last_action = (0., 0., 0.)
            if self.args.policy not in ('grounded', 'visual') or self.accepted_proposals:
                self.activate_motion(now)
            return
        if self.args.policy in ('grounded', 'visual') and self.args.proposal_interface != 'task_condition':
            chosen = self.call_planner('behavior_plan', self.pose, self.harness.subgoal()[0],
                (self.scan.ranges, self.scan.angle_min, self.scan.angle_increment), self.tracks, now,
                self.harness.selected_behavior, bound_goal=self.harness.bound_behavior_goal,
                **self.current_kwargs())
            desired_yaw = self.harness.subgoal()[1] if chosen['behavior_id'] == 'CONTINUE' else self.pose[2]
            desired_yaw = chosen.get('goal_yaw', desired_yaw)
            if not self.graph.supports_pose((*chosen['goal'], self.pose[2] if desired_yaw is None else desired_yaw)):
                chosen['feasible'] = False
            if not chosen['feasible']:
                self.refresh_bank(now)
                valid = [c for c in self.bank if c['feasible']]
                chosen = min(valid, key=lambda c: c['plan']['predicted_score']) if valid else chosen
                self.overrides += 1
                if self.overrides % 10 == 1:
                    self.record_event({"event": "CURRENT_MOTION_GATE", "sim_time": now,
                        "requested_behavior": self.harness.selected_behavior,
                        "executed_behavior": chosen['behavior_id'], "feasible": chosen['feasible']})
                if self.args.verified_recovery and self.args.policy == 'grounded' and chosen['feasible']:
                    self.harness.provisional(chosen, self.tracks, self.pose, now)
                elif self.args.stable_modes and not self.args.verified_recovery and chosen['feasible']:
                    previous = self.harness.selected_behavior
                    replacement = chosen['behavior_id']
                    if replacement != previous:
                        # A rejected semantic action must update the shared
                        # state; otherwise every tick retries the same invalid
                        # action and destroys the fallback's steering progress.
                        self.harness._event('EXECUTION_FEASIBILITY_CHANGED', now,
                            previous_behavior=previous, provisional_behavior=replacement,
                            reason='active behavior no longer has a supported feasible rollout')
                        self.harness.selected_behavior = replacement
                        self.harness.interaction = replacement
                        self.harness.bound_behavior_goal = tuple(chosen['goal'])
                        self.harness.evidence_ids = [t.evidence_id for t in self.tracks]
                        self.harness.query_epoch += 1
        else:
            valid = [c for c in self.bank if c['feasible']]
            chosen = min(valid, key=lambda c: c['plan']['predicted_score']) if valid else self.bank[0]
        plan = chosen['plan']
        # Latest sensors can advance while the independent planner computes.
        # Ordinary state-version changes are expected; bounded drift is checked.
        angles = tuple(self.mode_payload.get('wheel_steering_rad', ()))
        position_drift = math.dist(planning_pose[:2], self.pose[:2])
        heading_drift = abs(wrap(planning_pose[2]-self.pose[2]))
        steering_drift = max((abs(a-b) for a,b in zip(planning_angles,angles)), default=0.)
        trajectory_valid = (time.monotonic()-planning_wall <= 1.5 and
            position_drift <= .18 and heading_drift <= .35)
        self.trajectory_version += 1
        self.runtime_timeline.emit(event='trajectory_validation',
            trajectory_version=self.trajectory_version, input_state_version=planning_state_version,
            current_state_version=self.state_version, valid=trajectory_valid,
            position_drift_m=position_drift, heading_drift_rad=heading_drift,
            steering_drift_rad=steering_drift)
        if self.args.execution_architecture == 'process' and not trajectory_valid:
            chosen['feasible'] = False
        self.command_lease_until = time.monotonic()+.7 if chosen['feasible'] else 0.
        action = tuple(plan['action']) if chosen['feasible'] else (0., 0., 0.)
        # Stop before verifying terminal arrival; completion requires a measured
        # stationary interval, not a single pose sample during motion.
        dist = math.dist(self.pose[:2], self.goal)
        yaw_error = abs(wrap(self.pose[2]-self.goal_yaw))
        if self.harness.stage == 'ARRIVE' and dist < .30 and yaw_error < math.radians(16):
            action = (0., 0., 0.)
        if self.args.station_handoff:
            if self.harness.stage == 'HANDOFF_SELECT':
                action = (0., 0., 0.)
            elif self.harness.stage == 'HANDOFF_APPROACH' and dist <= .09 and yaw_error <= math.radians(10):
                action = (0., 0., 0.)
        if ((self.args.long_task or self.args.scenario in ('collection_delivery', 'collection_return_delivery'))
                and self.harness.at_service_pose(self.pose)):
            action = (0., 0., 0.)
        self.publish(action)
        if self.args.stable_modes:
            self.planner.note_execution(plan, now, action)
        self.last_action = action
        if math.hypot(action[0], action[1]) < .01:
            self.wait_time += min(dt, .5)
        compact = {k: v for k, v in plan.items() if k not in
            ('candidate_population', 'candidate_summary', 'selected_trajectory', 'predicted_executed_controls',
             'hybrid_schedule_candidates')}
        self.trace.write(json.dumps({"timestamp": now, "elapsed": now-self.started, "pose": self.pose,
            "task_stage": self.harness.stage, "interaction": self.harness.interaction,
            "anchor_id": self.harness.anchor_id, "local_subgoal": self.harness.subgoal()[0],
            "behavior_id": chosen['behavior_id'], "semantic_behavior": self.harness.selected_behavior,
            "motion_condition": getattr(self.harness, 'motion_condition', None),
            "tracks": [t.to_dict() for t in self.tracks], "encounter_ids": encounter,
            "command": action, "actuation": self.mode_payload, "plan": compact})+'\n')
        self.trace.flush()

    def finish(self, status):
        if self.done:
            return
        self.publish((0., 0., 0.))
        self.done = True
        self.trace.flush()
        rows = [json.loads(s) for s in (self.path/'trace.jsonl').read_text().splitlines()]
        modes = [r.get('actuation', {}).get('mode', 'UNKNOWN') for r in rows]
        result = {"status": status, "policy": self.args.policy, "scenario": self.args.scenario,
            "seed": self.args.seed, "task_success": status == 'ARRIVED' and self.harness.stage == 'DONE',
            "task_success_criterion": ("verified_room_arrivals_and_ordered_service_acknowledgements; not physical handling"
                                       if self.args.long_task else "verified_delivery_navigation_only; see independent interaction audit"),
            "task_stage": self.harness.stage, "verified_stage_events": self.harness.stage_events,
            "elapsed_sim_s": self.now()-self.started if self.motion_started else 0,
            "path_length_m": sum(math.dist(a['pose'][:2], b['pose'][:2]) for a, b in zip(rows, rows[1:])),
            "final_pose": self.pose, "goal": self.goal,
            "position_error_m": math.dist(self.pose[:2], self.goal) if self.pose else None,
            "heading_error_deg": math.degrees(abs(wrap(self.pose[2]-self.goal_yaw))) if self.pose else None,
            "semantic_queries": self.actual_queries, "accepted_proposals": self.accepted_proposals,
            "rejected_proposals": self.rejections, "semantic_errors": self.vlm_errors,
            "motion_gate_overrides": self.overrides, "wait_time_s": self.wait_time,
            "mode_counts": {m: modes.count(m) for m in sorted(set(modes))},
            "mode_switches": sum(a != b for a, b in zip(modes, modes[1:])),
            "frame_count": self.frame_index, "overview_frame_count": self.overview_index,
            "mean_bank_planning_ms": sum(self.plan_times)/len(self.plan_times) if self.plan_times else None,
            "dynamic_entity_truth_used_by_policy": False}
        result['planner'] = self.args.planner
        result['proposal_selector'] = self.args.proposal_selector
        result['proposal_interface'] = self.args.proposal_interface
        result['active_task_condition'] = self.active_task_condition
        result['direct_vlm_commands'] = self.direct_command_count
        result['mppi_used_for_control'] = self.args.policy != 'direct_vlm'
        result['freeze_object_labels'] = self.args.freeze_object_labels
        result['vlm_api_requests'] = self.vlm_http_attempts
        result['continuation_lookahead'] = self.args.continuation_lookahead
        result['footprint_map_recovery'] = self.args.map_recovery
        (self.path/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps(result, ensure_ascii=False), flush=True)


def clean_environment(args):
    base = {k: os.environ[k] for k in ('HOME', 'USER', 'LOGNAME') if k in os.environ}
    base.update(PATH='/usr/local/bin:/usr/bin:/bin', LANG='C.UTF-8')
    command = f"source /opt/ros/humble/setup.bash && source {ROOT}/install/local_setup.bash && env -0"
    raw = subprocess.check_output(['bash', '--noprofile', '--norc', '-c', command], env=base)
    env = dict((k.decode(), v.decode()) for k, v in
        (item.split(b'=', 1) for item in raw.split(b'\0') if b'=' in item))
    for key in ('DISPLAY', 'XAUTHORITY', 'DEEPSEEK_API_KEY', 'DEEPSEEK_VISION_MODEL', 'DEEPSEEK_BASE_URL',
                'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY', 'http_proxy', 'https_proxy', 'no_proxy'):
        if key in os.environ:
            env[key] = os.environ[key]
    env.update(ROS_DOMAIN_ID=str(args.ros_domain_id), RMW_IMPLEMENTATION='rmw_fastrtps_cpp',
        IGN_PARTITION=args.ign_partition, IGN_IP='127.0.0.1',
        IGN_GAZEBO_RESOURCE_PATH=str(PKG/'models/lab_assets'), ROS_LOG_DIR=str(Path(args.output)/'ros_logs'))
    matrix_version = re.fullmatch(r'matrix_v(\d+)', Path(args.output).resolve().parent.name)
    if matrix_version:
        version = int(matrix_version.group(1))
        # Separate concurrently finishing development matrices as well as
        # trials. Different Ignition partitions alone do not isolate ROS DDS.
        env['ROS_DOMAIN_ID'] = str(30+(version % 6)*35+(args.ros_domain_id-120))
        env['IGN_PARTITION'] = f'{args.ign_partition}_v{version}'
    return env


def orchestrate(args):
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    fixture_script = HERE/'grounded_delivery_scene.py'
    if args.scenario in RESEARCH_SCENARIOS:
        from motion_planning_research_scene import create_world as create_research_world
        metadata = create_research_world(root/'world.sdf', args.scenario)
        fixture_script = HERE/'motion_planning_research_scene.py'
    elif args.scenario == 'continuous_delivery':
        from continuous_delivery_scene import create_world as create_continuous_world
        metadata = create_continuous_world(root/'world.sdf', args.scenario)
        fixture_script = HERE/'continuous_delivery_scene.py'
    else:
        metadata = create_world(root/'world.sdf', args.scenario)
    tree = ET.parse(root/'world.sdf')
    tree.getroot().find('world/physics/real_time_factor').text = str(args.real_time_factor)
    tree.write(root/'world.sdf', encoding='unicode', xml_declaration=True)
    metadata['real_time_factor'] = args.real_time_factor
    if args.long_task:
        from rangermini_dynamic_semantic.long_service_task import compile_task, validate_task, prepare_service_layout
        metadata['multiroom_layout'] = prepare_service_layout(root/'world.sdf')
        graph = SpatialGraph(root/'world.sdf')
        plan = (dict(validate_task(json.loads(Path(args.task_plan).read_text()), graph),
                     source=('archived_language_model_plan' if json.loads(Path(args.task_plan).read_text()).get('source') in
                             ('real_language_model','archived_language_model_plan') else 'supplied_plan'))
                if args.task_plan else compile_task(args.instruction, graph, root/'task_parser_response.json'))
        args.task_plan = str(root/'long_task_plan.json')
        Path(args.task_plan).write_text(json.dumps(plan, ensure_ascii=False, indent=2))
        metadata['long_task'] = plan
        metadata['service_confirmation'] = args.service_confirmation
        from rangermini_dynamic_semantic.long_service_task import LongServiceTaskHarness
        preflight = LongServiceTaskHarness(graph, args.instruction, plan, confirmation=args.service_confirmation)
        metadata['service_poses'] = preflight.service_poses
    if args.station_handoff:
        from rangermini_dynamic_semantic.station_handoff import station_conditions
        station = station_conditions(SpatialGraph(root/'world.sdf'),root/'world.sdf')
        metadata['station_registration'] = dict(version='office619_station_v2',
            target_position=next(iter(station.values()))['target_pose'][:2],
            heading_tolerance_deg=10,position_tolerance_m=.10,
            approach_region=dict(last_entry_radius_m=.75),
            source='registered desk and receiving pose; fixed before selector execution')
    source_files = [PKG/'rangermini_dynamic_semantic/long_service_task.py', PKG/'rangermini_dynamic_semantic/upstream_mppi.py', PKG/'rangermini_dynamic_semantic/nav2_omni_mppi.py', PKG/'rangermini_dynamic_semantic/multiwaypoint_delivery.py', Path(__file__).resolve(), HERE/'grounded_delivery_scene.py', HERE/'freeze_static_global_reference.py',
        PKG/'rangermini_dynamic_semantic/grounded_navigation_harness.py',
        PKG/'rangermini_dynamic_semantic/delivery_execution_recovery.py',
        PKG/'rangermini_dynamic_semantic/station_handoff.py',
        PKG/'rangermini_dynamic_semantic/interaction_mppi.py',
        PKG/'rangermini_dynamic_semantic/stable_interaction_mppi.py',
        PKG/'rangermini_dynamic_semantic/grounded_visual_proposal.py',
        PKG/'rangermini_dynamic_semantic/visual_behavior_selector.py',
        PKG/'rangermini_dynamic_semantic/observation_mppi.py',
        PKG/'rangermini_dynamic_semantic/hybrid_sequence_mppi.py',
        PKG/'rangermini_dynamic_semantic/local_pose_navigation.py',
        PKG/'rangermini_dynamic_semantic/continuation_evaluator.py',
        PKG/'rangermini_dynamic_semantic/footprint_route_guide.py',
        PKG/'rangermini_dynamic_semantic/ranger_4ws_kinematics.py']
    source_files.extend([PKG/'rangermini_dynamic_semantic/planner_process.py',
                         PKG/'rangermini_dynamic_semantic/native_kernels.py',
                         PKG/'rangermini_dynamic_semantic/runtime_timeline.py',
                         PKG/'rangermini_dynamic_semantic/ranger_motion_adapter.py',
                         PKG/'rangermini_dynamic_semantic/command_semantics_mppi.py',
                         PKG/'rangermini_dynamic_semantic/execution_freshness.py'])
    native_core = ROOT/'integration/mppi_native_core'
    source_files.extend(native_core/name for name in ('core.cpp', 'CMakeLists.txt', 'build.py'))
    native_binary = native_core/'build/libranger_mppi_native.so'
    if not native_binary.is_file():
        raise RuntimeError('Build native MPPI kernels: python3 integration/mppi_native_core/build.py')
    metadata['numeric_backend'] = dict(
        wheel_state_prediction='C++17', scan_footprint_queries='C++17',
        sampling_and_sequence_search='Python',
        library_sha256=hashlib.sha256(native_binary.read_bytes()).hexdigest(),
        provenance=json.loads((native_core/'provenance.json').read_text()))
    if args.motion_executor == 'cpp':
        motion_binary=ROOT/'install/ranger_sim_motion_cpp/lib/ranger_sim_motion_cpp/ranger_motion_adapter_cpp'
        if not motion_binary.is_file():
            raise RuntimeError('Build ranger_sim_motion_cpp before selecting the C++ motion executor')
        motion_source=ROOT/'src/ranger_sim_motion_cpp/src/motion_adapter.cpp'
        source_files.append(motion_source)
        metadata['motion_executor']=dict(implementation='rclcpp',
            binary_sha256=hashlib.sha256(motion_binary.read_bytes()).hexdigest(),
            shared_wheel_model='integration/mppi_native_core/core.cpp')
    else:
        metadata['motion_executor']=dict(implementation='rclpy',shared_wheel_model='integration/mppi_native_core/core.cpp')
    if fixture_script.name != 'grounded_delivery_scene.py':
        source_files.append(fixture_script)
    if fixture_script.name == 'motion_planning_research_scene.py':
        source_files.append(HERE/'continuous_delivery_scene.py')
    metadata['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    metadata['adapter_parameters'] = adapter_parameters(args)
    (root/'source_snapshot').mkdir()
    for path in source_files:
        (root/'source_snapshot'/path.name).write_bytes(path.read_bytes())
    (root/'source_snapshot/libranger_mppi_native.so').write_bytes(native_binary.read_bytes())
    if args.motion_executor == 'cpp':
        (root/'source_snapshot/ranger_motion_adapter_cpp').write_bytes(motion_binary.read_bytes())
    (root/'scene_manifest.json').write_text(json.dumps(metadata, indent=2))
    if args.planner == 'nav2-omni':
        import shutil
        adapter = ROOT/'integration/nav2_mppi_adapter'
        shutil.copytree(adapter, root/'upstream_adapter', ignore=shutil.ignore_patterns('build'), dirs_exist_ok=True)
        metadata['upstream_solver'] = dict(name='Nav2 MPPI', version='1.1.20', motion_model='Omni',
            optimizer_modified=False, lateral_sampling=True,
            binary_sha256=hashlib.sha256((adapter/'build/libranger_nav2_mppi.so').read_bytes()).hexdigest(),
            installed_optimizer_sha256=hashlib.sha256(Path('/opt/ros/humble/lib/libmppi_controller.so').read_bytes()).hexdigest())
    elif args.planner not in ('conditioned','ranger-mppi'):
        import shutil
        adapter = ROOT/'integration/mppi_h_adapter'
        shutil.copytree(adapter, root/'upstream_adapter', ignore=shutil.ignore_patterns('build'), dirs_exist_ok=True)
        metadata['upstream_solver'] = json.loads((adapter/'provenance.json').read_text())
    children = []
    env = clean_environment(args)
    metadata['transport_isolation'] = {k: env[k] for k in ('ROS_DOMAIN_ID', 'IGN_PARTITION')}
    (root/'scene_manifest.json').write_text(json.dumps(metadata, indent=2))
    def spawn(command, name):
        log = (root/(name+'.log')).open('w')
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children.append((proc, log))
        return proc
    try:
        spawn(['python3', str(HERE/'observe_runtime_timeline.py'), '--output',
               str(root/'independent_runtime_timeline.jsonl')], 'runtime_observer')
        spawn(['ign', 'gazebo', '-s', '-r', str(root/'world.sdf')], 'gazebo')
        time.sleep(2)
        spawn(['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
            '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
            '/model/rangermini_2_0/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist',
            '/model/rangermini_2_0/pose@geometry_msgs/msg/PoseStamped[ignition.msgs.Pose',
            '/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan',
            '/camera/image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/camera/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/scene/navigation@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/world/ranger_lab_corridor/set_pose@ros_gz_interfaces/srv/SetEntityPose',
            '--ros-args', '-r', '/model/rangermini_2_0/cmd_vel:=/cmd_vel_gazebo'], 'bridge')
        spawn(['ros2', 'run', 'rangermini_dynamic_semantic', 'gazebo_odom_adapter'], 'odom')
        adapter_entry = (['ros2','run','ranger_sim_motion_cpp','ranger_motion_adapter_cpp']
                         if args.motion_executor == 'cpp' else
                         ['ros2','run','rangermini_dynamic_semantic','ranger_motion_adapter'])
        adapter_command = adapter_entry + ['--ros-args',
            '-p', 'input_topic:=/cmd_vel_safe', '-p', 'output_topic:=/cmd_vel_gazebo',
            '-p', 'command_timeout:=0.7', '-p', 'use_sim_time:=true']
        for key, value in adapter_parameters(args).items():
            adapter_command.extend(['-p', f'{key}:={value}'])
        spawn(adapter_command, 'adapter')
        spawn(['python3', str(fixture_script), '--scenario', args.scenario,
            '--output', str(root/'fixture_truth.jsonl')], 'fixture')
        cmd = ['python3', str(Path(__file__).resolve()), '--controller', '--world', str(root/'world.sdf'),
            '--output', str(root/'episode'), '--scenario', args.scenario, '--policy', args.policy,
            '--seed', str(args.seed), '--samples', str(args.samples), '--max-sim-time', str(args.max_sim_time),
            '--max-wall-time', str(args.max_wall_time), '--real-time-factor', str(args.real_time_factor),
            '--instruction', args.instruction, '--proposal-selector', args.proposal_selector,
            '--direct-query-period', str(args.direct_query_period),
            '--execution-architecture', args.execution_architecture,
            '--motion-executor',args.motion_executor,
            '--planner', args.planner, '--upstream-samples', str(args.upstream_samples)]
        cmd.extend(['--proposal-interface', args.proposal_interface])
        if args.spatial_conditions_only:
            cmd.append('--spatial-conditions-only')
        cmd.extend(['--task-heading-scale-deg',str(args.task_heading_scale_deg),
                    '--task-clearance-scale-m',str(args.task_clearance_scale_m)])
        if args.static_global_map:
            cmd.append('--static-global-map')
        if args.registered_object_evidence:
            cmd.append('--registered-object-evidence')
        if args.unrestricted_task_motion:
            cmd.append('--unrestricted-task-motion')
        if args.obstacle_seeded_sampling:
            cmd.append('--obstacle-seeded-sampling')
        if args.retain_corridor_heading:
            cmd.append('--retain-corridor-heading')
        if args.long_task:
            cmd.extend(['--long-task', '--task-plan', args.task_plan,
                        '--service-confirmation', args.service_confirmation])
        if args.freeze_object_labels:
            cmd.append('--freeze-object-labels')
        if args.stable_modes:
            cmd.append('--stable-modes')
        if args.hybrid_sequences:
            cmd.extend(['--hybrid-sequences','--hybrid-max-switches',str(args.hybrid_max_switches)])
        if args.hybrid_fixed_timing:
            cmd.append('--hybrid-fixed-timing')
        if args.local_pose_goal:
            cmd.extend(['--local-pose-goal',*map(str,args.local_pose_goal)])
        if args.frozen_global_reference:
            cmd.extend(['--frozen-global-reference',str(Path(args.frozen_global_reference).resolve())])
        if args.verified_recovery:
            cmd.append('--verified-recovery')
        if args.station_handoff:
            cmd.append('--station-handoff')
        if args.station_terminal_pose_only:
            cmd.append('--station-terminal-pose-only')
        if args.station_proposal_replay:
            source = Path(args.station_proposal_replay).resolve()
            (root/'recorded_station_proposal.json').write_bytes(source.read_bytes())
            cmd.extend(['--station-proposal-replay',str(root/'recorded_station_proposal.json')])
        if args.task_condition_replay:
            for flag, source_name, target_name in [
                ('--task-condition-replay', args.task_condition_replay, 'recorded_task_condition.json'),
                ('--task-condition-default-replay', args.task_condition_default_replay, 'recorded_task_default.json')]:
                (root/target_name).write_bytes(Path(source_name).read_bytes())
                cmd.extend([flag, str(root/target_name)])
        if args.continuation_lookahead:
            cmd.append('--continuation-lookahead')
        if args.map_recovery:
            cmd.append('--map-recovery')
        if args.ideal_steering_rollout:
            cmd.append('--ideal-steering-rollout')
        if args.unprojected_behavior_guides:
            cmd.append('--unprojected-behavior-guides')
        if args.scan_drop_window:
            cmd.extend(['--scan-drop-window', *map(str, args.scan_drop_window)])
        process = spawn(cmd, 'controller')
        process.wait(timeout=args.max_wall_time)
        if not (root/'episode/result.json').exists():
            raise RuntimeError('controller ended without terminal result; see controller.log')
        print((root/'episode/result.json').read_text(), flush=True)
    finally:
        demo.stop(children)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--planner', choices=('conditioned','ranger-mppi','nav2-omni','mppi-h','mppi-3d','mppi-4d'), default='conditioned')
    parser.add_argument('--retain-corridor-heading', action='store_true')
    parser.add_argument('--upstream-samples', type=int, default=3000)
    parser.add_argument('--policy', choices=['geometric', 'predictive', 'visual', 'grounded', 'direct_vlm'], default='grounded')
    parser.add_argument('--direct-query-period', type=float, default=2.5,
                        help='Simulation seconds between direct VLM velocity requests')
    parser.add_argument('--proposal-selector', choices=('vlm','cost'), default='vlm',
                        help='Same grounding/execution; change only proposal selection')
    parser.add_argument('--obstacle-seeded-sampling', action='store_true',
                        help='shared symmetric local sampling means from current obstacle tracks; same rollout cap')
    parser.add_argument('--proposal-interface', choices=('behavior','task_condition'), default='behavior',
                        help='task_condition lets the VLM ground task constraints; hybrid MPPI selects motion family and switching time')
    parser.add_argument('--freeze-object-labels', action='store_true',
                        help='Hold tracker labels fixed in matched selector comparisons')
    parser.add_argument('--spatial-conditions-only', action='store_true',
                        help='VLM grounds objects, clearance and heading only; no model-selected motion family')
    parser.add_argument('--task-heading-scale-deg', type=float, default=math.degrees(1.),
                        help='Normalize task heading preference; historical default is one radian')
    parser.add_argument('--task-clearance-scale-m', type=float, default=1.,
                        help='Normalize task clearance preference; historical default is one meter')
    parser.add_argument('--registered-object-evidence', action='store_true',
                        help='Expose registered static equipment anchors separately from live lidar tracks')
    parser.add_argument('--unrestricted-task-motion', action='store_true',
                        help='Task stages specify pose requirements only; all native motion families remain eligible')
    parser.add_argument('--scenario', choices=(*SCENARIOS, 'continuous_delivery', *RESEARCH_SCENARIOS), default='headon')
    parser.add_argument('--instruction', default=INSTRUCTION)
    parser.add_argument('--long-task', action='store_true', help='Parse one instruction into feedback-driven multi-room services')
    parser.add_argument('--task-plan', help='Explicit validated task JSON; bypass parser and log as supplied plan')
    parser.add_argument('--service-confirmation', choices=('external', 'simulation'), default='external')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--samples', type=int, default=6)
    parser.add_argument('--max-sim-time', type=float, default=150)
    parser.add_argument('--max-wall-time', type=float, default=420)
    parser.add_argument('--real-time-factor', type=float, default=.5, help='Requested Gazebo simulation/wall-time ratio; not an achieved rate guarantee')
    parser.add_argument('--ros-domain-id', type=int, default=108)
    parser.add_argument('--ign-partition', default='ranger_grounded_delivery')
    parser.add_argument('--execution-architecture', choices=('legacy','process'), default='process')
    parser.add_argument('--motion-executor',choices=('python','cpp'),default='python',
                        help='Simulation motion publisher; cpp uses rclcpp and the shared native wheel model')
    parser.add_argument('--controller', action='store_true')
    parser.add_argument('--stable-modes', action='store_true')
    parser.add_argument('--hybrid-sequences', action='store_true',
                        help='Opt-in bounded mode-structure and switching-step search; retains primitive candidates')
    parser.add_argument('--local-pose-goal',type=float,nargs=3,metavar=('X','Y','YAW'),
                        help='Single pose task for local-planner comparison; no preset portal stages')
    parser.add_argument('--hybrid-max-switches',type=int,choices=(0,1,2),default=2)
    parser.add_argument('--hybrid-fixed-timing',action='store_true',
                        help='Internal ablation: equal phase durations, same structures and rollout cap')
    parser.add_argument('--verified-recovery', action='store_true')
    parser.add_argument('--station-handoff', action='store_true',
                        help='Opt-in desk-referenced heading-retaining handoff task; requires office_handoff and verified recovery.')
    parser.add_argument('--station-terminal-pose-only', action='store_true',
                        help='Internal terminal-pose interface ablation of our shared 4WS MPPI; not a reproduced external algorithm.')
    parser.add_argument('--station-proposal-replay',
                        help='Reuse an original real station VLM output for matched control-mechanism pairs; logged as replay, not a new API call.')
    parser.add_argument('--task-condition-replay', help='Recorded object-specific visual task condition for a matched planner comparison.')
    parser.add_argument('--task-condition-default-replay', help='Recorded task condition used while the object is not observed.')
    parser.add_argument('--continuation-lookahead', action='store_true')
    parser.add_argument('--map-recovery', action='store_true')
    parser.add_argument('--static-global-map', action='store_true',
                        help='Keep global guidance on the registered static map; MPPI alone handles live dynamic tracks')
    parser.add_argument('--frozen-global-reference',
                        help='Frozen static-map path shared by all compared local planners; moving actors must be excluded')
    parser.add_argument('--ideal-steering-rollout', action='store_true',
                        help='Planning-only ablation; physical execution retains finite-rate 4WS.')
    parser.add_argument('--scan-drop-window', type=float, nargs=2, metavar=('START', 'DURATION'),
                        help='Controlled scan dropout in mission simulation seconds; evaluation only.')
    parser.add_argument('--unprojected-behavior-guides', action='store_true',
                        help='Ablation: retain the previous point-clipped behavior guides.')
    parser.add_argument('--world')
    args = parser.parse_args()
    if args.hybrid_sequences and args.planner not in ('conditioned','ranger-mppi'):
        parser.error('--hybrid-sequences requires the native conditioned/ranger-mppi planner')
    task_condition_planner = ((args.hybrid_sequences and args.planner in ('conditioned','ranger-mppi'))
                              or (args.planner == 'nav2-omni' and args.spatial_conditions_only))
    if args.proposal_interface == 'task_condition' and (
            args.policy != 'grounded' or args.proposal_selector != 'vlm' or not task_condition_planner):
        parser.error('--proposal-interface task_condition requires grounded/vlm with native hybrid MPPI or spatial-only Nav2 Omni')
    if args.spatial_conditions_only and args.proposal_interface != 'task_condition':
        parser.error('--spatial-conditions-only requires --proposal-interface task_condition')
    if not (math.isfinite(args.task_heading_scale_deg) and args.task_heading_scale_deg > 0
            and math.isfinite(args.task_clearance_scale_m) and args.task_clearance_scale_m > 0):
        parser.error('Task preference normalization scales must be positive and finite')
    if not math.isfinite(args.real_time_factor) or not 0 < args.real_time_factor <= 1:
        parser.error('--real-time-factor must lie in (0, 1]')
    if args.task_plan and not args.long_task:
        parser.error('--task-plan requires --long-task')
    if args.long_task and (args.station_handoff or not args.verified_recovery or not args.map_recovery or args.planner not in ('conditioned','ranger-mppi','nav2-omni')):
        parser.error('--long-task requires --verified-recovery --map-recovery and conditioned/ranger-mppi/nav2-omni planner')
    if args.controller and args.long_task and not args.task_plan:
        parser.error('long-task controller requires the archived --task-plan')
    if args.station_handoff and (args.scenario != 'office_handoff' or not args.verified_recovery or args.policy != 'grounded'):
        parser.error('--station-handoff requires --scenario office_handoff --verified-recovery --policy grounded')
    if (args.station_terminal_pose_only or args.station_proposal_replay) and not args.station_handoff:
        parser.error('station comparison switches require --station-handoff')
    if args.scan_drop_window and (args.scan_drop_window[0] < 0 or args.scan_drop_window[1] <= 0):
        parser.error("scan drop requires nonnegative start and positive duration")
    # Existing ViewDemo constructor interfaces, with the new runner overriding
    # control, task evaluation, semantic requests and result collection.
    args.condition, args.layout, args.scene = 'GEO_4WS', 'east_weave', 'furnished'
    args.method, args.target_class, args.semantic_provider = 'geometric', 'person', 'preset'
    args.clearance_threshold, args.trolley_clearance_threshold = .45, .35
    if args.direct_query_period <= 0:
        parser.error('--direct-query-period must be positive')
    if bool(args.task_condition_replay) != bool(args.task_condition_default_replay):
        parser.error('both recorded task conditions are required')
    if args.task_condition_replay and (args.proposal_interface != 'task_condition' or args.policy != 'grounded' or args.proposal_selector != 'vlm'):
        parser.error('task-condition replay requires grounded task-condition planning')
    needs_api = (args.policy == 'direct_vlm' or
        (args.policy in ('grounded', 'visual') and args.proposal_selector == 'vlm'
         and not args.station_proposal_replay and not args.task_condition_replay))
    if needs_api and not os.environ.get('DEEPSEEK_API_KEY'):
        raise SystemExit('This policy requires DEEPSEEK_API_KEY')
    if not args.controller:
        orchestrate(args)
        return
    demo.rclpy.init()
    node = GroundedDelivery(args)
    from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
    executor = (MultiThreadedExecutor(num_threads=4) if args.execution_architecture == 'process'
                else SingleThreadedExecutor())
    executor.add_node(node)
    wall = time.monotonic()
    try:
        while demo.rclpy.ok() and not node.done:
            executor.spin_once(timeout_sec=.03)
            if time.monotonic()-wall > args.max_wall_time-20:
                node.finish('WALL_TIMEOUT')
    finally:
        executor.shutdown(timeout_sec=35.)
        node.publish((0., 0., 0.))
        if args.execution_architecture == 'process':
            node.planner.close()
        node.runtime_timeline.close()
        node.trace.close(); node.frame_log.close(); node.overview_log.close(); node.events.close()
        node.observation_log.close()
        node.continuation_log.close()
        node.route_log.close()
        node.timing_log.close()
        node.command_timing_log.close()
        node.bank_diagnostic_log.close()
        node.pool.shutdown(wait=False)
        node.destroy_node()
        if demo.rclpy.ok():
            demo.rclpy.shutdown()


if __name__ == '__main__':
    main()
