"""Bounded second-stage 4WS rollout from an interaction candidate's end state.

Uses observed scan points, registered geometry and current track velocities.
No scene fixture poses, future actor schedules or invented future images.
This is a finite-horizon continuation preference, not a reachability proof.
"""
from dataclasses import replace
import math
import time

from .interaction_mppi import InteractionMPPI


class ContinuationEvaluator:
    def __init__(self, config, static_boxes, weight=.35):
        self.config = replace(config, samples_per_behavior=2, horizon=32)
        self.static_boxes = static_boxes
        self.weight = weight

    def evaluate(self, bank, pose, scan, tracks, stamp, target, target_yaw):
        started = time.perf_counter()
        records = []
        # Keep the first-stage sampler's RNG and execution commitment untouched.
        observer = InteractionMPPI(self.config, static_boxes=self.static_boxes)
        points = observer._scan_points_world(pose, scan)
        points = [p for p in points if all(
            math.dist(p, t.position) > t.radius+.18 for t in tracks)]
        for candidate in bank:
            plan = candidate['plan']
            path = plan['selected_trajectory']
            if not candidate['feasible'] or not path:
                continue
            branch = InteractionMPPI(self.config, prediction=True, static_boxes=self.static_boxes)
            branch.tracks = list(tracks)
            branch.predict_stamp = stamp+(len(path)-1)*self.config.dt
            branch.main_goal = target
            branch.main_yaw = target_yaw
            follow = branch.plan(path[-1], target, obstacle_points=points,
                current_mode=plan['terminal_mode'], goal_yaw=target_yaw,
                current_steering_angles=plan['terminal_steering_angles'],
                current_steering_transition=plan['terminal_steering_transition'])
            supported = bool(follow['selected_trajectory'])
            # Missing sampled continuation is uncertain, not proof of blockage.
            value = follow['predicted_score'] if supported else 30.0
            increment = self.weight*value
            record = {'behavior_id': candidate['behavior_id'],
                'branch_pose': path[-1], 'branch_mode': plan['terminal_mode'],
                'branch_steering_rad': plan['terminal_steering_angles'],
                'target': list(target), 'target_yaw': target_yaw,
                'continuation_sample_found': supported,
                'continuation_score': value, 'score_increment': increment,
                'first_stage_score': plan['predicted_score'],
                'continuation_trajectory': follow['selected_trajectory'],
                'continuation_clearance_m': follow['predicted_min_clearance_m']}
            candidate['continuation'] = record
            plan['predicted_score'] += increment
            if plan['cost_terms'] is not None:
                plan['cost_terms']['continuation'] = increment
            records.append(record)
        return {'sim_time': stamp, 'measured_pose': list(pose),
                'wall_ms': (time.perf_counter()-started)*1000, 'candidates': records}
