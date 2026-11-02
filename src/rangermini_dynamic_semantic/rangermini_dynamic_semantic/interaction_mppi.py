"""Behavior conditions and tracked dynamic occupancy inside MPPI weights."""
from dataclasses import replace
import math
import time
import numpy as np

from .observation_mppi import ObservationMPPI, ObservationConfig
from . import native_kernels


class InteractionMPPI(ObservationMPPI):
    def _scan_points_world(self, pose, scan):
        # Opt-in acquisition-pose compensation for rotating station approach.
        return super()._scan_points_world(getattr(self,'scan_observation_pose',pose),scan)

    def __init__(self, config, prediction=True, static_boxes=None):
        super().__init__(config, ObservationConfig(observation_weight=0,
            arrival_radius=.24, enforce_approach_progress=False,
            initial_switch_cost=1.5, internal_switch_cost=1.0,
            reference_weight=0, stop_action_penalty=4))
        self.prediction = prediction
        self.tracks = []
        self.interaction_behavior = "CONTINUE"
        self.main_goal = None
        self.predict_stamp = 0
        self.static_boxes = np.asarray(static_boxes or [], dtype=float).reshape(-1, 5)

    def _reference(self, behavior, pose, goal, hypotheses, goal_yaw):
        # A service pose's terminal heading must not replace the direction to
        # its position while still approaching. Each rollout step switches to
        # final orientation only inside the position capture region.
        if (getattr(self, 'service_pose_sampling', False)
                and math.dist(pose[:2], goal) > self.obs.arrival_radius):
            return super()._reference(behavior, pose, goal, hypotheses, None)
        tolerance = getattr(self, 'capture_heading_tolerance', None)
        if (getattr(self, 'capture_spin_only', False) and
                behavior == 'SPIN_THEN_DRIVE' and goal_yaw is not None):
            error = math.atan2(math.sin(goal_yaw-pose[2]), math.cos(goal_yaw-pose[2]))
            return 0.0, 0.0, 2.0*error if abs(error) > tolerance*.5 else 0.0, 'SPIN'
        if (tolerance is not None and behavior == 'SPIN_THEN_DRIVE' and
                goal_yaw is not None and math.dist(pose[:2], goal) < 1.2):
            error = math.atan2(math.sin(goal_yaw-pose[2]), math.cos(goal_yaw-pose[2]))
            if abs(error) > tolerance*.5:
                return 0.0, 0.0, 2.0*error, 'SPIN'
        return super()._reference(behavior, pose, goal, hypotheses, goal_yaw)

    def _clearance_cost(self, x, y, yaw, points):
        if getattr(self, 'use_native_clearance', True):
            return native_kernels.clearance(x, y, yaw, points,
                                           self.cfg.body_length, self.cfg.body_width)
        return self._clearance_cost_python(x, y, yaw, points)

    def _clearance_cost_python(self, x, y, yaw, points):
        if len(points) == 0:
            return 0, 20
        pts = np.asarray(points)
        dx, dy = pts[:, 0]-x, pts[:, 1]-y
        c, s = math.cos(yaw), math.sin(yaw)
        ox = np.maximum(np.abs(c*dx+s*dy)-self.cfg.body_length/2, 0)
        oy = np.maximum(np.abs(-s*dx+c*dy)-self.cfg.body_width/2, 0)
        ds = np.hypot(ox, oy)
        return float(np.sum(np.maximum(.55-ds, 0)**2)*3), float(ds.min())

    def _evaluate(self, *args, **kwargs):
        motion_group = args[4] if len(args) > 4 else kwargs.get("behavior")
        if getattr(self, 'capture_spin_only', False) and motion_group not in ('SPIN_THEN_DRIVE', 'STOP'):
            return None
        if self.interaction_behavior == "WAIT" and motion_group != "STOP":
            return None
        result = super()._evaluate(*args, **kwargs)
        if result is None:
            return None
        trajectory = np.asarray(result["trajectory"])
        if (getattr(self, 'require_navigation_progress', False) and motion_group != 'STOP'
                and self.interaction_behavior == 'CONTINUE'):
            start = trajectory[0]
            end = trajectory[-1]
            goal = args[1] if len(args) > 1 else kwargs['goal']
            distance = math.dist(start[:2], goal)
            first = result['controls'][0]
            if (not self._pose_goal_reached(start, goal, self.main_yaw)
                    and math.hypot(first[0], first[1])+abs(first[2]) < 1e-4):
                return None
            progress_threshold = min(.08, max(.005, (distance-self.obs.arrival_radius)*.5))
            progressed = math.dist(start[:2], end[:2]) > progress_threshold
            turn_threshold = .12 if distance > .30 else .03
            turned = abs(math.atan2(math.sin(end[2]-start[2]), math.cos(end[2]-start[2]))) > turn_threshold
            if not self._pose_goal_reached(start, goal, self.main_yaw) and not (progressed or turned):
                return None
        condition = getattr(self, 'active_motion_condition', None)
        if condition is not None and condition.get('retain_heading'):
            # Process requirement: terminal heading alone does not constrain
            # the manner of approach. Check every predicted physical pose.
            errors = np.abs(np.arctan2(np.sin(trajectory[:, 2]-condition['heading']),
                                      np.cos(trajectory[:, 2]-condition['heading'])))
            if np.any(errors > condition['heading_tolerance_rad']):
                return None
        heading_tolerance = getattr(self, 'capture_heading_tolerance', None)
        if (heading_tolerance is not None and self.main_yaw is not None and
                self.interaction_behavior != 'WAIT'):
            heading_error = abs(math.atan2(math.sin(trajectory[-1, 2]-self.main_yaw),
                                          math.cos(trajectory[-1, 2]-self.main_yaw)))
            if heading_error > heading_tolerance:
                return None
        if len(self.static_boxes):
            # Complete footprint against registered geometry, including low
            # plinths and thin frames missed by sparse instantaneous scan rays.
            mid = (trajectory[:-1]+trajectory[1:])/2
            samples = np.vstack([trajectory, mid])
            px, py, yaw = samples[:, 0:1], samples[:, 1:2], samples[:, 2:3]
            bx, by, angle, hx, hy = self.static_boxes.T
            dx, dy = bx-px, by-py
            c, s = np.cos(yaw), np.sin(yaw)
            ca, sa = np.cos(angle), np.sin(angle)
            co, si = np.abs(np.cos(yaw-angle)), np.abs(np.sin(yaw-angle))
            hl, hw = self.cfg.body_length/2, self.cfg.body_width/2
            separation = np.maximum.reduce([
                np.abs(dx*c+dy*s)-hl-hx*co-hy*si,
                np.abs(-dx*s+dy*c)-hw-hx*si-hy*co,
                np.abs(dx*ca+dy*sa)-hx-hl*co-hw*si,
                np.abs(-dx*sa+dy*ca)-hy-hl*si-hw*co])
            # A measured pose can be inside the preferred margin after a
            # steering/execution discrepancy. Permit bounded clearance for a
            # steering interval, then require the normal margin; otherwise
            # every rollout, including one moving away, becomes infeasible.
            sample_times = np.concatenate([np.arange(len(trajectory))*self.cfg.dt,
                (np.arange(len(trajectory)-1)+.5)*self.cfg.dt])
            initial_floor = np.where(separation[0] < .10,
                np.minimum(separation[0], .005), .10)
            required = np.minimum(.10, initial_floor[None, :]+
                np.maximum(0, sample_times[:, None]-1.2)*.06)
            if np.any(separation < required-.002):
                return None
            result['min_clearance'] = min(result['min_clearance'], float(separation.min()))
            result['cost_terms']['registered_map_min_gap'] = float(separation.min())
            result['cost_terms']['recovering_initial_map_margin'] = bool(np.any(initial_floor < .10))
        duration = np.arange(len(trajectory))*self.cfg.dt
        dynamic_cost = 0
        base_dynamic_cost = 0
        minimum = 20
        task_condition = getattr(self, 'active_motion_condition', None) or {}
        relevant_ids = set(task_condition.get('relevant_object_ids', ()))
        preferred_clearance = float(task_condition.get('preferred_clearance_m', .65))
        for track in self.tracks:
            registered_box = getattr(track, 'registered_box', None)
            if registered_box is not None:
                # Its exact hard geometry already belongs to static_boxes.
                # Registration exposes semantic identity, not a second dynamic
                # collision disk or artificial motion uncertainty.
                if track.evidence_id not in relevant_ids:
                    continue
                bx,by,angle,hx,hy=registered_box
                dx,dy=bx-trajectory[:,0],by-trajectory[:,1]
                yaw=trajectory[:,2];co=np.abs(np.cos(yaw-angle));si=np.abs(np.sin(yaw-angle))
                hl,hw=self.cfg.body_length/2,self.cfg.body_width/2
                gap=np.maximum.reduce([
                    np.abs(dx*np.cos(yaw)+dy*np.sin(yaw))-hl-hx*co-hy*si,
                    np.abs(-dx*np.sin(yaw)+dy*np.cos(yaw))-hw-hx*si-hy*co,
                    np.abs(dx*math.cos(angle)+dy*math.sin(angle))-hx-hl*co-hw*si,
                    np.abs(-dx*math.sin(angle)+dy*math.cos(angle))-hy-hl*si-hw*co])
                scale=float(task_condition.get('clearance_scale_m',1.))
                dynamic_cost+=float(np.sum((np.maximum(preferred_clearance-gap,0)/scale)**2))*self.cfg.dt*9
                minimum=min(minimum,float(gap.min()))
                continue
            age = max(0, self.predict_stamp-track.stamp)
            velocity = np.asarray(track.velocity) if self.prediction else np.zeros(2)
            centers = np.asarray(track.position)+(duration+age)[:, None]*velocity
            dx, dy = centers[:, 0]-trajectory[:, 0], centers[:, 1]-trajectory[:, 1]
            c, s = np.cos(trajectory[:, 2]), np.sin(trajectory[:, 2])
            uncertainty = np.minimum(.16, .025+.035*duration) if self.prediction else .025
            gap = np.hypot(np.maximum(np.abs(c*dx+s*dy)-self.cfg.body_length/2, 0),
                           np.maximum(np.abs(-s*dx+c*dy)-self.cfg.body_width/2, 0))-track.radius-uncertainty
            minimum = min(minimum, float(gap.min()))
            if np.any(gap < .12):
                return None
            target_clearance = preferred_clearance if track.evidence_id in relevant_ids else .65
            base_dynamic_cost += float(np.sum(np.maximum(.65-gap, 0)**2))*self.cfg.dt*9
            spatial_scale = (float(task_condition.get('clearance_scale_m', 1.))
                             if track.evidence_id in relevant_ids else 1.)
            dynamic_cost += float(np.sum((np.maximum(target_clearance-gap, 0)/spatial_scale)**2))*self.cfg.dt*9
        behavior_cost = 0
        if self.interaction_behavior == "WAIT":
            behavior_cost = float(np.sum(np.asarray(result["executed_controls"])[:, :2]**2))*self.cfg.dt*25
        elif self.interaction_behavior.startswith(("PASS", "YIELD")):
            # Keep orientation while translating around an encounter; no
            # chassis mode is commanded by this semantic objective.
            start_yaw = trajectory[0, 2]
            behavior_cost = float(np.sum(np.arctan2(np.sin(trajectory[:, 2]-start_yaw),
                np.cos(trajectory[:, 2]-start_yaw))**2))*self.cfg.dt*.8
        result["cost"] += dynamic_cost+behavior_cost
        result["cost_terms"].update(dynamic=dynamic_cost, interaction=behavior_cost)
        result['cost_terms']['task_clearance_preference'] = dynamic_cost-base_dynamic_cost
        if task_condition.get('heading_policy') == 'preserve_current_heading':
            # A task preference, not a commanded motion family: MPPI remains
            # free to realize it with translation, curved motion, or a hybrid
            # sequence if the joint objective and wheel-state rollout support it.
            preferred_heading = task_condition.get('heading', float(trajectory[0, 2]))
            yaw_error = np.arctan2(np.sin(trajectory[:, 2]-preferred_heading),
                                   np.cos(trajectory[:, 2]-preferred_heading))
            heading_scale = float(task_condition.get('heading_scale_rad', 1.))
            heading_preference = float(np.sum((yaw_error/heading_scale)**2))*self.cfg.dt*1.25
            result['cost'] += heading_preference
            result['cost_terms']['task_heading_preference'] = heading_preference
        result['cost_terms']['task_condition'] = {
            'anchor_id': task_condition.get('anchor_id'),
            'relevant_object_ids': sorted(relevant_ids),
            'heading_policy': task_condition.get('heading_policy'),
            'preferred_clearance_m': preferred_clearance if relevant_ids else None,
            'motion_preference': task_condition.get('motion_preference', 'auto')}
        # Score physical motion across the rollout, including hybrid phases.
        # A preference never masks families or bypasses steering/collision checks.
        preferred_mode = {'curved_driving': 'DUAL_ACKERMANN',
                          'parallel_translation': 'PARALLEL',
                          'reorient_then_advance': 'SPINNING'}.get(
                              task_condition.get('motion_preference', 'auto'))
        preference_cost = 0.0
        if preferred_mode:
            moving_modes = [mode for mode, control in zip(result['modes'], result['executed_controls'])
                            if math.hypot(control[0], control[1])+abs(control[2]) > .001]
            if preferred_mode == 'SPINNING':
                # Rotation may be followed by translation; do not penalize that
                # completion phase once the sequence includes reorientation.
                preference_cost = (self.cfg.dt*len(moving_modes)
                                   if moving_modes and 'SPINNING' not in moving_modes else 0.0)
            else:
                preference_cost = self.cfg.dt*sum(mode != preferred_mode for mode in moving_modes)
            result['cost'] += preference_cost
        result['cost_terms']['task_motion_preference'] = preference_cost
        if self.interaction_behavior == 'WAIT':
            # Keep the explicit waiting population consistent with the STOP
            # action penalty used inside progressing populations. Otherwise
            # a nearby zero-length wait guide evades that shared effort cost.
            wait_cost = getattr(self, 'wait_action_penalty', 0.0)
            result['cost'] += wait_cost
            result['cost_terms']['wait_action'] = wait_cost
        if self.main_goal is not None:
            # Compare all behavioral populations against the SAME service
            # progress objective; a nearby side guide must not appear cheaper
            # merely because it is closer than the delivery destination.
            local_goal = result["cost_terms"]["goal"]
            navigation = max(0, math.dist(trajectory[-1, :2], self.main_goal)-self.obs.arrival_radius)
            if self.main_yaw is not None:
                navigation += (4/7)*abs(math.atan2(math.sin(trajectory[-1, 2]-self.main_yaw),
                                                  math.cos(trajectory[-1, 2]-self.main_yaw)))
            guide_weight = 12.0
            guided = self.interaction_behavior.startswith(("PASS", "YIELD"))
            guide = guide_weight*local_goal if guided else 0
            # All behaviors pay the SAME intermediate route-reference cost.
            # Charging it only to CONTINUE gives WAIT a cost exemption and
            # can prefer indefinite stopping despite a feasible side-step.
            route_reference = 0.0
            if getattr(self, 'route_goal', None) is not None:
                route_reference = guide_weight*max(0.0,
                    math.dist(trajectory[-1, :2], self.route_goal)-.07)
                route_yaw = getattr(self, 'route_yaw', None)
                if task_condition.get('heading_policy') == 'preserve_current_heading':
                    route_yaw = task_condition['heading']
                if route_yaw is not None:
                    route_reference += guide_weight*(4/7)*abs(math.atan2(
                        math.sin(trajectory[-1, 2]-route_yaw),
                        math.cos(trajectory[-1, 2]-route_yaw)))
            if self.interaction_behavior.startswith("YIELD"):
                displacement = trajectory[:, :2]-trajectory[0, :2]
                heading = np.asarray([math.cos(trajectory[0, 2]), math.sin(trajectory[0, 2])])
                forward_progress = displacement@heading
                if np.max(forward_progress) > .35:
                    return None
                guide += float(np.sum(forward_progress**2))*self.cfg.dt*8
            result["cost"] += self.obs.goal_weight*(navigation-local_goal)+guide+route_reference
            result["cost_terms"].update(goal=navigation, behavior_guide=guide,
                                       route_reference=route_reference)
        result["min_clearance"] = min(result["min_clearance"], minimum)
        return result

    def behavior_plan(self, state, main_goal, scan, tracks, stamp, behavior, **kwargs):
        self.tracks = list(tracks)
        self.predict_stamp = stamp
        self.interaction_behavior = behavior
        self.main_goal = main_goal
        goal = main_goal
        yaw = kwargs.pop("goal_yaw", None)
        bound_goal = kwargs.pop("bound_goal", None)
        self.main_yaw = yaw
        self.obstacle_reference_goals = []
        if getattr(self, 'obstacle_seeded_sampling', False) and behavior == 'CONTINUE':
            # Local proposal means from CURRENT tracks, never a global detour
            # or fixture trajectory. Both conditioned and unconditioned MPPI
            # use the same symmetric seeds and unchanged rollout budget.
            heading = getattr(self, 'route_yaw', None)
            if heading is None:
                heading = math.atan2(main_goal[1]-state[1], main_goal[0]-state[0])
            forward = np.array([math.cos(heading), math.sin(heading)])
            lateral = np.array([-forward[1], forward[0]])
            relevant = []
            for track in self.tracks:
                center = np.asarray(track.position)+max(0., stamp-track.stamp)*np.asarray(track.velocity)
                delta = center-np.asarray(state[:2])
                ahead, side = float(delta@forward), float(delta@lateral)
                if -.8 < ahead < 4.5 and abs(side) < track.radius+self.cfg.body_width/2+.45:
                    relevant.append((max(0., ahead), side, track.radius))
            if relevant:
                _, side, radius = min(relevant)
                # Include the same dynamic uncertainty and terminal capture
                # tolerance used by rollout. Omitting capture tolerance makes
                # a nominal seed stop short of its own required separation.
                clearance_offset = (radius+self.cfg.body_width/2+.12+.16+
                                    self.obs.arrival_radius)
                for direction in (-1, 1):
                    reference = (np.asarray(state[:2])+.5*forward+
                                 (side+direction*clearance_offset)*lateral)
                    self.obstacle_reference_goals.append(tuple(map(float, reference)))
        if behavior == 'CONTINUE' and getattr(self, 'route_goal', None) is not None:
            goal = self.route_goal
            yaw = getattr(self, 'route_yaw', None)
            condition = getattr(self, 'active_motion_condition', None) or {}
            if condition.get('heading_policy') == 'preserve_current_heading':
                # A path tangent is a direction of travel, not a mandatory
                # body heading for a four-wheel-steered platform.
                yaw = condition['heading']
        if behavior == "WAIT":
            goal = state[:2]
            yaw = state[2]
        elif behavior.startswith(("PASS", "YIELD")):
            dx, dy = main_goal[0]-state[0], main_goal[1]-state[1]
            length = max(.01, math.hypot(dx, dy))
            forward = .03 if behavior.startswith("YIELD") else min(1.25, length)
            side = 1 if behavior.endswith("LEFT") else -1
            # These are local guide poses generated from the current state,
            # not pre-registered complete left/right routes.
            # Clear the FULL robot/object envelopes, including prediction
            # uncertainty. A 0.9 m guide left insufficient room after the
            # rollout's arrival tolerance and steering transition.
            lateral = 1.45
            gx = state[0]+forward*dx/length-side*lateral*dy/length
            gy = state[1]+forward*dy/length+side*lateral*dx/length
            # Do not generate a guide in the wall or inside unrelated rooms.
            if state[1] < -.4:
                gy = min(-.80, max(-3.40, gy))
            goal = (gx, gy)
            yaw = state[2]
        if bound_goal is not None and behavior.startswith("YIELD"):
            goal = bound_goal
        raw_goal = tuple(goal)
        support = getattr(self, 'guide_support', None)
        if behavior.startswith(('PASS', 'YIELD')) and support is not None:
            # Preserve the requested side and heading while shrinking the
            # guide into configuration space. Rollouts still check the path.
            if not support((*goal, yaw)):
                for index in range(19, 0, -1):
                    alpha = index/20.0
                    projected = (state[0]+alpha*(raw_goal[0]-state[0]),
                                 state[1]+alpha*(raw_goal[1]-state[1]))
                    if math.dist(projected, state[:2]) >= .15 and support((*projected, yaw)):
                        goal = projected
                        break
        points = self._scan_points_world(state, scan)
        if tracks:
            # Remove tracked current returns before adding time-indexed
            # occupancy; static and untracked returns remain hard obstacles.
            points = [p for p in points if all(math.dist(p, t.position) > t.radius+.18 for t in tracks)]
        route_tracking = (behavior == 'CONTINUE' and
                          getattr(self, 'route_goal', None) is not None)
        observation_config = self.obs
        if route_tracking:
            # Intermediate geometric references are staging constraints, not
            # task endpoints. A tight local tolerance is needed to complete
            # the ordered translate/reorient sequence around a narrow passage.
            self.obs = replace(self.obs, arrival_radius=0.07)
        try:
            plan = self.plan(state, goal, obstacle_points=points,
                             goal_yaw=yaw, **kwargs)
        finally:
            self.obs = observation_config
        feasible = bool(plan["selected_trajectory"])
        if behavior == 'CONTINUE' and feasible and math.dist(state[:2], goal) > .30:
            terminal = plan['selected_trajectory'][-1]
            moved = math.dist(state[:2], terminal[:2]) > .08
            turned = yaw is not None and abs(math.atan2(math.sin(terminal[2]-state[2]), math.cos(terminal[2]-state[2]))) > .12
            feasible = moved or turned
        if behavior.startswith(("PASS", "YIELD")) and feasible:
            feasible = (math.dist(state[:2], plan["selected_trajectory"][-1][:2]) > .10 or
                        (behavior.startswith("YIELD") and math.dist(state[:2], goal) < .3))
        return {"behavior_id": behavior, "goal": list(goal), "goal_yaw": yaw, "raw_goal": list(raw_goal),
                "guide_projected": math.dist(goal, raw_goal) > 1e-9,
                "feasible": feasible, "plan": plan}

    def bank(self, state, main_goal, scan, tracks, stamp, behavior_ids=None, **kwargs):
        started = time.perf_counter()
        behaviors = ["CONTINUE", "WAIT"]
        if state[1] < -.65:
            behaviors += ["PASS_LEFT", "PASS_RIGHT", "YIELD_LEFT", "YIELD_RIGHT"]
        if behavior_ids is not None:
            behaviors = [b for b in behaviors if b in behavior_ids]
        bank = [self.behavior_plan(state, main_goal, scan, tracks, stamp, b, **kwargs) for b in behaviors]
        self.bank_time_ms = (time.perf_counter()-started)*1000
        return bank
