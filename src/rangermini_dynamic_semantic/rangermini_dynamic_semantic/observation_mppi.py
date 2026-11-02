"""Reference implementation of the unified paper's local optimization step."""
from dataclasses import dataclass
import math
import time

from .command_semantics_mppi import CommandSemanticsMPPI, MPPIConfig, ranger_command, _wrap
from .task_observation import ViewConfig, view_score
from .ranger_4ws_kinematics import Ranger4WSKinematics, canonical_motion_mode


@dataclass(frozen=True)
class ObservationConfig:
    observation_weight: float = 3.0
    goal_weight: float = 7.0
    clearance_weight: float = 3.0
    terminal_view_weight: float = 1.0
    initial_switch_cost: float = 0.28
    internal_switch_cost: float = 0.28
    arrival_radius: float = 0.42
    arrival_heading_tolerance: float = math.radians(5.0)
    collision_margin: float = 0.035
    max_yaw_rate: float = 0.9
    lateral_candidates: bool = True
    omit_occlusion: bool = False
    enforce_approach_progress: bool = True
    ranger_4ws_modes: bool = True
    reference_weight: float = 8.0
    stop_action_penalty: float = 0.0


def project_control(control, behavior, cfg, max_yaw_rate=0.9):
    """Project requests before arbitration; retain forward and lateral signs."""
    vx, vy, wz = control
    lower = -cfg.max_speed if behavior == 'PARALLEL' and cfg.signed_parallel_velocity else 0.0
    vx = max(lower, min(cfg.max_speed, vx))
    vy = max(-cfg.max_speed, min(cfg.max_speed, vy))
    wz = max(-max_yaw_rate, min(max_yaw_rate, wz))
    if behavior == "ACKERMANN":
        vy = 0.0
        wz = max(-vx/cfg.min_turn_radius, min(vx/cfg.min_turn_radius, wz))
    elif behavior == "PARALLEL":
        wz = 0.0
        speed = math.hypot(vx, vy)
        if speed > cfg.max_speed:
            vx, vy = vx*cfg.max_speed/speed, vy*cfg.max_speed/speed
    elif behavior == "SIDE_SLIP":
        # Legacy callers may request a pure-lateral projection. New planning
        # populations include this reference inside PARALLEL.
        vx = 0.0
        wz = 0.0
        vy = max(-cfg.max_speed, min(cfg.max_speed, vy))
    elif behavior == "SPIN":
        vx = vy = 0.0
    elif behavior == "DRIVE":
        vy = wz = 0.0
    else:
        raise ValueError("unknown behavior projection")
    return vx, vy, wz


class ObservationMPPI(CommandSemanticsMPPI):
    """One MPPI solver with kinematically grouped candidate populations.

    Parallel translation includes oblique and pure-lateral reference strata
    in a shared update. STOP remains an explicitly scored alternative.
    """
    def __init__(self, config=None, observation=None, view=None, revalidation=None):
        cfg = config or MPPIConfig(allow_reverse=False)
        if cfg.allow_reverse:
            raise ValueError("unified paper profile requires allow_reverse=False")
        if cfg.samples_per_behavior < 1 or cfg.horizon < 1 or cfg.dt <= 0:
            raise ValueError("positive sample count, horizon and dt required")
        super().__init__(cfg, method="command_aware")
        self.obs = observation or ObservationConfig()
        if min(self.obs.observation_weight, self.obs.initial_switch_cost,
               self.obs.internal_switch_cost) < 0:
            raise ValueError("nonnegative objective weights required")
        self.view = view or ViewConfig()
        # Optional numerical task-risk objective; defaults retain the recorded
        # observation-aware planner. Only the local planner generates controls.
        self.revalidation = revalidation
        self._nominal_sample = False
        self._lateral_reference_sample = False

    def _pose_goal_reached(self, pose, goal, goal_yaw):
        return (math.dist(pose[:2],goal) <= self.obs.arrival_radius and
                (goal_yaw is None or abs(_wrap(goal_yaw-pose[2])) <= self.obs.arrival_heading_tolerance))

    def _reference(self, behavior, pose, goal, hypotheses, goal_yaw):
        if getattr(self, '_sample_reference_goal', None) is not None:
            goal = self._sample_reference_goal
            goal_yaw = None
        x, y, yaw = pose
        distance = math.hypot(goal[0]-x, goal[1]-y)
        bearing = math.atan2(goal[1]-y, goal[0]-x)
        if distance <= 1.2:
            if goal_yaw is not None:
                bearing = goal_yaw
            elif hypotheses:
                target = max(hypotheses, key=lambda h: h[3])
                bearing = math.atan2(target[1]-y, target[0]-x)
        error = _wrap(bearing-yaw)
        if distance <= self.obs.arrival_radius:
            if behavior == 'SPIN_THEN_DRIVE' and goal_yaw is not None:
                angular = _wrap(goal_yaw-yaw)
                return 0.0,0.0,(2.0*angular if abs(angular)>self.obs.arrival_heading_tolerance*.5 else 0.0),'SPIN'
            return 0.0,0.0,0.0,'DRIVE'
        speed = min(self.cfg.max_speed, 0.65*distance)
        if behavior == "ACKERMANN":
            return speed, 0.0, 1.8*error, "ACKERMANN"
        if behavior == "PARALLEL":
            if self._lateral_reference_sample:
                lateral_error = -(goal[0]-x)*math.sin(yaw)+(goal[1]-y)*math.cos(yaw)
                return 0.0, max(-speed, min(speed, 1.4*lateral_error)), 0.0, "PARALLEL"
            direction = math.atan2(goal[1]-y, goal[0]-x)-yaw
            return speed*math.cos(direction), speed*math.sin(direction), 0.0, "PARALLEL"
        if behavior == "SIDE_SLIP":
            lateral_error = -(goal[0]-x)*math.sin(yaw)+(goal[1]-y)*math.cos(yaw)
            return 0.0, max(-speed, min(speed, 1.4*lateral_error)), 0.0, "SIDE_SLIP"
        if abs(error) > 0.18:
            return 0.0, 0.0, 2.0*error, "SPIN"
        return speed, 0.0, 0.0, "DRIVE"

    def _evaluate(self, start, goal, points, hypotheses, behavior, initial_mode,
                  goal_yaw, sequence=None, current_steering_angles=None,
                  current_steering_transition=False, reference_trajectory=None):
        pose = list(start)
        controls, modes = [], []
        trajectory = [tuple(start)]
        clearance_cost = view_cost = reference_cost = 0.0
        clearance_min = float("inf")
        chassis = Ranger4WSKinematics(
            wheelbase=self.cfg.wheelbase, max_speed=self.cfg.max_speed,
            use_native=getattr(self, 'use_native_kinematics', True))
        initial_mode = canonical_motion_mode(initial_mode)
        chassis.mode = initial_mode or "STOPPED"
        if current_steering_angles is not None and len(current_steering_angles) == 4:
            chassis.angles = [float(a) for a in current_steering_angles]
        chassis.transition_pending = bool(current_steering_transition)
        executed_controls = []
        steering_trace, alignment_trace = [], []
        for k in range(self.cfg.horizon):
            if sequence is not None:
                control = sequence[k]
            elif behavior == "STOP":
                control = (0.0, 0.0, 0.0)
            else:
                vx, vy, wz, projection = self._reference(behavior, pose, goal,
                                                         hypotheses, goal_yaw)
                scale = (0.12 if behavior == "SPIN_THEN_DRIVE" else 0.16)
                if self._nominal_sample:
                    scale = 0.0
                scale *= 0.65 + 0.35*k/max(1, self.cfg.horizon-1)
                # Gaussian noise is sampled only in coordinates of this mode.
                if projection == "SPIN":
                    noisy = (0.0, 0.0, wz+self.rng.gauss(0, 1.5*scale))
                elif projection == "PARALLEL":
                    noisy = (vx+self.rng.gauss(0, scale), vy+self.rng.gauss(0, scale), 0.0)
                elif projection == "DRIVE":
                    noisy = (vx+self.rng.gauss(0, scale), 0.0, 0.0)
                elif projection == "SIDE_SLIP":
                    noisy = (0.0, vy+self.rng.gauss(0, scale), 0.0)
                else:
                    noisy = (vx+self.rng.gauss(0, scale), 0.0, wz+self.rng.gauss(0, 1.5*scale))
                control = project_control(noisy, projection, self.cfg, self.obs.max_yaw_rate)
            executed = ranger_command(*control, min_turn_radius=self.cfg.min_turn_radius,
                                      wheelbase=self.cfg.wheelbase)
            if getattr(self, 'ideal_steering_rollout', False):
                # Ablation changes prediction only, never the Gazebo adapter.
                instant_mode = chassis._requested_mode(*executed[:3])
                chassis.angles = list(chassis._target_wheels(*executed[:3], instant_mode)[0])
                chassis.mode = instant_mode
                chassis.transition_pending = False
            (vx, vy, wz), mode, angles, aligning = chassis.step(*executed[:3], self.cfg.dt)
            steering_trace.append(list(angles))
            alignment_trace.append(bool(aligning))
            controls.append(control)
            executed_controls.append((vx, vy, wz))
            modes.append(mode)
            # Intermediate checks reduce endpoint-only collision omissions.
            origin = tuple(pose)
            for fraction in (0.5, 1.0):
                midyaw = origin[2] + 0.5*wz*self.cfg.dt*fraction
                c, s = math.cos(midyaw), math.sin(midyaw)
                intermediate = (origin[0]+(c*vx-s*vy)*self.cfg.dt*fraction,
                                origin[1]+(s*vx+c*vy)*self.cfg.dt*fraction,
                                _wrap(origin[2]+wz*self.cfg.dt*fraction))
                _, clearance = self._clearance_cost(*intermediate, points)
                clearance_min = min(clearance_min, clearance)
                if clearance < self.obs.collision_margin:
                    return None
                if fraction == 1.0:
                    pose[:] = intermediate
            clearance_cost += self.cfg.dt*max(0.55-clearance, 0.0)**2
            trajectory.append(tuple(pose))
            if reference_trajectory:
                ref = reference_trajectory[min(k, len(reference_trajectory)-1)]
                reference_cost += self.cfg.dt*math.dist(pose[:2], ref[:2])**2
            visibility = view_score(pose, hypotheses, points, self.view,
                                    self.obs.omit_occlusion)
            view_cost += self.cfg.dt*(1-visibility)
        view_cost += self.obs.terminal_view_weight*(1-visibility)
        distance = math.hypot(goal[0]-pose[0], goal[1]-pose[1])
        goal_cost = max(distance-self.obs.arrival_radius, 0.0)
        if goal_yaw is not None:
            goal_cost += (4/7)*abs(_wrap(goal_yaw-pose[2]))
        if self.obs.enforce_approach_progress and self.revalidation is None:
            start_distance = math.hypot(goal[0]-start[0],goal[1]-start[1])
            start_potential = max(start_distance-self.obs.arrival_radius,0.0)
            if goal_yaw is not None:
                start_potential += (4/7)*abs(_wrap(goal_yaw-start[2]))
            # Before arrival, an equally good stationary view cannot replace
            # the active approach objective. A changing-mode candidate may
            # begin with STOP but must predict lower task potential by H.
            if (start_distance > self.obs.arrival_radius and
                    goal_cost >= start_potential-1e-4):
                return None
        active = [m for m in modes if m != "STOPPED"]
        initial_changes = int(bool(active) and initial_mode not in (None, "STOPPED", active[0]))
        internal_changes = sum(a != b for a, b in zip(active, active[1:]))
        switch_cost = self.obs.initial_switch_cost*initial_changes + self.obs.internal_switch_cost*internal_changes
        revalidation_terms = None
        if self.revalidation is None:
            cost = (self.obs.goal_weight*goal_cost + self.obs.clearance_weight*clearance_cost
                    + self.obs.observation_weight*view_cost + switch_cost
                    + self.obs.reference_weight*reference_cost)
        else:
            # A one-step terminal observation reduces decision risk, rather
            # than accumulating correlated predicted frames as new evidence.
            duration = self.cfg.horizon*self.cfg.dt if active else 0.0
            revalidation_terms = self.revalidation.evaluate(trajectory, visibility, duration)
            cost = (revalidation_terms["objective"]
                    + self.obs.clearance_weight*clearance_cost + switch_cost)
        terms = ({"goal": goal_cost, "clearance": clearance_cost,
                  "observation": view_cost, "switch": switch_cost,
                  "candidate_tracking": reference_cost}
                 if revalidation_terms is None else
                 {"expected_task_risk": revalidation_terms["expected_posterior_risk"],
                  "effort": revalidation_terms["effort_cost"],
                  "clearance": clearance_cost, "switch": switch_cost})
        return {"cost": cost, "controls": controls, "executed_controls": executed_controls,
                "steering_trace": steering_trace, "alignment_trace": alignment_trace,
                "modes": modes, "trajectory": trajectory,
                "terminal_steering_angles": list(chassis.angles),
                "terminal_steering_transition": chassis.transition_pending,
                "terminal_mode": chassis.mode,
                "min_clearance": clearance_min,
                "cost_terms": terms,
                "terminal_visibility": visibility,
                "revalidation_terms": revalidation_terms}

    def plan(self, state, goal, scan=None, current_mode=None, goal_yaw=None,
             hypotheses=(), obstacle_points=None, current_steering_angles=None,
             current_steering_transition=False, reference_trajectory=None):
        started = time.perf_counter()
        from .native_kernels import prepare_points
        points = prepare_points(obstacle_points if obstacle_points is not None
                                else self._scan_points_world(state, scan))
        mode = canonical_motion_mode(current_mode if current_mode is not None else self.last_command_mode)
        behaviors = ["ACKERMANN"]
        if self.obs.ranger_4ws_modes:
            behaviors.append("SPIN_THEN_DRIVE")
        if self.obs.lateral_candidates and self.obs.ranger_4ws_modes:
            behaviors.insert(1, "PARALLEL")
        # At a pose waypoint, translation is complete but orientation may not
        # be. A 4WS platform can finish it using its stationary rotation family.
        if (self.obs.ranger_4ws_modes and math.dist(state[:2],goal)<=self.obs.arrival_radius
                and goal_yaw is not None and not self._pose_goal_reached(state,goal,goal_yaw)):
            behaviors = ['SPIN_THEN_DRIVE']
        groups = []
        population = []
        rejected = 0
        total_candidates = 0
        sample_budget = {}
        for behavior in behaviors + ["STOP"]:
            samples = []
            # Preserve the old total budget: the two translation reference
            # strata now feed one population and one weighted update.
            n = self.cfg.samples_per_behavior
            count = 1 if behavior == "STOP" else (2*n if behavior == "PARALLEL" else n)
            sample_budget[behavior] = count
            for sample_index in range(count):
                references = [None]+list(getattr(self, 'obstacle_reference_goals', ()))
                self._sample_reference_goal = references[sample_index % len(references)] if behavior != 'STOP' else None
                self._lateral_reference_sample = behavior == "PARALLEL" and sample_index >= n
                self._nominal_sample = (sample_index < len(references) or
                                       (behavior == "PARALLEL" and sample_index == n))
                total_candidates += 1
                sample = self._evaluate(state, goal, points, hypotheses, behavior, mode,
                                        goal_yaw, current_steering_angles=current_steering_angles,
                                        current_steering_transition=current_steering_transition,
                                        reference_trajectory=reference_trajectory)
                if sample is None:
                    rejected += 1
                else:
                    samples.append(sample)
                    sample["behavior"] = behavior
            self._lateral_reference_sample = False
            self._sample_reference_goal = None
            self._nominal_sample = False
            if not samples:
                continue
            if behavior == "STOP" and not self._pose_goal_reached(state,goal,goal_yaw):
                penalty = self.obs.stop_action_penalty
                for item in samples:
                    item["cost"] += penalty
                    item["cost_terms"] = dict(item["cost_terms"], stop_action=penalty)
            diverse = []
            for item in sorted(samples, key=lambda s: s["cost"]):
                endpoint = item["trajectory"][-1][:2]
                if all(math.dist(endpoint, prior["trajectory"][-1][:2]) >= 0.12
                       for prior in diverse):
                    diverse.append(item)
                if len(diverse) == 4:
                    break
            for rank, item in enumerate(diverse):
                population.append({
                    "candidate_id": f"{behavior}_{rank+1}",
                    "motion_behavior": behavior,
                    "feasible": True,
                    "cost": item["cost"],
                    "min_clearance_m": item["min_clearance"],
                    "trajectory": item["trajectory"],
                    "controls": item["controls"],
                    "executed_controls": item["executed_controls"],
                    "modes": item["modes"],
                })
            minimum = min(s["cost"] for s in samples)
            weights = [math.exp(-min(60, (s["cost"]-minimum)/self.cfg.temperature)) for s in samples]
            normalizer = sum(weights)
            weights = [w/normalizer for w in weights]
            # Do not average incompatible first-step modes within spin/drive.
            best_sample = min(samples, key=lambda s: s["cost"])
            first_mode = best_sample["modes"][0]
            compatible = [(s, w) for s, w in zip(samples, weights) if s["modes"][0] == first_mode]
            z = sum(w for _, w in compatible)
            weighted = [tuple(sum(w*s["controls"][k][i] for s, w in compatible)/z
                              for i in range(3)) for k in range(self.cfg.horizon)]
            checked = self._evaluate(state, goal, points, hypotheses, behavior,
                                     mode, goal_yaw, sequence=weighted,
                                     current_steering_angles=current_steering_angles,
                                     current_steering_transition=current_steering_transition,
                                     reference_trajectory=reference_trajectory)
            fallback = checked is None
            checked = best_sample if fallback else checked
            if behavior == "STOP" and not self._pose_goal_reached(state,goal,goal_yaw):
                checked["cost"] += self.obs.stop_action_penalty
                checked["cost_terms"] = dict(checked["cost_terms"],
                                             stop_action=self.obs.stop_action_penalty)
            groups.append({"behavior": behavior, "evaluation": checked,
                           "weighted_action_fallback": fallback})
        if not groups:
            action, selected, detail = (0.0, 0.0, 0.0), "STOP", None
        else:
            if (getattr(self, 'require_navigation_progress', False)
                    and not self._pose_goal_reached(state, goal, goal_yaw)):
                progressing = [g for g in groups if g['behavior'] != 'STOP']
                groups = progressing or groups
            best = min(groups, key=lambda g: g["evaluation"]["cost"])
            detail = best["evaluation"]
            action, selected = detail["controls"][0], best["behavior"]
        revalidation_decision = None
        if self.revalidation is not None:
            revalidation_decision = self.revalidation.choose([
                {"candidate_id": g["behavior"], "feasible": True,
                 "objective": g["evaluation"]["cost"]} for g in groups])
            if revalidation_decision["action"] == "no_new_observation":
                action, selected = (0.0, 0.0, 0.0), "NO_NEW_OBSERVATION"
                risk = revalidation_decision["objective"]
                detail = {"cost": risk, "cost_terms": {"current_task_risk": risk},
                          "trajectory": [tuple(state)],
                          "terminal_visibility": view_score(state, hypotheses, points, self.view,
                                                            self.obs.omit_occlusion),
                          "min_clearance": self._clearance_cost(*state, points)[1],
                          "revalidation_terms": {"prior_risk": risk,
                                                 "expected_posterior_risk": risk,
                                                 "expected_task_risk_reduction": 0.0,
                                                 "effort_cost": 0.0}}
        command_mode = ranger_command(*action)[3]
        if detail and detail.get("modes"):
            # Report the mode produced by the same four-wheel rollout that
            # selected the action. In particular, a pure vy command is
            # Pure-lateral and oblique translation both execute as PARALLEL.
            command_mode = detail["modes"][0]
        self.last_command_mode = command_mode
        return {"action": action, "selected_behavior": selected,
                "sampling_profile": "merged_translation_v1",
                "applied_motion_condition": getattr(self, "active_motion_condition", None),
                "family_sample_budget": sample_budget,
                "objective_type": "task_risk" if self.revalidation is not None else "observability",
                "command_mode": command_mode, "initial_mode": mode,
                "candidate_count": total_candidates, "rejected_candidates": rejected,
                "planning_time_ms": (time.perf_counter()-started)*1000,
                "predicted_score": detail["cost"] if detail else None,
                "cost_terms": detail["cost_terms"] if detail else None,
                "terminal_visibility": detail["terminal_visibility"] if detail else 0.0,
                "revalidation_terms": detail["revalidation_terms"] if detail else None,
                "revalidation_decision": revalidation_decision,
                "selected_trajectory": detail["trajectory"] if detail else [],
                "terminal_steering_angles": detail.get("terminal_steering_angles") if detail else None,
                "terminal_steering_transition": detail.get("terminal_steering_transition", False) if detail else False,
                "terminal_mode": detail.get("terminal_mode", "STOPPED") if detail else "STOPPED",
                "predicted_executed_controls": detail["executed_controls"] if detail else [],
                "candidate_population": population,
                "predicted_min_clearance_m": detail["min_clearance"] if detail else None,
                "current_observed_clearance_m": self._clearance_cost(*state, points)[1] if len(points) else None,
                "candidate_summary": [{"behavior": g["behavior"],
                    "trajectory": g["evaluation"]["trajectory"],
                    "predicted_executed_controls": g["evaluation"]["executed_controls"],
                    "predicted_modes": g["evaluation"]["modes"],
                    "terminal_steering_angles": g["evaluation"].get("terminal_steering_angles"),
                    "terminal_steering_transition": g["evaluation"].get("terminal_steering_transition", False),
                    "predicted_min_clearance_m": g["evaluation"]["min_clearance"],
                    "cost": g["evaluation"]["cost"],
                    "cost_terms": g["evaluation"]["cost_terms"],
                    "terminal_visibility": g["evaluation"]["terminal_visibility"],
                    "revalidation_terms": g["evaluation"]["revalidation_terms"],
                    "weighted_action_fallback": g["weighted_action_fallback"]} for g in groups]}
