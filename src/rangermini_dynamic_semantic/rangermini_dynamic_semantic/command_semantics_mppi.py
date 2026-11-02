"""Local MPPI planner with matched ideal and Ranger-command rollout models.

The module is ROS-independent so the exact command arbiter and rollout can be
replayed in deterministic experiments. It models planar kinematics; it is not a
wheel, steering-servo, or tire-dynamics simulator.
"""
from dataclasses import dataclass
import math
import random
import time


@dataclass(frozen=True)
class MPPIConfig:
    samples_per_behavior: int = 32
    horizon: int = 16
    dt: float = 0.12
    temperature: float = 2.0
    max_speed: float = 0.42
    min_turn_radius: float = 0.4764
    wheelbase: float = 0.494
    body_length: float = 0.751
    body_width: float = 0.500
    lidar_x: float = 0.30
    lidar_pitch: float = 0.523599
    random_seed: int = 17
    account_initial_mode_switch: bool = True
    allow_reverse: bool = True
    signed_parallel_velocity: bool = False


def ranger_command(vx, vy, wz, min_turn_radius=0.4764, wheelbase=0.494,
                   track=0.364, deadband=1e-4):
    """Mirror the Ranger ROS command arbiter; return vx, vy, wz, mode, reason."""
    speed = math.hypot(vx, vy)
    if speed > 1.5:
        scale = 1.5 / speed
        vx, vy = vx * scale, vy * scale
    wz = max(-4.8, min(4.8, wz))
    if math.hypot(vx, vy) <= deadband and abs(wz) <= deadband:
        return 0.0, 0.0, 0.0, "STOPPED", "zero command"
    if abs(vy) > deadband:
        return vx, vy, 0.0, "PARALLEL", "angular.z ignored in parallel mode"
    if abs(vx) <= deadband and abs(wz) > deadband:
        return 0.0, 0.0, wz, "SPINNING", "pure angular command"
    if abs(vx) <= deadband:
        return 0.0, 0.0, 0.0, "STOPPED", "below command deadband"
    if abs(wz) <= deadband:
        return vx, 0.0, 0.0, "DUAL_ACKERMANN", "straight Ackermann motion"
    requested_radius = abs(vx / wz)
    if requested_radius < min_turn_radius:
        wz = math.copysign(abs(vx) / min_turn_radius, wz)
        reason = "Ackermann curvature clamped to minimum radius"
    else:
        reason = "feasible Ackermann arc"
    radius = abs(vx / wz)
    inner = math.copysign(
        min(math.atan((wheelbase * 0.5) / radius), math.radians(40.0)), vx * wz)
    inner_abs = abs(inner)
    central = math.atan2(
        wheelbase * math.sin(inner_abs),
        wheelbase * math.cos(inner_abs) + track * math.sin(inner_abs))
    central = math.copysign(central, inner)
    executed_wz = 2.0 * vx * math.sin(central) / wheelbase
    return vx, 0.0, executed_wz, "DUAL_ACKERMANN", reason


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class CommandSemanticsMPPI:
    """Sample behavior-conditioned control sequences and score executable rollouts."""
    BEHAVIORS = ("ACKERMANN", "PARALLEL", "SPIN_THEN_DRIVE")

    def __init__(self, config=None, method="command_aware"):
        self.cfg = config or MPPIConfig()
        if method not in ("ideal_twist", "posthoc_adapter", "command_aware"):
            raise ValueError(f"unknown rollout method: {method}")
        self.method = method
        self.rng = random.Random(self.cfg.random_seed)
        self.last_command_mode = None

    def _nominal(self, behavior, state, goal, goal_yaw=None):
        x, y, yaw = state
        dx, dy = goal[0] - x, goal[1] - y
        distance = math.hypot(dx, dy)
        error = _wrap(math.atan2(dy, dx) - yaw)
        if goal_yaw is not None and distance <= 0.42:
            error = _wrap(goal_yaw - yaw)
        v = min(self.cfg.max_speed, max(0.08, 0.65 * distance))
        if behavior == "ACKERMANN":
            # Respect the curvature envelope in the nominal sequence.
            wz = max(-v / self.cfg.min_turn_radius,
                     min(v / self.cfg.min_turn_radius, 1.8 * error))
            return v, 0.0, wz
        if behavior == "PARALLEL":
            return v * math.cos(error), v * math.sin(error), 0.0
        if abs(error) > 0.18:
            return 0.0, 0.0, max(-0.9, min(0.9, 2.0 * error))
        return v, 0.0, 0.0

    def _scan_points_world(self, state, scan):
        x, y, yaw = state
        points = []
        if scan is None:
            return points
        ranges, a0, da = scan
        sx = x + self.cfg.lidar_x * math.cos(yaw)
        sy = y + self.cfg.lidar_x * math.sin(yaw)
        cp = math.cos(self.cfg.lidar_pitch)
        stride = max(1, len(ranges) // 100)
        for i in range(0, len(ranges), stride):
            r = float(ranges[i])
            if not math.isfinite(r) or r < 0.12 or r > 5.0:
                continue
            a = a0 + i * da
            bx, by = r * cp * math.cos(a), r * math.sin(a)
            points.append((sx + math.cos(yaw) * bx - math.sin(yaw) * by,
                           sy + math.sin(yaw) * bx + math.cos(yaw) * by))
        return points

    def _apply_model(self, u):
        if self.method in ("ideal_twist", "posthoc_adapter"):
            return u[0], u[1], u[2], "IDEAL_TWIST", "continuous planar rollout"
        return ranger_command(*u, min_turn_radius=self.cfg.min_turn_radius,
                              wheelbase=self.cfg.wheelbase)

    def _clearance_cost(self, x, y, yaw, points):
        if len(points) == 0:
            return 0.0, 20.0
        c, s = math.cos(yaw), math.sin(yaw)
        half_l, half_w = self.cfg.body_length * 0.5, self.cfg.body_width * 0.5
        best = float("inf")
        penalty = 0.0
        for px, py in points:
            dx, dy = px - x, py - y
            bx, by = c * dx + s * dy, -s * dx + c * dy
            ox = max(abs(bx) - half_l, 0.0)
            oy = max(abs(by) - half_w, 0.0)
            d = math.hypot(ox, oy)
            if d < best:
                best = d
            if d < 0.55:
                penalty += (0.55 - d) ** 2 * 3.0
        if best < 0.035:
            penalty += 500.0
        return penalty, best

    def _rollout(self, start, goal, points, behavior, noise_scale, initial_mode=None,
                 goal_yaw=None):
        x, y, yaw = start
        total = 0.0
        min_clearance = float("inf")
        first = None
        sequence = []
        # Include the transition from the executing mode into this horizon.
        # The fixed ideal/post-hoc baseline keeps its original cost model.
        prev_mode = (initial_mode if self.method == "command_aware" and
                     self.cfg.account_initial_mode_switch else None)
        mode_switches = 0
        for k in range(self.cfg.horizon):
            nominal = self._nominal(behavior, (x, y, yaw), goal, goal_yaw)
            decay = 0.65 + 0.35 * (k / max(1, self.cfg.horizon - 1))
            u = (max(-self.cfg.max_speed, min(self.cfg.max_speed,
                 nominal[0] + self.rng.gauss(0.0, noise_scale * decay))),
                 max(-self.cfg.max_speed, min(self.cfg.max_speed,
                 nominal[1] + self.rng.gauss(0.0, noise_scale * decay))),
                 max(-1.2, min(1.2,
                 nominal[2] + self.rng.gauss(0.0, noise_scale * 1.5 * decay))))
            if behavior == "ACKERMANN":
                u = (u[0], 0.0, u[2])
            elif behavior == "PARALLEL":
                u = (u[0], u[1], 0.0)
            elif abs(nominal[2]) > 0.18:
                u = (0.0, 0.0, u[2])
            else:
                u = (u[0], 0.0, 0.0)
            if not self.cfg.allow_reverse:
                u = (max(0.0, u[0]), u[1], u[2])
            vx, vy, wz, mode, _ = self._apply_model(u)
            if first is None:
                first = u
            sequence.append(u)
            if prev_mode is not None and mode != prev_mode:
                mode_switches += 1
                total += 0.18
            prev_mode = mode
            c, s = math.cos(yaw), math.sin(yaw)
            x += (c * vx - s * vy) * self.cfg.dt
            y += (s * vx + c * vy) * self.cfg.dt
            yaw = _wrap(yaw + wz * self.cfg.dt)
            obs_cost, clearance = self._clearance_cost(x, y, yaw, points)
            total += obs_cost * self.cfg.dt
            min_clearance = min(min_clearance, clearance)
        goal_dist = math.hypot(goal[0] - x, goal[1] - y)
        total += 7.0 * goal_dist + 0.10 * mode_switches
        if goal_yaw is not None:
            total += 4.0 * abs(_wrap(goal_yaw - yaw))
        return total, first, min_clearance, mode_switches, sequence

    def plan(self, state, goal, scan=None, current_mode=None, goal_yaw=None):
        started = time.perf_counter()
        state = tuple(map(float, state))
        goal = tuple(map(float, goal))
        points = self._scan_points_world(state, scan)
        initial_mode = current_mode or self.last_command_mode
        candidates = []
        for behavior in self.BEHAVIORS:
            group = []
            for _ in range(self.cfg.samples_per_behavior):
                scale = 0.16 if behavior != "SPIN_THEN_DRIVE" else 0.12
                result = self._rollout(state, goal, points, behavior, scale, initial_mode,
                                       goal_yaw)
                group.append((*result, behavior))
            min_cost = min(c[0] for c in group)
            weights = [math.exp(-min(60.0, (c[0] - min_cost) /
                                    max(1e-6, self.cfg.temperature))) for c in group]
            z = sum(weights) or 1.0
            action = tuple(sum(w * c[4][0][i] for w, c in zip(weights, group)) / z
                           for i in range(3))
            expected_cost = sum(w * c[0] for w, c in zip(weights, group)) / z
            expected_clearance = sum(w * c[2] for w, c in zip(weights, group)) / z
            expected_switches = sum(w * c[3] for w, c in zip(weights, group)) / z
            candidates.append({
                "behavior": behavior,
                "action": action,
                "score": min_cost,
                "weighted_score": expected_cost,
                "min_clearance_m": expected_clearance,
                "mode_switches": expected_switches,
                "sample_count": len(group),
            })
        best = min(candidates, key=lambda c: c["weighted_score"])
        command_mode = ranger_command(*best["action"],
                                     min_turn_radius=self.cfg.min_turn_radius,
                                     wheelbase=self.cfg.wheelbase)[3]
        if command_mode != "STOPPED":
            self.last_command_mode = command_mode
        return {"action": best["action"], "selected_behavior": best["behavior"],
                "initial_mode": initial_mode, "command_mode": command_mode,
                "initial_mode_switch_accounted": self.method == "command_aware" and
                    self.cfg.account_initial_mode_switch,
                "allow_reverse": self.cfg.allow_reverse, "goal_yaw": goal_yaw,
                "predicted_score": best["weighted_score"],
                "predicted_min_clearance_m": best["min_clearance_m"],
                "predicted_mode_switches": best["mode_switches"],
                "method": self.method,
                "candidate_count": sum(c["sample_count"] for c in candidates),
                "planning_time_ms": 1000.0 * (time.perf_counter() - started),
                "candidate_summary": candidates}

