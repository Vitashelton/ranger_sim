"""Task-conditioned sparse landmark placement using expected pose information."""
from dataclasses import dataclass
import json
import math


@dataclass(frozen=True)
class Landmark:
    id: int
    name: str
    x: float
    y: float
    yaw: float
    quality: float


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def load_problem(path):
    with open(path, "r", encoding="utf-8") as stream:
        cfg = json.load(stream)
    landmarks = {
        int(v["id"]): Landmark(
            int(v["id"]), str(v["name"]), float(v["x"]), float(v["y"]),
            math.radians(float(v["yaw_deg"])), float(v.get("quality", 1.0)))
        for v in cfg["candidates"]
    }
    return cfg, landmarks


def expected_information(pose, landmark, max_range, half_fov):
    """Return detection probability and 2-D translation Fisher information.

    The observation is a tag-relative range/bearing measurement. Its covariance
    increases with range and oblique tag incidence; therefore a landmark is
    useful only when it provides geometrically informative observations.
    """
    x, y, yaw = [float(v) for v in pose]
    dx, dy = landmark.x - x, landmark.y - y
    r = math.hypot(dx, dy)
    if r < 0.25 or r > max_range:
        return 0.0, (0.0, 0.0, 0.0)
    bearing = wrap(math.atan2(dy, dx) - yaw)
    if abs(bearing) > half_fov:
        return 0.0, (0.0, 0.0, 0.0)
    # Tag normal should point approximately toward the observing robot.
    incidence = abs(wrap(math.atan2(y - landmark.y, x - landmark.x) - landmark.yaw))
    incidence_cos = max(0.0, math.cos(incidence))
    if incidence_cos <= 0.05:
        return 0.0, (0.0, 0.0, 0.0)

    p = landmark.quality * math.exp(-0.18 * r) * math.cos(
        0.5 * math.pi * abs(bearing) / half_fov) * incidence_cos
    p = max(0.0, min(0.995, p))
    sigma_r = 0.025 + 0.012 * r * r / max(incidence_cos, 0.2)
    sigma_b = math.radians(0.8 + 0.45 * r) / max(incidence_cos, 0.2)
    c, s = dx / r, dy / r
    wr, wb = 1.0 / (sigma_r * sigma_r), 1.0 / (sigma_b * sigma_b)
    # J=[[-c,-s],[s/r,-c/r]]; return symmetric Ixx,Ixy,Iyy.
    ixx = p * (wr * c * c + wb * s * s / (r * r))
    ixy = p * (wr * c * s - wb * c * s / (r * r))
    iyy = p * (wr * s * s + wb * c * c / (r * r))
    return p, (ixx, ixy, iyy)


def pose_logdet(pose, selected, landmarks, cfg):
    prior = float(cfg.get("prior_information", 0.25))
    a, b, d = prior, 0.0, prior
    for lid in selected:
        _, info = expected_information(
            pose, landmarks[lid], float(cfg["max_detection_range"]),
            math.radians(float(cfg["camera_half_fov_deg"])))
        a += info[0]
        b += info[1]
        d += info[2]
    return math.log(max(a * d - b * b, 1e-12))


def route_utility(route, selected, landmarks, cfg):
    return sum(pose_logdet(p, selected, landmarks, cfg) for p in route) / max(len(route), 1)


def greedy_select(route, landmarks, cfg, budget=None):
    budget = int(cfg["budget"] if budget is None else budget)
    selected = set()
    while len(selected) < min(budget, len(landmarks)):
        base = route_utility(route, selected, landmarks, cfg)
        best = max(
            (lid for lid in landmarks if lid not in selected),
            key=lambda lid: route_utility(route, selected | {lid}, landmarks, cfg) - base)
        selected.add(best)
    return selected


def route_metrics(route, selected, landmarks, cfg):
    probs, pebs = [], []
    prior = float(cfg.get("prior_information", 0.25))
    for pose in route:
        miss = 1.0
        a, b, d = prior, 0.0, prior
        for lid in selected:
            p, info = expected_information(
                pose, landmarks[lid], float(cfg["max_detection_range"]),
                math.radians(float(cfg["camera_half_fov_deg"])))
            miss *= 1.0 - p
            a += info[0]; b += info[1]; d += info[2]
        det = max(a * d - b * b, 1e-12)
        # sqrt(trace(inv(FIM))) is the 2-D position error bound proxy.
        pebs.append(math.sqrt(max((a + d) / det, 0.0)))
        probs.append(1.0 - miss)
    return {
        "mean_detection_probability": sum(probs) / max(len(probs), 1),
        "mean_position_error_bound_m": sum(pebs) / max(len(pebs), 1),
        "route_information": route_utility(route, selected, landmarks, cfg),
    }
