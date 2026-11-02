"""Task-scoped asynchronous evidence and a planar camera observation model.

No simulator state, ROS imports, or model calls enter this module. Registered
target priors are distinct from evidence confirming the target's identity.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TaskScope:
    map_id: str
    version: int
    node_id: str
    region: str
    target_id: str


@dataclass(frozen=True)
class Evidence:
    request_id: str
    scope: TaskScope
    captured_at: float
    received_at: float
    capture_pose: tuple
    value: str
    geometry_consistent: bool


class EvidenceStore:
    """Reject envelopes before changing any task fact; query by exact scope."""
    VALUES = {"confirmed", "absent", "uncertain"}

    def __init__(self, registered_regions, ttl_s=2.0):
        self.regions = frozenset(registered_regions)
        if not math.isfinite(ttl_s) or ttl_s <= 0:
            raise ValueError("positive finite evidence TTL required")
        self.ttl = ttl_s
        self.records = {}
        self.seen = set()
        self.requests = {}

    def register_request(self, request_id, scope, captured_at, capture_pose):
        key = (scope, request_id)
        if key in self.requests or len(capture_pose) != 3:
            raise ValueError("duplicate request or invalid capture pose")
        self.requests[key] = (float(captured_at), tuple(capture_pose))

    def admit(self, evidence, active_scope, now):
        e = evidence
        reason = None
        if e.scope != active_scope:
            reason = "scope_mismatch"
        elif e.scope.region not in self.regions:
            reason = "unregistered_region"
        elif e.value not in self.VALUES:
            reason = "invalid_value"
        elif not all(math.isfinite(x) for x in (*e.capture_pose, e.captured_at,
                                               e.received_at, now)):
            reason = "nonfinite_geometry_or_time"
        elif len(e.capture_pose) != 3 or not e.geometry_consistent:
            reason = "geometry_veto"
        elif (e.scope, e.request_id) not in self.requests:
            reason = "unrequested_evidence"
        elif (abs(self.requests[(e.scope, e.request_id)][0]-e.captured_at) > 1e-6 or
              any(abs(a-b) > 1e-6 for a, b in zip(
                  self.requests[(e.scope, e.request_id)][1], e.capture_pose))):
            reason = "capture_envelope_mismatch"
        elif not e.captured_at <= e.received_at <= now:
            reason = "invalid_time_order"
        elif now - e.captured_at > self.ttl:
            reason = "expired"
        elif (e.scope, e.request_id) in self.seen:
            reason = "duplicate"
        if reason:
            return False, reason
        self.seen.add((e.scope, e.request_id))
        self.records.setdefault(e.scope, []).append(e)
        return True, self.state(active_scope, now)

    def state(self, scope, now):
        records = [e for e in self.records.get(scope, ())
                   if 0 <= now - e.captured_at <= self.ttl]
        values = {e.value for e in records if e.value != "uncertain"}
        if len(values) > 1:
            return "conflicting"
        return next(iter(values)) if values else "unresolved"


@dataclass
class QueryBudget:
    """Reserve worst-case resources on dispatch, settle actual cost on return."""
    query_limit: int
    byte_limit: int
    wait_limit_s: float
    queries: int = 0
    image_bytes: int = 0
    wait_s: float = 0.0

    def __post_init__(self):
        if min(self.query_limit, self.byte_limit, self.wait_limit_s) < 0:
            raise ValueError("budgets must be nonnegative")
        self.pending = {}

    def reserve(self, request_id, nbytes, wait_cap_s):
        if request_id in self.pending or nbytes < 0 or wait_cap_s < 0:
            return False
        reserved_wait = sum(self.pending.values())
        if (self.queries + 1 > self.query_limit or
                self.image_bytes + nbytes > self.byte_limit or
                self.wait_s + reserved_wait + wait_cap_s > self.wait_limit_s):
            return False
        self.queries += 1
        self.image_bytes += nbytes
        self.pending[request_id] = wait_cap_s
        return True

    def settle(self, request_id, elapsed_s):
        if request_id not in self.pending:
            raise ValueError("unknown or already settled request")
        cap = self.pending.pop(request_id)
        self.wait_s += min(cap, max(0.0, elapsed_s))
        return elapsed_s <= cap


@dataclass(frozen=True)
class ViewConfig:
    camera_x: float = 0.34
    camera_z: float = 0.43
    width: int = 640
    height: int = 480
    horizontal_fov: float = 1.396
    min_depth: float = 0.15
    depth_scale: float = 0.05
    boundary_scale_px: float = 12.0
    min_range: float = 0.5
    max_range: float = 2.5
    range_scale: float = 0.4
    cell_size: float = 0.10
    ray_radius: float = 0.10
    endpoint_exclusion: float = 0.15
    occlusion_beta: float = 2.0


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))


def project_target(pose, target, cfg=None):
    """World point to optical image (u,v,depth); None behind the camera."""
    c = cfg or ViewConfig()
    x, y, yaw = pose
    co, si = math.cos(yaw), math.sin(yaw)
    dx, dy = target[0]-x-c.camera_x*co, target[1]-y-c.camera_x*si
    depth, left = co*dx+si*dy, -si*dx+co*dy
    if depth <= 0:
        return None
    focal = c.width/(2*math.tan(c.horizontal_fov/2))
    return (c.width/2-focal*left/depth,
            c.height/2-focal*(target[2]-c.camera_z)/depth, depth)


def view_score(pose, hypotheses, occupied_points, cfg=None, omit_occlusion=False):
    """Evaluate F*R*O with normalized nonnegative prior weights.

    hypotheses: ((world_x, world_y, world_z, weight), ...).
    occupied_points: observed planar endpoint positions. This is explicitly a
    2-D simulation approximation, not native MID-360S volumetric ray tracing.
    An empty hypothesis set yields zero; behind-camera hypotheses yield zero.
    """
    c = cfg or ViewConfig()
    if min(c.depth_scale, c.boundary_scale_px, c.range_scale, c.cell_size) <= 0:
        raise ValueError("positive view scales required")
    if c.occlusion_beta < 0 or c.ray_radius < 0:
        raise ValueError("nonnegative occlusion parameters required")
    if any(not math.isfinite(v) for h in hypotheses for v in h):
        raise ValueError("finite target hypotheses required")
    if any(h[3] < 0 for h in hypotheses):
        raise ValueError("negative hypothesis weight")
    mass = sum(h[3] for h in hypotheses)
    if mass <= 0:
        return 0.0
    x, y, yaw = pose
    co, si = math.cos(yaw), math.sin(yaw)
    cx, cy = x + c.camera_x * co, y + c.camera_x * si
    focal = c.width / (2 * math.tan(c.horizontal_fov / 2))
    score = 0.0
    for tx, ty, tz, weight in hypotheses:
        dx, dy = tx - cx, ty - cy
        depth, left = co * dx + si * dy, -si * dx + co * dy
        if depth <= 0:
            continue
        u = c.width / 2 - focal * left / depth
        v = c.height / 2 - focal * (tz - c.camera_z) / depth
        fov = _sigmoid((depth - c.min_depth) / c.depth_scale)
        for boundary in (u, c.width-u, v, c.height-v):
            fov *= _sigmoid(boundary / c.boundary_scale_px)
        distance = math.sqrt(dx*dx + dy*dy + (tz-c.camera_z)**2)
        outside = max(c.min_range-distance, 0)**2 + max(distance-c.max_range, 0)**2
        useful_range = math.exp(-outside / c.range_scale**2)
        cells = set()
        length2 = dx*dx + dy*dy
        length = math.sqrt(length2)
        if length > c.endpoint_exclusion and not omit_occlusion:
            for ox, oy in occupied_points:
                fraction = ((ox-cx)*dx + (oy-cy)*dy) / length2
                if 0 < fraction < 1-c.endpoint_exclusion/length:
                    ray_dist = math.hypot(ox-cx-fraction*dx, oy-cy-fraction*dy)
                    if ray_dist <= c.ray_radius:
                        cells.add((math.floor(ox/c.cell_size), math.floor(oy/c.cell_size)))
        ray_clear = math.exp(-c.occlusion_beta * len(cells))
        score += weight/mass * fov * useful_range * ray_clear
    return min(1.0, max(0.0, score))
