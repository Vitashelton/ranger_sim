"""Planar four-wheel-steering kinematics used by the RangerMini Gazebo adapter.

The model follows the rigid-body/pure-rolling assumptions used in the supplied
4WS reference: wheel-point velocity is v_i = [vx-w*y_i, vy+w*x_i]. It models
steering geometry and mode-change dwell, not tire forces or actuator torque.
"""
from dataclasses import dataclass
import math
from . import native_kernels


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def canonical_motion_mode(mode):
    """Treat archived pure-lateral labels as parallel-steering execution."""
    return "PARALLEL" if mode == "SIDE_SLIP" else mode


def _solve3(a, b):
    """Small pivoted Gaussian solve for a 3x3 normal-equation system."""
    m = [list(a[i]) + [b[i]] for i in range(3)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            return [0.0, 0.0, 0.0]
        m[col], m[pivot] = m[pivot], m[col]
        scale = m[col][col]
        m[col] = [v / scale for v in m[col]]
        for row in range(3):
            if row == col:
                continue
            factor = m[row][col]
            m[row] = [m[row][j] - factor * m[col][j] for j in range(4)]
    return [m[i][3] for i in range(3)]


@dataclass
class Ranger4WSKinematics:
    wheelbase: float = 0.494
    track: float = 0.364
    max_speed: float = 1.5
    max_steer_rate: float = 2.5
    max_ackermann_angle: float = math.radians(40.0)
    max_parallel_angle: float = math.radians(90.0)
    deadband: float = 1e-4
    use_native: bool = True

    def __post_init__(self):
        self.angles = [0.0] * 4
        self.mode = "STOPPED"
        self.transition_pending = False

    @property
    def wheel_positions(self):
        return ((self.wheelbase/2, self.track/2),
                (self.wheelbase/2, -self.track/2),
                (-self.wheelbase/2, self.track/2),
                (-self.wheelbase/2, -self.track/2))

    def _requested_mode(self, vx, vy, wz):
        if math.hypot(vx, vy) <= self.deadband and abs(wz) <= self.deadband:
            return "STOPPED"
        # Pure lateral motion is the +/-90-degree boundary of parallel
        # steering, not a separate execution mode on the V2 command interface.
        if abs(vy) > self.deadband:
            return "PARALLEL"
        if abs(vx) <= self.deadband and abs(wz) > self.deadband:
            return "SPINNING"
        return "DUAL_ACKERMANN"

    def _target_wheels(self, vx, vy, wz, mode):
        targets, speeds = [], []
        for x, y in self.wheel_positions:
            wx, wy = vx - wz*y, vy + wz*x
            angle = math.atan2(wy, wx) if math.hypot(wx, wy) > self.deadband else self.angles[len(targets)]
            # A wheel can reverse instead of turning through pi.
            alt = _wrap(angle + math.pi)
            if abs(_wrap(alt-self.angles[len(targets)])) < abs(_wrap(angle-self.angles[len(targets)])):
                angle, speed = alt, -math.hypot(wx, wy)
            else:
                speed = math.hypot(wx, wy)
            limit = (self.max_parallel_angle if mode in ("PARALLEL", "SIDE_SLIP")
                     else self.max_ackermann_angle if mode == "DUAL_ACKERMANN"
                     else math.pi/2)
            angle = max(-limit, min(limit, angle))
            targets.append(angle); speeds.append(speed)
        return targets, speeds

    def _twist_from_wheels(self, speeds):
        # For the symmetric rectangular wheel layout, A^T A is diagonal:
        # diag(4, 4, sum_i(x_i^2+y_i^2)). This is the same least-squares
        # fit as the general normal-equation solve, without rebuilding A.
        vx_sum = vy_sum = yaw_sum = radius_sum = 0.0
        for (x, y), angle, speed in zip(self.wheel_positions, self.angles, speeds):
            vxw, vyw = speed*math.cos(angle), speed*math.sin(angle)
            vx_sum += vxw
            vy_sum += vyw
            yaw_sum += -y*vxw+x*vyw
            radius_sum += x*x+y*y
        if radius_sum < 1e-12:
            return [0.0, 0.0, 0.0]
        return [vx_sum/4.0, vy_sum/4.0, yaw_sum/radius_sum]

    def step(self, vx, vy, wz, dt):
        if self.use_native:
            return native_kernels.wheel_step(self, vx, vy, wz, dt)
        return self._step_python(vx, vy, wz, dt)

    def _step_python(self, vx, vy, wz, dt):
        """Return realized body twist, mode, steering angles, and transition flag."""
        self.mode = canonical_motion_mode(self.mode)
        speed = math.hypot(vx, vy)
        if speed > self.max_speed:
            scale = self.max_speed/speed
            vx, vy = vx*scale, vy*scale
        mode = self._requested_mode(vx, vy, wz)
        targets, _ = self._target_wheels(vx, vy, wz, mode)
        changed = mode != self.mode and mode != "STOPPED"
        if changed:
            self.transition_pending = True
        max_step = self.max_steer_rate*max(0.0, dt)
        aligned = True
        for i, target in enumerate(targets):
            err = _wrap(target-self.angles[i])
            if abs(err) > 0.025:
                aligned = False
            self.angles[i] = _wrap(self.angles[i] + max(-max_step, min(max_step, err)))
        # Like the reference planner, hold the chassis while changing modes or
        # while the steering modules are being positioned for the new mode.
        if mode == "STOPPED":
            self.mode = mode
            return (0.0, 0.0, 0.0), mode, tuple(self.angles), False
        if self.transition_pending:
            self.mode = mode
            if aligned:
                self.transition_pending = False
            return (0.0, 0.0, 0.0), mode, tuple(self.angles), True
        wheel_speeds = []
        for (x, y), angle in zip(self.wheel_positions, self.angles):
            wx, wy = vx-wz*y, vy+wz*x
            wheel_speeds.append(wx*math.cos(angle)+wy*math.sin(angle))
        wheel_speeds = [max(-self.max_speed, min(self.max_speed, s)) for s in wheel_speeds]
        twist = self._twist_from_wheels(wheel_speeds)
        peak = max((math.hypot(twist[0]-twist[2]*y, twist[1]+twist[2]*x)
                    for x, y in self.wheel_positions), default=0.0)
        if peak > self.max_speed:
            twist = [v*self.max_speed/peak for v in twist]
        self.mode = mode
        return tuple(twist), mode, tuple(self.angles), False
