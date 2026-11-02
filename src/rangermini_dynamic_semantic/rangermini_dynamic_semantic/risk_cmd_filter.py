#!/usr/bin/env python3
"""
Goal/topology-aware shared-control safety filter.

v3.4 change:
The task is not "avoid the wall at any cost"; it is "pass through the doorway".
Therefore the filter adds a doorway/corridor topological constraint and a centerline
guidance cost. This prevents the optimizer from selecting a silly path that goes
around the outside of the wall just because it has larger clearance.
"""
import math
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class RectObstacle:
    xmin: float
    xmax: float
    ymin: float
    ymax: float


def yaw_from_quat(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def dist_rect(x, y, r):
    dx = max(r.xmin - x, 0.0, x - r.xmax)
    dy = max(r.ymin - y, 0.0, y - r.ymax)
    return math.hypot(dx, dy)


class RiskCmdFilter(Node):
    def __init__(self):
        super().__init__("risk_cmd_filter")

        defaults = {
            "robot_radius": 0.34,
            "safe_distance": 0.16,
            "risk_sigma": 0.24,
            "horizon": 1.35,
            "dt": 0.10,
            "lambda_risk": 0.65,
            "w_vx": 0.8,
            "w_vy": 1.1,
            "w_wz": 0.4,
            "w_center": 5.0,
            "w_progress": 1.2,
            "max_vx": 0.50,
            "max_vy": 0.35,
            "max_wz": 0.8,
            "doorway_width": 1.10,
            "wall_ymin": 1.10,
            "wall_ymax": 1.45,
            "world_half_width": 3.20,
            "goal_y": 2.75,
            "door_center_x": 0.0,
            "corridor_half_width": 0.58,
            "display_stride": 4,
        }
        for k, v in defaults.items():
            self.declare_parameter(k, v)

        self.cmd_h = Twist()
        self.x = 0.0
        self.y = -1.65
        self.yaw = math.radians(90.0)

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel_safe", 10)
        self.min_pub = self.create_publisher(Float32, "/min_distance", 10)
        self.int_pub = self.create_publisher(Float32, "/intervention_score", 10)
        self.risk_pub = self.create_publisher(Float32, "/risk_score", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/debug/candidate_paths", 10)

        self.create_subscription(Twist, "/cmd_vel_raw", self.on_cmd, 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.timer = self.create_timer(0.05, self.step)

    def p(self, name):
        return self.get_parameter(name).value

    def on_cmd(self, msg):
        self.cmd_h = msg

    def on_odom(self, msg):
        self.x = float(msg.pose.pose.position.x)
        self.y = float(msg.pose.pose.position.y)
        self.yaw = yaw_from_quat(msg.pose.pose.orientation)

    def obstacles(self):
        w = float(self.p("doorway_width"))
        left = -w / 2.0
        right = w / 2.0
        y0 = float(self.p("wall_ymin"))
        y1 = float(self.p("wall_ymax"))
        hw = float(self.p("world_half_width"))
        return [
            RectObstacle(-hw, left, y0, y1),
            RectObstacle(right, hw, y0, y1),
            RectObstacle(left - 0.12, left, y0 - 0.18, y1 + 0.18),
            RectObstacle(right, right + 0.12, y0 - 0.18, y1 + 0.18),
        ]

    def clearance(self, x, y):
        return min(dist_rect(x, y, o) for o in self.obstacles()) - float(self.p("robot_radius"))

    def rollout(self, u):
        vx, vy, wz = u
        x, y, yaw = self.x, self.y, self.yaw
        dt = float(self.p("dt"))
        steps = max(2, int(float(self.p("horizon")) / dt))
        pts = []
        for _ in range(steps):
            x += (math.cos(yaw) * vx - math.sin(yaw) * vy) * dt
            y += (math.sin(yaw) * vx + math.cos(yaw) * vy) * dt
            yaw += wz * dt
            pts.append((x, y, yaw))
        return pts

    def topological_violation(self, traj):
        """Reject paths that try to bypass the wall outside the doorway corridor."""
        center = float(self.p("door_center_x"))
        corridor_half = float(self.p("corridor_half_width"))
        # From the approach zone to the gate, require the rollout to stay near the door corridor.
        # This turns the task from "avoid obstacle" into "pass through the doorway".
        for x, y, _ in traj:
            if y <= float(self.p("wall_ymax")) + 0.35:
                if abs(x - center) > corridor_half:
                    return True
        return False

    def evaluate(self, traj):
        sigma = float(self.p("risk_sigma"))
        min_clearance = 999.0
        risk = 0.0
        collision = False
        for x, y, _ in traj:
            c = self.clearance(x, y)
            min_clearance = min(min_clearance, c)
            if c < 0.0:
                collision = True
                risk += 100.0
            else:
                risk += math.exp(-c / max(sigma, 1e-6))
        risk /= max(len(traj), 1)
        topo = self.topological_violation(traj)
        return risk, min_clearance, collision, topo

    def candidates(self, uh):
        vx_h, vy_h, wz_h = uh
        max_vx = float(self.p("max_vx"))
        max_vy = float(self.p("max_vy"))
        max_wz = float(self.p("max_wz"))

        # Body vy controls world lateral motion. With yaw≈90°, world dx≈-vy.
        # A feedback vy around x pulls the robot back to the doorway center.
        center = float(self.p("door_center_x"))
        vy_center = max(min(1.0 * (self.x - center), max_vy), -max_vy)

        out = []
        base_vx = max(0.20, min(max(vx_h, 0.25), max_vx))
        vx_samples = sorted(set([0.18, base_vx, min(base_vx + 0.08, max_vx)]))
        vy_samples = sorted(set([
            max(min(vy_h, max_vy), -max_vy),
            0.0,
            vy_center,
            max(min(vy_center + 0.08, max_vy), -max_vy),
            max(min(vy_center - 0.08, max_vy), -max_vy),
        ]))
        wz_samples = [-0.15, 0.0, 0.15]

        for vx in vx_samples:
            for vy in vy_samples:
                for wz in wz_samples:
                    out.append((
                        max(0.0, min(float(vx), max_vx)),
                        max(-max_vy, min(float(vy), max_vy)),
                        max(-max_wz, min(float(wz), max_wz)),
                    ))
        out.append((0.0, 0.0, 0.0))
        return out

    def cost(self, u, uh, risk, traj):
        center = float(self.p("door_center_x"))
        center_cost = sum((x - center) ** 2 for x, _, _ in traj) / max(len(traj), 1)
        progress = max(traj[-1][1] - self.y, 0.0) if traj else 0.0
        # prefer forward progress through the gate; penalize leaving centerline
        return (
            float(self.p("w_vx")) * (u[0] - uh[0]) ** 2
            + float(self.p("w_vy")) * (u[1] - uh[1]) ** 2
            + float(self.p("w_wz")) * (u[2] - uh[2]) ** 2
            + float(self.p("lambda_risk")) * risk
            + float(self.p("w_center")) * center_cost
            - float(self.p("w_progress")) * progress
        )

    def step(self):
        uh = (
            float(self.cmd_h.linear.x),
            float(self.cmd_h.linear.y),
            float(self.cmd_h.angular.z),
        )

        infos = []
        best = None
        best_cost = 1e9
        best_risk = 0.0
        best_clearance = 999.0
        best_traj = []

        if self.y < float(self.p("goal_y")):
            for i, u in enumerate(self.candidates(uh)):
                traj = self.rollout(u)
                risk, clear, collision, topo = self.evaluate(traj)
                rejected = collision or topo or clear < float(self.p("safe_distance"))
                if not rejected:
                    c = self.cost(u, uh, risk, traj)
                    if c < best_cost:
                        best = u
                        best_cost = c
                        best_risk = risk
                        best_clearance = clear
                        best_traj = traj
                infos.append((i, traj, rejected))

        if best is None:
            # If all moving candidates are rejected, creep forward slowly at the doorway center
            # rather than turning around the wall.
            best = (0.05, 0.0, 0.0)
            best_traj = self.rollout(best)
            best_risk, best_clearance, _, _ = self.evaluate(best_traj)

        self.publish(best, uh, best_risk, best_clearance, infos, best_traj)

    def publish(self, best, uh, risk, clearance, infos, selected):
        msg = Twist()
        msg.linear.x = float(best[0])
        msg.linear.y = float(best[1])
        msg.angular.z = float(best[2])
        self.cmd_pub.publish(msg)

        nh = math.sqrt(uh[0] ** 2 + uh[1] ** 2 + uh[2] ** 2)
        nd = math.sqrt((best[0] - uh[0]) ** 2 + (best[1] - uh[1]) ** 2 + (best[2] - uh[2]) ** 2)

        self.min_pub.publish(Float32(data=float(clearance)))
        self.int_pub.publish(Float32(data=float(nd / (nh + 1e-4))))
        self.risk_pub.publish(Float32(data=float(risk)))
        self.markers(infos, selected, uh, best)

    def line(self, arr, ns, idx, traj, rgba, width):
        m = Marker()
        m.header.frame_id = "odom"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = int(idx)
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.lifetime = Duration(sec=0, nanosec=350000000)
        m.scale.x = float(width)
        m.color.r = float(rgba[0])
        m.color.g = float(rgba[1])
        m.color.b = float(rgba[2])
        m.color.a = float(rgba[3])
        for x, y, _ in traj:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.08
            m.points.append(p)
        arr.markers.append(m)

    def markers(self, infos, selected, uh, best):
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        stride = max(1, int(self.p("display_stride")))
        shown = 0
        for idx, traj, rejected in infos:
            if idx % stride != 0:
                continue
            if shown > 24:
                break
            if rejected:
                self.line(arr, "rejected_rollouts", idx, traj, (1.0, 0.0, 0.0, 0.65), 0.018)
            else:
                self.line(arr, "candidate_rollouts", idx, traj, (0.45, 0.45, 0.45, 0.28), 0.012)
            shown += 1

        self.line(arr, "selected_rollout", 10000, selected, (0.0, 0.25, 1.0, 0.95), 0.045)
        self.line(arr, "human_input_arrow", 11000, self.rollout((uh[0] * 1.2, uh[1] * 1.2, uh[2])), (1.0, 0.75, 0.0, 0.9), 0.05)
        self.line(arr, "safe_output_arrow", 12000, self.rollout((best[0] * 1.2, best[1] * 1.2, best[2])), (0.0, 0.85, 0.1, 0.9), 0.05)

        self.marker_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    n = RiskCmdFilter()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
