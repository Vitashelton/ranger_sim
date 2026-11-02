#!/usr/bin/env python3
import math
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Twist, Point, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, String
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class RectObstacle:
    xmin: float
    xmax: float
    ymin: float
    ymax: float


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def angle_error(target, current):
    return math.atan2(math.sin(target - current), math.cos(target - current))


def dist_rect(x, y, r):
    dx = max(r.xmin - x, 0.0, x - r.xmax)
    dy = max(r.ymin - y, 0.0, y - r.ymax)
    return math.hypot(dx, dy)


class CorridorSemanticFilter(Node):
    """
    v4.3 route-aware semantic shared-control filter.

    The previous version could get stuck because the target door-front was too close to
    the upper wall and the safety margin made it infeasible. v4.3 uses a reachable
    door-front and a route-progress controller:
      raw input -> unsafe centerline
      route target -> above S2 -> Room 906 door-front
      risk filter -> reject unsafe candidate rollouts
    """
    def __init__(self):
        super().__init__("corridor_semantic_filter")
        defaults = {
            "robot_radius": 0.45,
            "safe_distance": 0.20,
            "risk_sigma": 0.32,
            "horizon": 1.80,
            "dt": 0.10,
            "lambda_risk": 0.75,
            "w_human": 0.35,
            "w_goal": 2.40,
            "w_route": 3.80,
            "w_target": 0.45,
            "w_center": 0.12,
            "w_progress": 1.40,
            "max_vx": 0.70,
            "max_vy": 0.50,
            "max_wz": 0.70,
            "corridor_ymin": 0.0,
            "corridor_ymax": 5.0,
            "corridor_center_y": 2.50,
            "goal_x": 13.35,
            "goal_y": 4.15,
            "route_xs": [4.80, 8.60, 10.80, 12.20, 13.35],
            "route_ys": [2.35, 2.85, 3.80, 3.95, 4.15],
            "display_stride": 6,
            "enabled": False,
        }
        for k, v in defaults.items():
            self.declare_parameter(k, v)

        self.cmd_h = Twist()
        self.x = 1.2
        self.y = 2.3
        self.yaw = 0.0
        self.have_odom = False
        self.goal_x = float(self.get_parameter("goal_x").value)
        self.goal_y = float(self.get_parameter("goal_y").value)
        self.target_room = "906"
        self.route_points = self.route_from_room(self.target_room)
        self.route_index = 0
        self.turnaround_active = False
        self.turnaround_yaw = 0.0
        self.enabled = bool(self.get_parameter("enabled").value)

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel_safe", 10)
        self.min_pub = self.create_publisher(Float32, "/min_distance", 10)
        self.int_pub = self.create_publisher(Float32, "/intervention_score", 10)
        self.risk_pub = self.create_publisher(Float32, "/risk_score", 10)
        self.raw_risk_pub = self.create_publisher(Float32, "/raw_human_risk_score", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/debug/candidate_paths", 10)

        self.create_subscription(Twist, "/cmd_vel_raw", self.on_cmd, 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.create_subscription(PoseStamped, "/semantic_target_pose", self.on_target_pose, 10)
        self.create_subscription(String, "/task_goal", self.on_task_goal, 10)
        self.create_subscription(String, "/task_control", self.on_task_control, 10)
        self.timer = self.create_timer(0.05, self.step)
        self.get_logger().info("v4.3 route-aware corridor semantic filter started.")

    def p(self, name):
        return self.get_parameter(name).value

    def on_cmd(self, msg):
        self.cmd_h = msg

    def on_odom(self, msg):
        self.x = float(msg.pose.pose.position.x)
        self.y = float(msg.pose.pose.position.y)
        self.yaw = yaw_from_quat(msg.pose.pose.orientation)
        self.have_odom = True

    def on_target_pose(self, msg):
        # Use semantic target, but clamp y to a reachable door-front inside corridor safety margin.
        self.goal_x = float(msg.pose.position.x)
        self.goal_y = min(float(msg.pose.position.y), 4.20)

    def route_from_room(self, room):
        routes = {
            "lobby": [(1.20, 2.30)],
            "902": [(2.2, 2.50), (3.00, 3.30), (3.00, 4.15)],
            "904": [(4.8, 2.35), (6.8, 2.60), (8.00, 3.35), (8.00, 4.15)],
            "906": [(4.8, 2.35), (8.6, 2.85), (10.8, 3.80), (12.2, 3.95), (13.35, 4.15)],
            "908": [(4.8, 2.35), (8.6, 2.85), (10.8, 3.80), (12.8, 3.95),
                    (14.0, 2.70), (16.5, 2.70), (18.0, 3.50), (18.55, 4.15)],
            "corridor_junction": [(4.8, 2.35), (8.6, 2.85), (10.8, 3.80),
                                  (11.0, 3.75)],
        }
        return list(routes.get(room, routes["906"]))

    def on_task_goal(self, msg):
        room = msg.data.strip()
        goals = {"lobby": (1.2, 2.3), "902": (3.0, 4.15),
                 "904": (8.0, 4.15), "906": (13.35, 4.15),
                 "908": (18.55, 4.15), "corridor_junction": (11.0, 3.75)}
        if room not in goals:
            self.get_logger().warn(f"Unknown task room: {room}")
            return
        self.target_room = room
        self.goal_x, self.goal_y = goals[room]
        route = self.route_from_room(room)
        # Runtime retasking uses Ranger Mini's spin mode instead of commanding
        # a long reverse motion. Determine "behind" in the current body frame.
        dx = self.goal_x - self.x
        dy = self.goal_y - self.y
        # A route is authored from the lobby.  When a search or patrol retasks
        # the robot from one door to a farther door, discard waypoints that are
        # already behind the robot; otherwise the controller first drives back
        # toward the lobby before making forward progress again.
        if dx >= 0.0:
            while len(route) > 1 and route[0][0] < self.x - 0.50:
                route.pop(0)
        local_dx = math.cos(self.yaw) * dx + math.sin(self.yaw) * dy
        if local_dx < -0.50:
            self.turnaround_active = True
            # Corridor travel is predominantly along world +/-X. Spin to the
            # required corridor heading, then resume forward / crab motion.
            self.turnaround_yaw = 0.0 if dx >= 0.0 else math.pi
            route = [(self.x, 2.50), (self.goal_x, 2.50), (self.goal_x, 3.35),
                     (self.goal_x, self.goal_y)]
            self.get_logger().info(
                f"Room {room} is behind the chassis: stop-and-spin retask "
                f"to yaw={self.turnaround_yaw:.2f} rad")
        else:
            self.turnaround_active = False
        self.route_points = route
        self.route_index = 0
        self.get_logger().info(f"Task switched to Room {room}: goal={goals[room]}")

    def on_task_control(self, msg):
        command = msg.data.strip().upper()
        if command == "START":
            if not self.have_odom:
                self.get_logger().error("START rejected: no unified /odom received")
                self.enabled = False
                self.cmd_pub.publish(Twist())
                return
            if not (0.0 < self.x < 22.0 and 0.20 < self.y < 4.80):
                self.get_logger().error(
                    f"START rejected: odom pose ({self.x:.2f}, {self.y:.2f}) "
                    "is outside the corridor world frame")
                self.enabled = False
                self.cmd_pub.publish(Twist())
                return
            self.enabled = True
            self.get_logger().info("Navigation task started")
        elif command == "STOP":
            self.enabled = False
            self.cmd_pub.publish(Twist())
            self.get_logger().warn("Navigation task stopped")

    def route(self):
        return self.route_points

    def active_goal(self):
        r = self.route()
        while self.route_index < len(r) - 1:
            rx, ry = r[self.route_index]
            if math.hypot(self.x - rx, self.y - ry) > 0.55:
                break
            self.route_index += 1
        if self.route_index < len(r):
            rx, ry = r[self.route_index]
            return rx, ry, f"{self.target_room}:wp_{self.route_index}"
        return self.goal_x, self.goal_y, "semantic_target"

    def obstacles(self):
        return [
            RectObstacle(5.85, 6.55, 0.95, 1.45),       # S1
            RectObstacle(11.275, 12.325, 2.275, 3.025), # S2
            RectObstacle(15.225, 15.775, 3.425, 3.975), # S3
        ]

    def corridor_violation(self, x, y):
        return (
            y <= float(self.p("corridor_ymin")) + 0.20
            or y >= float(self.p("corridor_ymax")) - 0.20
            or x <= 0.0
            or x >= 22.0
        )

    def clearance(self, x, y):
        obs_clear = min(dist_rect(x, y, o) for o in self.obstacles())
        wall_clear = min(y - float(self.p("corridor_ymin")),
                         float(self.p("corridor_ymax")) - y)
        return min(obs_clear, wall_clear) - float(self.p("robot_radius"))

    def rollout(self, u, horizon=None):
        vx, vy, wz = u
        x, y, yaw = self.x, self.y, self.yaw
        dt = float(self.p("dt"))
        h = float(self.p("horizon")) if horizon is None else float(horizon)
        steps = max(2, int(h / dt))
        pts = []
        for _ in range(steps):
            x += (math.cos(yaw) * vx - math.sin(yaw) * vy) * dt
            y += (math.sin(yaw) * vx + math.cos(yaw) * vy) * dt
            yaw += wz * dt
            pts.append((x, y, yaw))
        return pts

    def evaluate(self, traj):
        sigma = float(self.p("risk_sigma"))
        mn = 999.0
        risk = 0.0
        collision = False
        for x, y, _ in traj:
            if self.corridor_violation(x, y):
                collision = True
                risk += 60.0
                mn = min(mn, -0.1)
                continue
            c = self.clearance(x, y)
            mn = min(mn, c)
            if c < 0:
                collision = True
                risk += 100.0
            else:
                risk += math.exp(-c / max(sigma, 1e-6))
        return risk / max(len(traj), 1), mn, collision

    def candidates(self, uh):
        gx, gy, _ = self.active_goal()
        dx = gx - self.x
        dy = gy - self.y
        norm = max(math.hypot(dx, dy), 1e-6)
        # Generate only commands the Ranger driver can actually execute:
        # parallel translation, dual-Ackermann motion, spinning, and stop.
        local_dx = math.cos(self.yaw) * dx + math.sin(self.yaw) * dy
        local_dy = -math.sin(self.yaw) * dx + math.cos(self.yaw) * dy
        guide_angle = math.atan2(local_dy, local_dx)
        max_v = min(float(self.p("max_vx")), 0.35)
        speeds = [0.12, 0.22, 0.30, max_v]
        parallel_angles = [-1.20, -0.75, -0.35, 0.0, 0.35, 0.75, 1.20, guide_angle]
        out = []
        for speed in speeds:
            for angle in parallel_angles:
                angle = max(-1.50, min(1.50, angle))
                out.append((speed * math.cos(angle), speed * math.sin(angle), 0.0))
            for wz in [-0.70, -0.40, -0.18, 0.18, 0.40, 0.70]:
                # v/|w| >= 0.4764 keeps the command in dual-Ackermann mode.
                if speed / abs(wz) >= 0.4764:
                    out.append((speed, 0.0, wz))
        out.extend([(0.0, 0.0, -0.45), (0.0, 0.0, 0.45)])
        out.append((0.0, 0.0, 0.0))
        return out

    def route_distance_cost(self, traj):
        r = self.route()
        def segment_distance(px, py, a, b):
            ax, ay = a
            bx, by = b
            dx, dy = bx - ax, by - ay
            denom = dx * dx + dy * dy
            if denom < 1e-9:
                return math.hypot(px - ax, py - ay)
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
            return math.hypot(px - (ax + t * dx), py - (ay + t * dy))
        total = 0.0
        for x, y, _ in traj:
            if len(r) < 2:
                d = math.hypot(x - r[0][0], y - r[0][1])
            else:
                d = min(segment_distance(x, y, r[i], r[i + 1])
                        for i in range(len(r) - 1))
            total += d*d
        return total / max(len(traj), 1)

    def cost(self, u, uh, risk, clearance, traj):
        gx, gy, mode = self.active_goal()
        end_x, end_y, _ = traj[-1]
        local_goal_cost = math.hypot(end_x - gx, end_y - gy)
        final_goal_cost = math.hypot(end_x - self.goal_x, end_y - self.goal_y)
        center_y = float(self.p("corridor_center_y"))
        center_cost = sum((y - center_y) ** 2 for _, y, _ in traj) / max(len(traj), 1)
        route_cost = self.route_distance_cost(traj)
        before = math.hypot(self.x - gx, self.y - gy)
        progress = before - math.hypot(end_x - gx, end_y - gy)
        human_cost = (u[0] - uh[0]) ** 2 + (u[1] - uh[1]) ** 2 + 0.25 * (u[2] - uh[2]) ** 2
        clearance_bonus = max(clearance, 0.0)

        return (
            float(self.p("w_human")) * human_cost
            + float(self.p("lambda_risk")) * risk
            + float(self.p("w_goal")) * local_goal_cost
            + float(self.p("w_target")) * final_goal_cost
            + float(self.p("w_route")) * route_cost
            + float(self.p("w_center")) * center_cost
            - float(self.p("w_progress")) * progress
            - 0.25 * clearance_bonus
        )

    def guide_fallback(self):
        gx, gy, _ = self.active_goal()
        dx = gx - self.x
        dy = gy - self.y
        local_dx = math.cos(self.yaw) * dx + math.sin(self.yaw) * dy
        local_dy = -math.sin(self.yaw) * dx + math.cos(self.yaw) * dy
        if local_dx < -0.30:
            return (0.0, 0.0, 0.35 if local_dy >= 0.0 else -0.35)
        angle = max(-1.50, min(1.50, math.atan2(local_dy, local_dx)))
        return (0.18 * math.cos(angle), 0.18 * math.sin(angle), 0.0)

    def step(self):
        uh = (float(self.cmd_h.linear.x), float(self.cmd_h.linear.y), float(self.cmd_h.angular.z))

        if not self.enabled or not self.have_odom:
            self.cmd_pub.publish(Twist())
            return

        # A task switched to the opposite corridor direction is a discrete
        # Ranger motion-mode transition: stop translation, spin, settle, then
        # hand control back to the route follower. Never back down the corridor.
        if self.turnaround_active:
            err = angle_error(self.turnaround_yaw, self.yaw)
            if abs(err) > 0.10:
                wz = math.copysign(min(0.45, max(0.20, 0.8 * abs(err))), err)
                spin = (0.0, 0.0, wz)
                traj = self.rollout(spin, horizon=0.8)
                risk, clearance, _ = self.evaluate(traj)
                self.publish(spin, uh, risk, clearance, [], traj)
                return
            self.turnaround_active = False
            self.cmd_pub.publish(Twist())
            self.get_logger().info("Retask turnaround complete; route execution resumed")
            return

        if math.hypot(self.x - self.goal_x, self.y - self.goal_y) < 0.40:
            stop_traj = self.rollout((0.0, 0.0, 0.0))
            self.publish((0.0, 0.0, 0.0), uh, 0.0, self.clearance(self.x, self.y), [], stop_traj)
            return

        raw_traj = self.rollout(uh)
        raw_risk, raw_clear, raw_collision = self.evaluate(raw_traj)
        self.raw_risk_pub.publish(Float32(data=float(raw_risk)))

        infos = []
        best = None
        bestc = 1e9
        br = 0.0
        bc = 999.0
        bt = []
        safe_distance = float(self.p("safe_distance"))

        for i, u in enumerate(self.candidates(uh)):
            tr = self.rollout(u)
            risk, clear, collision = self.evaluate(tr)
            rejected = collision or clear < safe_distance
            if not rejected:
                c = self.cost(u, uh, risk, clear, tr)
                if c < bestc:
                    best, bestc, br, bc, bt = u, c, risk, clear, tr
            infos.append((i, tr, rejected))

        if best is None:
            best = self.guide_fallback()
            bt = self.rollout(best)
            br, bc, _ = self.evaluate(bt)

        self.publish(best, uh, br, bc, infos, bt, raw_traj=raw_traj)

    def publish(self, best, uh, risk, clearance, infos, selected, raw_traj=None):
        msg = Twist()
        msg.linear.x = float(best[0])
        msg.linear.y = float(best[1])
        msg.angular.z = float(best[2])
        self.cmd_pub.publish(msg)

        nh = math.sqrt(uh[0]**2 + uh[1]**2 + uh[2]**2)
        nd = math.sqrt((best[0]-uh[0])**2 + (best[1]-uh[1])**2 + (best[2]-uh[2])**2)
        self.min_pub.publish(Float32(data=float(clearance)))
        self.int_pub.publish(Float32(data=float(nd / (nh + 1e-4))))
        self.risk_pub.publish(Float32(data=float(risk)))
        self.markers(infos, selected, uh, best, raw_traj)

    def line(self, arr, ns, idx, traj, rgba, width, lifetime_ns=450000000):
        m = Marker()
        m.header.frame_id = "odom"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = int(idx)
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.lifetime = Duration(sec=0, nanosec=int(lifetime_ns))
        m.scale.x = float(width)
        m.color.r, m.color.g, m.color.b, m.color.a = [float(v) for v in rgba]
        for x, y, _ in traj:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.08
            m.points.append(p)
        arr.markers.append(m)

    def dashed_line(self, arr, ns, idx, traj, rgba, width):
        m = Marker()
        m.header.frame_id = "odom"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = int(idx)
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.lifetime = Duration(sec=0, nanosec=450000000)
        m.scale.x = float(width)
        m.color.r, m.color.g, m.color.b, m.color.a = [float(v) for v in rgba]
        for k in range(0, max(len(traj)-1, 0), 3):
            for x, y, _ in [traj[k], traj[k+1]]:
                p = Point()
                p.x = float(x)
                p.y = float(y)
                p.z = 0.10
                m.points.append(p)
        arr.markers.append(m)

    def marker_text(self, arr, idx, x, y, text, color=(0.0, 0.0, 0.0, 1.0)):
        m = Marker()
        m.header.frame_id = "odom"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "filter_text"
        m.id = int(idx)
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.lifetime = Duration(sec=0, nanosec=900000000)
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = 0.35
        m.scale.z = 0.22
        m.color.r, m.color.g, m.color.b, m.color.a = [float(v) for v in color]
        m.text = text
        arr.markers.append(m)

    def markers(self, infos, selected, uh, best, raw_traj):
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        stride = max(1, int(self.p("display_stride")))
        shown = 0
        for idx, tr, rej in infos:
            if idx % stride:
                continue
            if shown > 24:
                break
            if rej:
                self.line(arr, "rejected_rollouts", idx, tr, (1.0, 0.0, 0.0, 0.48), 0.015)
            else:
                self.line(arr, "candidate_rollouts", idx, tr, (0.45, 0.45, 0.45, 0.18), 0.010)
            shown += 1

        if raw_traj is not None:
            self.dashed_line(arr, "raw_human_prediction", 9000, raw_traj, (1.0, 0.78, 0.0, 0.95), 0.045)

        self.line(arr, "selected_rollout", 10000, selected, (0.0, 0.25, 1.0, 0.95), 0.055)
        self.line(arr, "human_input_arrow", 11000, self.rollout((uh[0]*1.1, uh[1]*1.1, uh[2])), (1.0, 0.75, 0.0, 0.90), 0.035)
        self.line(arr, "safe_output_arrow", 12000, self.rollout((best[0]*1.1, best[1]*1.1, best[2])), (0.0, 0.85, 0.1, 0.95), 0.055)

        gx, gy, mode = self.active_goal()
        self.marker_text(arr, 13000, gx, gy, f"active route target: {mode}", (0.0, 0.25, 1.0, 1.0))
        self.marker_pub.publish(arr)

def main(args=None):
    rclpy.init(args=args)
    n = CorridorSemanticFilter()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
