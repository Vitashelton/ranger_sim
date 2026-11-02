"""YAML-driven, wall-safe NPC pose scheduler for the existing corridor world."""
import json
import math
import random
import time
from pathlib import Path

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.parameter import Parameter
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String


NPC_IDS = ("teacher_zhang", "student_li", "visitor")


class NpcScheduleNode(Node):
    def __init__(self):
        super().__init__("office_rpg_npc_schedule")
        default_cfg = str(Path(get_package_share_directory(
            "rangermini_dynamic_semantic")) / "config" / "npc_schedules.yaml")
        self.declare_parameter("schedule_file", default_cfg)
        self.declare_parameter("cycle_sec", 180.0)
        self.declare_parameter("initial_time", -1.0)
        self.declare_parameter("random_seed", -1)
        self.declare_parameter("set_pose_service",
                               "/world/corridor_902_904_906_908/set_pose")
        with open(str(self.get_parameter("schedule_file").value), encoding="utf-8") as stream:
            self.cfg = yaml.safe_load(stream)
        configured_initial = float(self.cfg.get("initial_time", 0.0))
        parameter_initial = float(self.get_parameter("initial_time").value)
        self.initial_time = configured_initial if parameter_initial < 0.0 else parameter_initial
        configured_seed = int(self.cfg.get("random_seed", 17))
        parameter_seed = int(self.get_parameter("random_seed").value)
        self.random_seed = configured_seed if parameter_seed < 0 else parameter_seed
        configured_cycle = float(self.cfg.get("cycle_sec", 180.0))
        if float(self.get_parameter("cycle_sec").value) == 180.0:
            self.set_parameters([Parameter("cycle_sec", value=configured_cycle)])
        self.started = time.monotonic()
        self.sim_time = None
        self.sim_origin = 0.0
        self.client = self.create_client(
            SetEntityPose, str(self.get_parameter("set_pose_service").value))
        self.state_pub = self.create_publisher(
            String, "/office_rpg/sim/npc_schedule_state", 10)
        self.create_subscription(
            String, "/office_rpg/schedule_reset", self.on_reset, 10)
        self.create_subscription(Clock, "/clock", self.on_clock, 10)
        self.timer = self.create_timer(0.25, self.tick)
        self.warned_service = False

    def on_reset(self, _msg):
        self.started = time.monotonic()
        self.sim_origin = self.sim_time or 0.0
        self.get_logger().info(
            f"NPC schedule reset to configured initial_time={self.initial_time:.2f}")

    def on_clock(self, msg):
        self.sim_time = float(msg.clock.sec) + 1.0e-9 * float(msg.clock.nanosec)

    def elapsed_time(self):
        cycle = float(self.get_parameter("cycle_sec").value)
        if self.sim_time is not None:
            elapsed = self.initial_time + max(0.0, self.sim_time - self.sim_origin)
        else:
            elapsed = self.initial_time + (time.monotonic() - self.started)
        return elapsed % cycle

    def stable_offset(self, npc_id, location_name, radius):
        rng = random.Random(f"{self.random_seed}:{npc_id}:{location_name}")
        angle = rng.uniform(-math.pi, math.pi)
        distance = radius * math.sqrt(rng.random())
        return distance * math.cos(angle), distance * math.sin(angle)

    @staticmethod
    def interpolate_polyline(points, fraction):
        if len(points) == 1:
            return float(points[0]["x"]), float(points[0]["y"]), 0.0
        lengths = [math.hypot(float(b["x"])-float(a["x"]),
                              float(b["y"])-float(a["y"]))
                   for a, b in zip(points[:-1], points[1:])]
        total = sum(lengths) or 1.0
        distance = max(0.0, min(1.0, fraction)) * total
        for index, length in enumerate(lengths):
            if distance <= length or index == len(lengths) - 1:
                a, b = points[index], points[index + 1]
                ratio = 0.0 if length == 0 else min(1.0, distance / length)
                dx, dy = float(b["x"])-float(a["x"]), float(b["y"])-float(a["y"])
                return (float(a["x"])+ratio*dx, float(a["y"])+ratio*dy,
                        math.atan2(dy, dx))
            distance -= length
        return float(points[-1]["x"]), float(points[-1]["y"]), 0.0

    def state_at(self, npc_id, elapsed):
        npc = self.cfg[npc_id]
        entries = npc["schedule"]
        entry = next((item for item in entries
                      if float(item["start_sec"]) <= elapsed < float(item["end_sec"])),
                     entries[-1])
        if "transition" in entry:
            name = entry["transition"]
            duration = max(0.001, float(entry["end_sec"])-float(entry["start_sec"]))
            x, y, yaw = self.interpolate_polyline(
                self.cfg["transitions"][name],
                (elapsed-float(entry["start_sec"]))/duration)
            region = name
            schedule_state = "TRANSITION"
            dx, dy = self.stable_offset(npc_id, name, 0.04)
        else:
            region = entry["region"]
            anchor = self.cfg["regions"][region]
            x, y, yaw = float(anchor["x"]), float(anchor["y"]), float(anchor.get("yaw", 0.0))
            schedule_state = "AT_REGION"
            dx, dy = self.stable_offset(
                npc_id, region, float(anchor.get("random_radius", 0.0)))
        x += dx
        y += dy
        return {
            "identity_id": npc_id, "display_name": npc["display_name"],
            "current_region": region, "activity_state": entry.get("activity_state", "IDLE"),
            "schedule_state": schedule_state, "visible": True,
            "simulated_detection_confidence": float(npc["simulated_detection_confidence"]),
            "x": x, "y": y, "yaw": yaw,
        }

    def request_pose(self, state):
        if not self.client.service_is_ready():
            if not self.warned_service:
                self.get_logger().warn("Waiting for Gazebo set_pose bridge; publishing schedule metadata only")
                self.warned_service = True
            return
        request = SetEntityPose.Request()
        request.entity = Entity(name=state["identity_id"], type=Entity.MODEL)
        request.pose = Pose()
        request.pose.position.x = state["x"]
        request.pose.position.y = state["y"]
        request.pose.position.z = 0.02
        request.pose.orientation.z = math.sin(state["yaw"] / 2.0)
        request.pose.orientation.w = math.cos(state["yaw"] / 2.0)
        self.client.call_async(request)

    def tick(self):
        elapsed = self.elapsed_time()
        states = [self.state_at(npc_id, elapsed) for npc_id in NPC_IDS]
        for state in states:
            self.request_pose(state)
        payload = {"simulation_stub": True, "truth_scope": "adapter_only",
                   "schedule_time_sec": round(elapsed, 2),
                   "initial_time": self.initial_time, "random_seed": self.random_seed,
                   "npcs": states}
        self.state_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))


def main(args=None):
    rclpy.init(args=args)
    node = NpcScheduleNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
