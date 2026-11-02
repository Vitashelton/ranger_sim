"""Process singleton ROS bridge; Streamlit reruns reuse the same node/executor."""
import atexit
import json
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger


class _BridgeNode(Node):
    def __init__(self):
        super().__init__("office_rpg_streamlit_bridge")
        self.lock = threading.RLock()
        self.cache = {"mission_status": "", "npc_states": "",
                      "semantic_regions": "", "reminder_event": "",
                      "patrol_report": "", "developer_truth": "",
                      "topology_graph": "", "llm_planner_debug": "",
                      "plan_gate_result": ""}
        self.events = []
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/office_rpg/mission_status",
                                 lambda m: self._store("mission_status", m.data), latched)
        self.create_subscription(String, "/office_rpg/npc_states",
                                 lambda m: self._store("npc_states", m.data), 10)
        self.create_subscription(String, "/office_rpg/semantic_regions",
                                 lambda m: self._store("semantic_regions", m.data), latched)
        self.create_subscription(String, "/office_rpg/reminder_event",
                                 lambda m: self._store("reminder_event", m.data), 10)
        self.create_subscription(String, "/office_rpg/patrol_report",
                                 lambda m: self._store("patrol_report", m.data), latched)
        self.create_subscription(String, "/office_rpg/topology_graph",
                                 lambda m: self._store("topology_graph", m.data), latched)
        self.create_subscription(String, "/office_rpg/llm_planner_debug",
                                 lambda m: self._store("llm_planner_debug", m.data), latched)
        self.create_subscription(String, "/office_rpg/plan_gate_result",
                                 lambda m: self._store("plan_gate_result", m.data), latched)
        self.create_subscription(String, "/office_rpg/event_log", self._event, 50)
        self.mission_pub = self.create_publisher(String, "/office_rpg/mission_text", 10)
        self.provider_pub = self.create_publisher(String, "/office_rpg/llm_provider", 10)
        self.stop_client = self.create_client(Trigger, "/office_rpg/stop")
        self.reset_client = self.create_client(Trigger, "/office_rpg/reset")
        self.truth_subscription = None

    def set_developer_truth(self, enabled):
        """Subscribe to simulation truth only while developer display is enabled."""
        with self.lock:
            if enabled and self.truth_subscription is None:
                self.truth_subscription = self.create_subscription(
                    String, "/office_rpg/sim/npc_ground_truth",
                    lambda m: self._store("developer_truth", m.data), 10)
            elif not enabled and self.truth_subscription is not None:
                self.destroy_subscription(self.truth_subscription)
                self.truth_subscription = None
                self.cache["developer_truth"] = ""

    def _store(self, key, value):
        with self.lock:
            self.cache[key] = value

    def _event(self, msg):
        with self.lock:
            self.events.append(msg.data)
            self.events = self.events[-100:]

    def snapshot(self):
        with self.lock:
            return {**self.cache, "events": list(self.events)}


class RosBridge:
    _instance = None
    _guard = threading.Lock()

    def __new__(cls):
        with cls._guard:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._start()
        return cls._instance

    def _start(self):
        if not rclpy.ok():
            rclpy.init()
        self.node = _BridgeNode()
        self.executor = MultiThreadedExecutor(num_threads=2)
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True,
                                       name="office-rpg-ros-executor")
        self.thread.start()
        self.closed = False

    def snapshot(self):
        return self.node.snapshot()

    def send_mission(self, text, provider):
        self.node.provider_pub.publish(String(data=provider))
        self.node.mission_pub.publish(String(data=text))

    def set_developer_truth(self, enabled):
        self.node.set_developer_truth(bool(enabled))

    def _call(self, client):
        if not client.wait_for_service(timeout_sec=1.0):
            return False, "ROS 服务尚未就绪"
        future = client.call_async(Trigger.Request())
        deadline = time.monotonic() + 2.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not future.done():
            return False, "ROS 服务调用超时"
        result = future.result()
        return bool(result.success), result.message

    def stop(self):
        return self._call(self.node.stop_client)

    def reset(self):
        return self._call(self.node.reset_client)

    def shutdown(self):
        if getattr(self, "closed", True):
            return
        self.closed = True
        self.executor.shutdown(timeout_sec=1.0)
        if self.thread.is_alive() and threading.current_thread() is not self.thread:
            self.thread.join(timeout=1.0)
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


atexit.register(lambda: RosBridge._instance and RosBridge._instance.shutdown())
