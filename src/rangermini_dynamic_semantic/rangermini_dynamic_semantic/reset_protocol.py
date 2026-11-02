"""Shared atomic-reset transport and deterministic state hashing."""
import json

from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from .reset_contract import canonical_hash, canonical_json


RESET_REQUEST_TOPIC = "/benchmark/reset/request"
RESET_ACK_TOPIC = "/benchmark/reset/ack"
RESET_READY_TOPIC = "/benchmark/reset/ready"


def reset_qos():
    return QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL)


class ResetParticipant:
    """Bind one node to the reset barrier without changing its algorithm."""

    def __init__(self, node, participant_name, reset_callback, ready_callback):
        self.node = node
        self.participant_name = participant_name
        self.reset_callback = reset_callback
        self.ready_callback = ready_callback
        self.active_request = None
        self.resetting = False
        self.ack_pub = node.create_publisher(String, RESET_ACK_TOPIC, 20)
        node.create_subscription(
            String, RESET_REQUEST_TOPIC, self.on_request, reset_qos())
        node.create_subscription(
            String, RESET_READY_TOPIC, self.on_ready, reset_qos())

    @property
    def reset_id(self):
        return str((self.active_request or {}).get("reset_id", "startup"))

    @property
    def reset_epoch(self):
        return int((self.active_request or {}).get("reset_epoch", 0))

    def on_request(self, msg):
        try:
            request = json.loads(msg.data)
        except ValueError:
            self.node.get_logger().error("Rejected malformed reset request")
            return
        if not request.get("reset_id"):
            self.node.get_logger().error("Rejected reset request without reset_id")
            return
        self.active_request = request
        self.resetting = True
        self.reset_callback(request)

    def acknowledge(self, state):
        if self.active_request is None:
            return
        deterministic_state = dict(state)
        payload = {
            "reset_id": self.reset_id,
            "reset_epoch": self.reset_epoch,
            "node": self.participant_name,
            "status": "READY",
            "state_hash": canonical_hash(deterministic_state),
            "state": deterministic_state,
            "timestamp": self.node.get_clock().now().nanoseconds * 1.0e-9,
            "trial_context": self.active_request.get("trial_context", {}),
        }
        self.ack_pub.publish(String(data=canonical_json(payload)))

    def on_ready(self, msg):
        try:
            ready = json.loads(msg.data)
        except ValueError:
            return
        if ready.get("reset_id") != self.reset_id:
            return
        self.resetting = False
        self.ready_callback(ready)

    def context_fields(self):
        return {"reset_id": self.reset_id, "reset_epoch": self.reset_epoch}
