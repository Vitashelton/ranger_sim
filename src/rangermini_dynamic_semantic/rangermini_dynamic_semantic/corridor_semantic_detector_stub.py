\
#!/usr/bin/env python3
"""
A lightweight semantic detector stub.

This is not YOLO. It publishes YOLO-like detections as JSON strings so the
corridor benchmark has the same high-level structure as a real semantic
navigation system:

RGB/depth/LiDAR -> detector -> localized semantic objects -> memory -> target pose.

To replace this with YOLO later, publish the same JSON format or bridge from
vision_msgs/Detection2DArray / Detection3DArray into the memory node.
"""
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class CorridorSemanticDetectorStub(Node):
    def __init__(self):
        super().__init__("corridor_semantic_detector_stub")
        self.declare_parameter("output_topic", "/semantic_detections_json")
        self.declare_parameter("rate_hz", 1.0)
        self.pub = self.create_publisher(String, self.get_parameter("output_topic").value, 10)

        self.objects = [
            {"label": "wood_door", "room": "902", "material": "wood", "x": 3.00, "y": 5.00, "confidence": 0.92},
            {"label": "wood_door", "room": "904", "material": "wood", "x": 8.00, "y": 5.00, "confidence": 0.91},
            {"label": "glass_door", "room": "906", "material": "glass", "x": 13.35, "y": 5.00, "confidence": 0.94},
            {"label": "glass_door", "room": "908", "material": "glass", "x": 18.55, "y": 5.00, "confidence": 0.93},
            {"label": "door_front", "room": "902", "material": "wood", "x": 3.00, "y": 4.15, "confidence": 0.90},
            {"label": "door_front", "room": "904", "material": "wood", "x": 8.00, "y": 4.15, "confidence": 0.90},
            {"label": "door_front", "room": "906", "material": "glass", "x": 13.35, "y": 4.15, "confidence": 0.95},
            {"label": "door_front", "room": "908", "material": "glass", "x": 18.55, "y": 4.15, "confidence": 0.92},
        ]

        self.timer = self.create_timer(1.0 / float(self.get_parameter("rate_hz").value), self.on_timer)
        self.get_logger().info("Publishing semantic detector stub observations on /semantic_detections_json")

    def on_timer(self):
        msg = String()
        msg.data = json.dumps({"frame_id": "odom", "detections": self.objects}, ensure_ascii=False)
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CorridorSemanticDetectorStub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
