#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


class DoorwayMarkers(Node):
    def __init__(self):
        super().__init__("doorway_markers")
        self.declare_parameter("topic", "/debug/doorway_markers")
        self.declare_parameter("doorway_width", 1.10)
        self.declare_parameter("wall_ymin", 1.10)
        self.declare_parameter("wall_ymax", 1.45)
        self.declare_parameter("world_half_width", 3.20)
        self.declare_parameter("corridor_half_width", 0.58)
        self.pub = self.create_publisher(MarkerArray, self.get_parameter("topic").value, 10)
        self.timer = self.create_timer(0.4, self.on_timer)

    def cube(self, idx, x, y, sx, sy, color, ns="doorway"):
        m = Marker()
        m.header.frame_id = "odom"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = int(idx)
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = 0.08
        m.pose.orientation.w = 1.0
        m.scale.x = float(sx)
        m.scale.y = float(sy)
        m.scale.z = 0.16
        m.color.r, m.color.g, m.color.b, m.color.a = [float(c) for c in color]
        return m

    def line(self, idx, points, color, ns="corridor"):
        m = Marker()
        m.header.frame_id = "odom"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = int(idx)
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.025
        m.color.r, m.color.g, m.color.b, m.color.a = [float(c) for c in color]
        for x, y in points:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.11
            m.points.append(p)
        return m

    def on_timer(self):
        w = float(self.get_parameter("doorway_width").value)
        left = -w * 0.5
        right = w * 0.5
        y0 = float(self.get_parameter("wall_ymin").value)
        y1 = float(self.get_parameter("wall_ymax").value)
        hw = float(self.get_parameter("world_half_width").value)
        corridor = float(self.get_parameter("corridor_half_width").value)

        arr = MarkerArray()
        left_w = left - (-hw)
        right_w = hw - right
        arr.markers.append(self.cube(1, -hw + left_w/2, (y0+y1)/2, left_w, y1-y0, (0.05,0.05,0.05,0.95)))
        arr.markers.append(self.cube(2, right + right_w/2, (y0+y1)/2, right_w, y1-y0, (0.05,0.05,0.05,0.95)))
        arr.markers.append(self.cube(3, left-0.16, (y0+y1)/2, 0.42, 0.90, (1.0,0.0,0.0,0.18), ns="risk_zone"))
        arr.markers.append(self.cube(4, right+0.16, (y0+y1)/2, 0.42, 0.90, (1.0,0.0,0.0,0.18), ns="risk_zone"))
        arr.markers.append(self.cube(5, 0.0, 2.85, 0.20, 0.20, (0.0,1.0,0.0,0.85), ns="goal"))
        # Corridor/topology guidance lines
        arr.markers.append(self.line(6, [(0.0, -1.8), (0.0, 2.9)], (0.1, 0.9, 1.0, 0.75), ns="centerline"))
        arr.markers.append(self.line(7, [(-corridor, -1.8), (-corridor, 1.9)], (0.1, 0.9, 1.0, 0.35), ns="corridor_left"))
        arr.markers.append(self.line(8, [(corridor, -1.8), (corridor, 1.9)], (0.1, 0.9, 1.0, 0.35), ns="corridor_right"))
        self.pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = DoorwayMarkers()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
