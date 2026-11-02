\
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import String

class CorridorMarkers(Node):
    def __init__(self):
        super().__init__("corridor_markers")
        self.declare_parameter("topic", "/debug/corridor_markers")
        self.declare_parameter("target_room", "906")
        self.declare_parameter("target_door_material", "glass")
        self.declare_parameter("show_inflation", True)
        self.declare_parameter("inflation_margin", 0.56)
        self.declare_parameter("goal_y", 4.15)
        self.pub = self.create_publisher(MarkerArray, self.get_parameter("topic").value, 10)
        self.target_room = str(self.get_parameter("target_room").value)
        self.create_subscription(String, "/task_goal", self.on_task_goal, 10)
        self.timer = self.create_timer(0.5, self.on_timer)

    def on_task_goal(self, msg):
        if msg.data.strip() in {"902", "904", "906", "908"}:
            self.target_room = msg.data.strip()

    def cube(self, idx, x, y, sx, sy, color, ns="cube", z=0.08, sz=0.16):
        m = Marker()
        m.header.frame_id = "odom"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = int(idx)
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = float(z)
        m.pose.orientation.w = 1.0
        m.scale.x = float(sx)
        m.scale.y = float(sy)
        m.scale.z = float(sz)
        m.color.r, m.color.g, m.color.b, m.color.a = [float(c) for c in color]
        return m

    def line(self, idx, pts, color, width=0.03, ns="line"):
        m = Marker()
        m.header.frame_id = "odom"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = int(idx)
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = float(width)
        m.color.r, m.color.g, m.color.b, m.color.a = [float(c) for c in color]
        for x, y in pts:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.12
            m.points.append(p)
        return m

    def text(self, idx, x, y, text, scale=0.30, color=(0.1,0.1,0.1,1.0), ns="text"):
        m = Marker()
        m.header.frame_id = "odom"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = int(idx)
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = 0.32
        m.scale.z = float(scale)
        m.color.r, m.color.g, m.color.b, m.color.a = [float(c) for c in color]
        m.text = text
        return m

    def star(self, idx, x, y, color, ns="star"):
        m = Marker()
        m.header.frame_id = "odom"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = int(idx)
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = 0.16
        m.scale.x = m.scale.y = m.scale.z = 0.20
        m.color.r, m.color.g, m.color.b, m.color.a = [float(c) for c in color]
        return m

    def on_timer(self):
        goal_y = float(self.get_parameter("goal_y").value)
        rooms = [
            ("902", 3.0, "wood", (0.70,0.45,0.20,0.95)),
            ("904", 8.0, "wood", (0.70,0.45,0.20,0.95)),
            ("906", 13.35, "glass", (0.20,0.75,1.00,0.95)),
            ("908", 18.55, "glass", (0.20,0.75,1.00,0.95)),
        ]
        obstacles = [
            ("S1", "small box", 6.2, 1.2, 0.7, 0.5),
            ("S2", "map-change box", 11.8, 2.65, 1.05, 0.75),
            ("S3", "pedestrian proxy", 15.5, 3.7, 0.55, 0.55),
        ]

        arr = MarkerArray()
        arr.markers.append(self.line(1, [(0.0,0.0), (22.0,0.0), (22.0,5.0), (0.0,5.0), (0.0,0.0)], (0.25,0.25,0.25,0.9), 0.02, ns="corridor_outline"))
        arr.markers.append(self.line(2, [(0.0,2.5), (22.0,2.5)], (0.45,0.45,0.45,0.35), 0.012, ns="centerline"))
        arr.markers.append(self.cube(3, 11.0, 5.03, 22.0, 0.10, (0.55,0.55,0.55,0.85), ns="top_facade", z=0.06, sz=0.12))
        arr.markers.append(self.cube(4, 11.0, -0.03, 22.0, 0.10, (0.55,0.55,0.55,0.85), ns="bottom_wall", z=0.06, sz=0.12))

        margin = float(self.get_parameter("inflation_margin").value)
        show_infl = bool(self.get_parameter("show_inflation").value)
        for i, (sid, label, x, y, sx, sy) in enumerate(obstacles):
            if show_infl:
                arr.markers.append(self.cube(100+i, x, y, sx+2*margin, sy+2*margin, (1.0,0.0,0.0,0.09), ns="risk_inflation", z=0.035, sz=0.035))
            arr.markers.append(self.cube(120+i, x, y, sx, sy, (0.20,0.20,0.20,0.95), ns="obstacle"))
            arr.markers.append(self.text(140+i, x, y-0.48, f"{sid} {label}", 0.22, (0.0,0.0,0.0,0.95), ns="obstacle_text"))

        target_room = self.target_room
        for i, (room, x, material, color) in enumerate(rooms):
            arr.markers.append(self.text(200+i, x, 6.30, f"Room {room}", 0.38, (0.0,0.0,0.0,1.0), ns="room_text"))
            arr.markers.append(self.cube(220+i, x, 5.0, 0.95, 0.10, color, ns="door", z=0.08, sz=0.16))
            arr.markers.append(self.text(240+i, x, 5.32, material, 0.20, color, ns="door_material"))
            is_target = room == target_room
            star_color = (0.0,0.85,0.1,1.0) if is_target else (0.15,0.60,0.25,0.35)
            arr.markers.append(self.star(260+i, x, goal_y, star_color, ns="door_front"))
            if is_target:
                arr.markers.append(self.text(280+i, x+0.35, goal_y-0.32, f"TARGET Room {room}", 0.24, (0.0,0.55,0.1,1.0), ns="door_front_text"))

        # Route prior: visually different from costmap, explains task-aware detour.
        task_routes = {
            "902": [(1.2,2.3), (2.2,2.5), (3.0,3.3), (3.0,goal_y)],
            "904": [(1.2,2.3), (4.8,2.35), (6.8,2.6), (8.0,3.35), (8.0,goal_y)],
            "906": [(1.2,2.3), (4.8,2.35), (8.6,2.85), (10.8,3.80), (12.2,3.95), (13.35,goal_y)],
            "908": [(1.2,2.3), (4.8,2.35), (8.6,2.85), (10.8,3.80),
                    (12.8,3.95), (14.0,2.70), (16.5,2.70), (18.0,3.50), (18.55,goal_y)],
        }
        route = task_routes[target_room]
        arr.markers.append(self.line(360, route, (0.20,0.60,0.35,0.78), 0.045, ns="semantic_route_prior"))
        for j, (x, y) in enumerate(route[2:-1], start=1):
            arr.markers.append(self.star(370+j, x, y, (0.0,0.25,1.0,0.80), ns="route_points"))
        arr.markers.append(self.text(390, route[-1][0], 4.55, f"active task route -> Room {target_room}", 0.22, (0.0,0.25,1.0,1.0), ns="route_text"))
        arr.markers.append(self.text(400, 12.4, -0.45, "Ranger-feasible candidate commands + runtime task retargeting", 0.22, (0.55,0.35,0.0,1.0), ns="note"))
        self.pub.publish(arr)

def main(args=None):
    rclpy.init(args=args)
    node = CorridorMarkers()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
