#!/usr/bin/env python3
"""Reuse the furnished world and schedule moving physical obstacle models.

This fixture writes independent requested/acknowledged actor-pose records. It
is not subscribed to by the online navigation controller.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from generate_office_delivery_world import make_world


SCENARIOS = ("plain", "headon", "crossing", "door_crossing")


def create_world(output, scenario):
    make_world("east_weave", output)
    tree = ET.parse(output)
    world = tree.getroot().find("world")
    person = world.find("model[@name='lab_student']")
    person_start = {"plain": [2.85, -6.05], "headon": [2.85, -6.05],
                    "crossing": [8.10, .45], "door_crossing": [12.80, .45]}[scenario]
    person.find("pose").text = f"{person_start[0]} {person_start[1]} .02 0 0 {-math.pi/2}"
    parts = [m for m in world.findall("model") if m.get("name", "").startswith(("service_trolley", "trolley_"))]
    source = world.find("model[@name='service_trolley']")
    origin = list(map(float, source.findtext("pose").split()))
    trolley = ET.Element("model", name="service_trolley")
    ET.SubElement(trolley, "static").text = "true"
    cart_start = [8.6, -2.1] if scenario == "headon" else [8.8, -.95]
    ET.SubElement(trolley, "pose").text = f"{cart_start[0]} {cart_start[1]} .12 0 0 0"
    for index, part in enumerate(parts):
        mp = list(map(float, part.findtext("pose").split()))
        for item in part.findall("link"):
            link = copy.deepcopy(item)
            link.set("name", f"cart_part_{index}_{link.get('name')}")
            lp = list(map(float, link.findtext("pose", "0 0 0 0 0 0").split()))
            pose = link.find("pose")
            if pose is None:
                pose = ET.SubElement(link, "pose")
            pose.text = " ".join(map(str, [mp[0]-origin[0]+lp[0], mp[1]-origin[1]+lp[1],
                mp[2]-.12+lp[2], lp[3], lp[4], mp[5]+lp[5]]))
            trolley.append(link)
        world.remove(part)
    # The collision envelope is independent of the rendered trolley asset.
    # GPU lidar uses visuals: an opaque lower equipment cabinet is needed so
    # it observes the cart body rather than intermittently hitting thin posts.
    base = trolley.find("link")
    collision = ET.SubElement(base, "collision", name="moving_cart_envelope")
    ET.SubElement(collision, "pose").text = "0 0 .39 0 0 0"
    ET.SubElement(ET.SubElement(ET.SubElement(collision, "geometry"), "box"), "size").text = ".70 .42 .96"
    cabinet = ET.SubElement(base, 'visual', name='lower_equipment_cabinet')
    ET.SubElement(cabinet, 'pose').text = '0 0 .35 0 0 0'
    ET.SubElement(ET.SubElement(ET.SubElement(cabinet, 'geometry'), 'box'), 'size').text = '.68 .40 .64'
    material = ET.SubElement(cabinet, 'material')
    ET.SubElement(material, 'ambient').text = '.30 .35 .39 1'
    ET.SubElement(material, 'diffuse').text = '.42 .48 .53 1'
    world.append(trolley)
    sensor = world.find("model[@name='navigation_overhead_camera']/link/sensor")
    world.find("model[@name='navigation_overhead_camera']/pose").text = f"8.2 -1.0 11 0 {math.pi/2} {-math.pi/2}"
    if sensor is not None:
        sensor.find("camera/horizontal_fov").text = "1.35"
        sensor.find("camera/image/width").text = "1280"
        sensor.find("camera/image/height").text = "720"
        sensor.find("update_rate").text = "5"
    world.find("physics/real_time_factor").text = ".5"
    tree.write(output, encoding="unicode", xml_declaration=True)
    return {"scenario": scenario, "person_initial": person_start,
            "trolley_initial": cart_start, "world_name": world.get("name"),
            "schedule_source": "evaluation_fixture_only", "start_pose": [3, -2.1, 0],
            "trolley_observation_geometry": "opaque .68 x .40 m lower equipment cabinet; full collision proxy .70 x .42 m"}


def fixture(args):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
    from geometry_msgs.msg import Pose
    from ros_gz_interfaces.msg import Entity
    from ros_gz_interfaces.srv import SetEntityPose
    from std_msgs.msg import String

    class Fixture(Node):
        def __init__(self):
            super().__init__("grounded_delivery_scene_fixture", parameter_overrides=[
                rclpy.parameter.Parameter("use_sim_time", value=True)])
            self.client = self.create_client(SetEntityPose, "/world/ranger_lab_corridor/set_pose")
            self.origin = None
            self.inflight = {}
            self.log = Path(args.output).open("w")
            qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
            self.create_subscription(String, "/harness/mission_start", self.start, qos)
            self.create_timer(.12, self.tick)

        def start(self, msg):
            if self.origin is None:
                self.origin = float(json.loads(msg.data)["sim_time"])

        def move(self, name, xy, z, yaw, stamp):
            if name in self.inflight and not self.inflight[name].done():
                return
            req = SetEntityPose.Request()
            req.entity = Entity(name=name, type=Entity.MODEL)
            req.pose = Pose()
            req.pose.position.x, req.pose.position.y, req.pose.position.z = float(xy[0]), float(xy[1]), z
            req.pose.orientation.z, req.pose.orientation.w = math.sin(yaw/2), math.cos(yaw/2)
            future = self.client.call_async(req)
            self.inflight[name] = future
            row = {"sim_time": stamp, "elapsed": stamp-self.origin, "entity": name,
                   "pose": [*xy, yaw], "source": "independent_scene_fixture"}
            def complete(result):
                try:
                    row["acknowledged"] = bool(result.result().success)
                except Exception:
                    row["acknowledged"] = False
                self.log.write(json.dumps(row)+"\n")
                self.log.flush()
            future.add_done_callback(complete)

        def tick(self):
            if self.origin is None or not self.client.service_is_ready():
                return
            now = self.get_clock().now().nanoseconds*1e-9
            elapsed = now-self.origin
            if args.scenario == "headon":
                x = 8.6-.22*min(27, max(0, elapsed-2))
                self.move("service_trolley", [x, -2.1], .12, 0, now)
            elif args.scenario in ("crossing", "door_crossing"):
                start = 8 if args.scenario == "crossing" else 18
                x = 8.1 if args.scenario == "crossing" else 12.8
                y = .45-.28*min(17.5, max(0, elapsed-start))
                self.move("lab_student", [x, y], .02, -math.pi/2, now)

    rclpy.init()
    node = Fixture()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.log.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    fixture(args)
