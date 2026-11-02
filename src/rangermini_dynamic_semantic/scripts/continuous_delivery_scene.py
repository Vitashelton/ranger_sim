#!/usr/bin/env python3
"""Sequential encounters in the established laboratory; independent fixture.

Actor triggers depend on measured robot proximity, never on a policy label.
Only acknowledged actor poses are passed to the offline evaluator.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from grounded_delivery_scene import create_world as create_base_world


SCENARIO = 'continuous_delivery'
CART_START = (8.6, -2.1)
PERSON_START = (12.8, .45)
PARKED_CART = (12.0, -1.95)
PORTAL_APPROACH = (12.8, -1.15)
CART_TRIGGER_DISTANCE = 4.0
PERSON_TRIGGER_DISTANCE = 3.0


def create_world(output, scenario=SCENARIO):
    if scenario != SCENARIO:
        raise ValueError('Expected '+SCENARIO)
    metadata = create_base_world(output, 'headon')
    tree = ET.parse(output)
    world = tree.getroot().find('world')
    world.find("model[@name='lab_student']/pose").text = '12.8 .45 .02 0 0 -1.57079632679'
    parked = copy.deepcopy(world.find("model[@name='service_trolley']"))
    parked.set('name', 'entry_equipment_station')
    parked.find('pose').text = f'{PARKED_CART[0]} {PARKED_CART[1]} .12 0 0 0'
    world.append(parked)
    tree.write(output, encoding='unicode', xml_declaration=True)
    metadata.update(scenario=SCENARIO, protocol_version='continuous_delivery_v1',
        person_initial=list(PERSON_START), trolley_initial=list(CART_START),
        parked_equipment_pose=list(PARKED_CART),
        actor_triggers={
            'trolley': {'distance_to_initial_actor_m': CART_TRIGGER_DISTANCE, 'speed_m_s': .22},
            'person': {'distance_to_portal_approach_m': PERSON_TRIGGER_DISTANCE, 'speed_m_s': .28}},
        trigger_source='independent measured robot pose; same geometry for every policy',
        purpose='successive oncoming encounter, parked equipment bypass and occupied destination access')
    return metadata


def actor_positions(now, triggers):
    cart_dt = 0.0 if triggers['trolley'] is None else max(0.0, now-triggers['trolley'])
    person_dt = 0.0 if triggers['person'] is None else max(0.0, now-triggers['person'])
    return (CART_START[0]-.22*min(27., cart_dt), CART_START[1]), (
        PERSON_START[0], PERSON_START[1]-.28*min(13.75, person_dt))


def fixture(args):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
    from geometry_msgs.msg import Pose, PoseStamped
    from ros_gz_interfaces.msg import Entity
    from ros_gz_interfaces.srv import SetEntityPose
    from std_msgs.msg import String

    class Fixture(Node):
        def __init__(self):
            super().__init__('continuous_delivery_fixture', parameter_overrides=[
                rclpy.parameter.Parameter('use_sim_time', value=True)])
            self.client = self.create_client(SetEntityPose, '/world/ranger_lab_corridor/set_pose')
            self.origin = None
            self.robot_pose = None
            self.triggers = {'trolley': None, 'person': None}
            self.inflight = {}
            self.log = Path(args.output).open('w')
            self.events = Path(args.output).with_name('fixture_events.jsonl').open('w')
            mission_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                     reliability=ReliabilityPolicy.RELIABLE)
            pose_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.create_subscription(String, '/harness/mission_start', self.start, mission_qos)
            self.create_subscription(PoseStamped, '/model/rangermini_2_0/pose', self.pose, pose_qos)
            self.create_timer(.12, self.tick)

        def start(self, msg):
            if self.origin is None:
                self.origin = float(json.loads(msg.data)['sim_time'])

        def pose(self, msg):
            self.robot_pose = (msg.pose.position.x, msg.pose.position.y)

        def move(self, name, xy, z, yaw, now):
            if name in self.inflight and not self.inflight[name].done():
                return
            request = SetEntityPose.Request()
            request.entity = Entity(name=name, type=Entity.MODEL)
            request.pose = Pose()
            request.pose.position.x, request.pose.position.y = map(float, xy)
            request.pose.position.z = z
            request.pose.orientation.z = math.sin(yaw/2)
            request.pose.orientation.w = math.cos(yaw/2)
            pending = self.client.call_async(request)
            self.inflight[name] = pending
            row = dict(sim_time=now, elapsed=now-self.origin, entity=name,
                       pose=[*xy, yaw], source='independent_scene_fixture')
            def complete(result):
                try:
                    row['acknowledged'] = bool(result.result().success)
                except Exception:
                    row['acknowledged'] = False
                self.log.write(json.dumps(row)+'\n')
                self.log.flush()
            pending.add_done_callback(complete)

        def tick(self):
            if self.origin is None or self.robot_pose is None or not self.client.service_is_ready():
                return
            now = self.get_clock().now().nanoseconds*1e-9
            for name, target, threshold in (
                    ('trolley', CART_START, CART_TRIGGER_DISTANCE),
                    ('person', PORTAL_APPROACH, PERSON_TRIGGER_DISTANCE)):
                if self.triggers[name] is None and math.dist(self.robot_pose, target) <= threshold:
                    self.triggers[name] = now
                    self.events.write(json.dumps(dict(event='ACTOR_TRIGGERED', actor=name,
                        sim_time=now, elapsed=now-self.origin, robot_pose=self.robot_pose,
                        reference=target, threshold_m=threshold))+'\n')
                    self.events.flush()
            cart, person = actor_positions(now, self.triggers)
            self.move('service_trolley', cart, .12, 0., now)
            self.move('lab_student', person, .02, -math.pi/2, now)

    rclpy.init()
    node = Fixture()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.log.close()
        node.events.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', choices=[SCENARIO], default=SCENARIO)
    parser.add_argument('--output')
    parser.add_argument('--generate-world')
    options = parser.parse_args()
    if options.generate_world:
        print(json.dumps(create_world(options.generate_world), ensure_ascii=False, indent=2))
    elif options.output:
        fixture(options)
    else:
        parser.error('--output or --generate-world is required')
