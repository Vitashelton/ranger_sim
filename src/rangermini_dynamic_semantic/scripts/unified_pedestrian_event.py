#!/usr/bin/env python3
"""Evaluator-side repeatable moving obstacle; no algorithm outputs."""
import argparse
import math
import json
import rclpy
from rclpy.node import Node
from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity


class Event(Node):
    def __init__(self,x):
        super().__init__('unified_pedestrian_event',parameter_overrides=[
            rclpy.parameter.Parameter('use_sim_time',value=True)])
        self.x = x
        self.client = self.create_client(SetEntityPose,'/world/ranger_lab_corridor/set_pose')
        self.create_timer(.2,self.tick)

    def tick(self):
        if not self.client.service_is_ready():
            return
        t = self.get_clock().now().nanoseconds*1e-9
        request = SetEntityPose.Request()
        request.entity.name = 'pedestrian_proxy'
        request.entity.type = Entity.MODEL
        request.pose.position.x = self.x+.7*math.sin(.65*t)
        request.pose.position.y = -3.45
        request.pose.position.z = .7
        request.pose.orientation.w = 1.
        future = self.client.call_async(request)
        commanded = {'sim_time':t,'x':request.pose.position.x,'y':request.pose.position.y}
        def completed(response):
            try:
                result = response.result()
                commanded['acknowledged'] = bool(result.success)
                commanded['ack_sim_time'] = self.get_clock().now().nanoseconds*1e-9
            except Exception as exc:
                commanded['acknowledged'] = False
                commanded['error'] = str(exc)
            print(json.dumps(commanded),flush=True)
        future.add_done_callback(completed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--x',type=float,required=True)
    args = p.parse_args()
    rclpy.init()
    node = Event(args.x)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':
    main()
