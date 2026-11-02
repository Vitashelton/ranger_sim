#!/usr/bin/env python3
"""Independent passive ROS observer. Never publishes motion commands."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import Twist
from rangermini_dynamic_semantic.runtime_timeline import RuntimeTimeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rclpy.init()
    node = Node('ranger_runtime_observer')
    log = RuntimeTimeline(args.output)
    subscriptions = []
    for kind, topic in ((LaserScan,'/scan'), (Odometry,'/odom'),
                        (String,'/motion_mode_debug'), (Clock,'/clock'),
                        (Twist,'/cmd_vel_safe')):
        def make_callback(topic):
            def callback(msg, info=None):
                stamp = msg.header.stamp if hasattr(msg,'header') else getattr(msg,'clock',None)
                log.emit(event='independent_receive', topic=topic,
                    acquisition_ros_s=stamp.sec+stamp.nanosec*1e-9 if stamp else None,
                    dds_received_system_ns=getattr(info,'received_timestamp',None),
                    dds_source_system_ns=getattr(info,'source_timestamp',None))
            return callback
        subscriptions.append(node.create_subscription(kind,topic,make_callback(topic),qos_profile_sensor_data))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        log.close()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
