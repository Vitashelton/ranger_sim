"""Rotate the two Gazebo door leaves about their fixed model-origin hinges."""
import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from generate_ranger_lab_world import DOOR_HINGES

def main():
    parser = argparse.ArgumentParser(description='Set Gazebo fire-door leaf angles (0=closed, 90=open).')
    parser.add_argument('north_angle', type=float)
    parser.add_argument('south_angle', type=float)
    args = parser.parse_args()
    if not all(0 <= angle <= 90 for angle in (args.north_angle,args.south_angle)):
        parser.error('Both leaf angles must be within 0..90 degrees.')
    import rclpy
    from ros_gz_interfaces.srv import SetEntityPose
    rclpy.init(args=[])
    node = rclpy.create_node('gazebo_fire_door_angle_control')
    client = node.create_client(SetEntityPose,'/world/ranger_lab_corridor/set_pose')
    try:
        if not client.wait_for_service(timeout_sec=3.0):
            raise RuntimeError('Gazebo set_pose service unavailable; source the simulation workspace and use its ROS domain.')
        for side,angle in [('north',args.north_angle),('south',args.south_angle)]:
            request=SetEntityPose.Request()
            request.entity.name=f'fire_door_{side}_leaf'
            request.entity.type=2
            request.pose.position.x,request.pose.position.y=DOOR_HINGES[side]
            yaw=math.radians(angle)*(1 if side=='north' else -1)
            request.pose.orientation.z=math.sin(yaw/2)
            request.pose.orientation.w=math.cos(yaw/2)
            future=client.call_async(request)
            rclpy.spin_until_future_complete(node,future,timeout_sec=3.0)
            if not future.done() or future.exception() or not future.result().success:
                raise RuntimeError(f'Gazebo did not acknowledge {side} leaf angle {angle}.')
            print(f'{side} leaf: {angle:g} degrees, fixed hinge {DOOR_HINGES[side]}')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__=='__main__':
    main()
