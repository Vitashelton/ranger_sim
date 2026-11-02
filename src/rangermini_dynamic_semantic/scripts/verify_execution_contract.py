#!/usr/bin/env python3
"""Check freshness admission and the real ROS adapter with a paused ROS clock."""
import argparse, json, math, time
from pathlib import Path
from rangermini_dynamic_semantic.execution_freshness import observation_fault


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    healthy={k:(10.,20.) for k in ('pose','scan','actuation')}
    cases=[]
    for name,now,wall,obs,expected in [
        ('fresh',10.1,20.1,healthy,None),
        ('laser_stale',10.8,20.8,{**healthy,'pose':(10.8,20.8),'actuation':(10.8,20.8)},'scan:stale'),
        ('wall_stale',10.1,22.,healthy,'pose:stale'),
        ('clock_regression',9.,20.,healthy,'pose:clock_regression'),
        ('invalid_timestamp',10.,20.,{**healthy,'scan':(math.nan,20.)},'scan:invalid_stamp'),
        ('partial_recovery',10.8,20.8,{**healthy,'scan':(10.8,20.8)},'pose:stale'),
        ('recovered',10.8,20.8,{k:(10.8,20.8) for k in healthy},None),
    ]:
        actual=observation_fault(now,wall,obs)
        assert actual==expected,(name,actual,expected)
        cases.append({'case':name,'fault':actual,'pass':True})
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from geometry_msgs.msg import Twist
    from rosgraph_msgs.msg import Clock
    from rangermini_dynamic_semantic.ranger_motion_adapter import RangerMotionAdapter
    rclpy.init(args=['--ros-args','-p','use_sim_time:=true','-p','wall_command_timeout:=0.25'])
    adapter=RangerMotionAdapter();probe=Node('execution_contract_probe')
    commands=probe.create_publisher(Twist,'/cmd_vel_safe',10)
    clock=probe.create_publisher(Clock,'/clock',10);observed=[]
    probe.create_subscription(Twist,'/cmd_vel_gazebo',lambda m:observed.append((time.monotonic(),m.linear.x,m.linear.y,m.angular.z)),10)
    ex=SingleThreadedExecutor();ex.add_node(adapter);ex.add_node(probe)
    start=time.monotonic()
    try:
        while time.monotonic()-start<1.5:
            delta=time.monotonic()-start
            c=Clock();c.clock.sec=10+int(delta);c.clock.nanosec=int((delta%1)*1e9);clock.publish(c)
            cmd=Twist();cmd.linear.x=.2;commands.publish(cmd);ex.spin_once(timeout_sec=.01)
        assert any(abs(x[1])>.01 for x in observed),'No motion command before fault'
        frozen=time.monotonic();observed.clear()
        while time.monotonic()-frozen<.7:ex.spin_once(timeout_sec=.01)
        zeros=[r[0]-frozen for r in observed if max(map(abs,r[1:]))<1e-9]
        assert zeros and min(zeros)<.45,zeros
        assert not any(max(map(abs,r[1:]))>1e-9 and r[0]-frozen>.35 for r in observed)
        result={'kind':'ROS adapter fault injection; not a Gazebo navigation or physical chassis trial',
                'freshness_cases':cases,'paused_ros_clock':True,'wall_timeout_s':.25,
                'first_zero_after_pause_s':min(zeros),'no_motion_after_350ms':True}
        Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result))
    finally:
        ex.shutdown();adapter.destroy_node();probe.destroy_node();rclpy.shutdown()
if __name__=='__main__':main()
