#!/usr/bin/env python3
"""ROS-message integration check with synthetic telemetry; no chassis commands."""
import argparse,json,subprocess,sys,time
from pathlib import Path
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ranger_msgs.msg import ActuatorStateArray,ActuatorState,MotionState
from std_msgs.msg import String
from rangermini_dynamic_semantic.ranger_feedback_bridge import calibrated_angles


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
 states=[]
 for i,value in enumerate([.1,.2,.3,.4],4):
  s=ActuatorState();s.id=i;s.motor.motor_angles=value;states.append(s)
 assert all(abs(a-b)<1e-6 for a,b in zip(calibrated_angles(states,[4,5,6,7],[1.,-1.,1.,-1.],[.01,0.,0.,0.]),[.11,-.2,.3,-.4]))
 try:calibrated_angles(states,[4,4,6,7],[1.]*4,[0.]*4)
 except ValueError:pass
 else:raise AssertionError('Duplicate wheel IDs accepted')
 process=subprocess.Popen([sys.executable,'-m','rangermini_dynamic_semantic.ranger_feedback_bridge','--ros-args','-p','calibration_ready:=true'],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
 rclpy.init();node=Node('ranger_feedback_probe');pubs=[node.create_publisher(t,topic,1) for t,topic in [(ActuatorStateArray,'/actuator_state'),(MotionState,'/motion_state'),(Odometry,'/odom')]];received=[]
 node.create_subscription(String,'/motion_mode_debug',lambda m:received.append((time.monotonic(),json.loads(m.data))),10)
 def send(wheels=True):
  stamp=node.get_clock().now().to_msg();a=ActuatorStateArray();a.header.stamp=stamp;a.states=states;m=MotionState();m.header.stamp=stamp;m.motion_mode=1;o=Odometry();o.header.stamp=stamp;o.twist.twist.linear.x=.15;o.twist.twist.linear.y=.1
  for i,msg in enumerate((a,m,o)):
   if i or wheels:pubs[i].publish(msg)
 try:
  start=time.monotonic()
  while time.monotonic()-start<5:
   send();rclpy.spin_once(node,timeout_sec=.02)
  valid=[d for _,d in received if d['feedback_valid']]
  assert valid,{'process':process.poll(),'received':received[-5:],'matches':[p.get_subscription_count() for p in pubs]}
  answer=valid[-1];assert answer['mode']=='PARALLEL';assert all(abs(a-b)<1e-6 for a,b in zip(answer['wheel_steering_rad'],[.1,.2,.3,.4]));assert abs(answer['executed']['vy']-.1)<1e-9
  cutoff=time.monotonic();received.clear()
  while time.monotonic()-cutoff<1:
   send(False);rclpy.spin_once(node,timeout_sec=.02)
  stale=[t-cutoff for t,d in received if not d['feedback_valid'] and 'wheels' in d.get('reason','')]
  assert stale,'No invalid state after wheel feedback dropout'
  result={'kind':'synthetic ROS telemetry integration; not physical wheel feedback','four_distinct_channels_preserved':True,'calibration_scale_and_zero_pass':True,'duplicate_mapping_rejected':True,'mode_and_velocity_pass':True,'stale_wheel_feedback_invalidated':True,'first_invalid_after_dropout_s':min(stale)}
  Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result))
 finally:
  node.destroy_node();rclpy.shutdown();process.terminate()
  try:_,stderr=process.communicate(timeout=5)
  except subprocess.TimeoutExpired:process.kill();_,stderr=process.communicate()
  if stderr:print(stderr,file=sys.stderr)
if __name__=='__main__':main()
