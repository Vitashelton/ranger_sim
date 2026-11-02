"""Read-only Ranger ROS feedback bridge for the MPPI vehicle-state contract.

This node publishes state only. Channel order, scale and zero offset must be
calibrated on the selected chassis before calibration_ready is enabled.
"""
import json
import math
import time

MODES = {0: 'DUAL_ACKERMANN', 1: 'PARALLEL', 2: 'SPINNING', 3: 'SIDE_SLIP'}


def calibrated_angles(states, channel_ids, scales, offsets):
    if not (len(channel_ids) == len(scales) == len(offsets) == 4):
        raise ValueError('Expected four wheel IDs, scales and offsets')
    if len(set(channel_ids)) != 4:
        raise ValueError('Wheel IDs must identify four distinct SDK channels')
    by_id = {int(s.id): s.motor.motor_angles for s in states}
    if len(by_id) != len(states):
        raise ValueError('Duplicate feedback channel IDs')
    angles = [float(by_id[int(i)])*scale+zero
              for i, scale, zero in zip(channel_ids, scales, offsets)]
    if not all(math.isfinite(v) for v in [*angles, *scales, *offsets]):
        raise ValueError('Non-finite wheel feedback or calibration')
    if any(abs(v) < 1e-12 for v in scales):
        raise ValueError('Angle scales must be nonzero')
    return angles


def main(args=None):
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import ExternalShutdownException
    from rclpy.clock import Clock, ClockType
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    from ranger_msgs.msg import ActuatorStateArray, MotionState
    from .ranger_4ws_kinematics import Ranger4WSKinematics
    from .ranger_motion_adapter import ranger_mini_v2_command

    class FeedbackBridge(Node):
        def __init__(self):
            super().__init__('ranger_mppi_feedback_bridge')
            for name,value in [('wheel_ids',[4,5,6,7]),('angle_scales',[1.,1.,1.,1.]),
                ('angle_offsets',[0.,0.,0.,0.]),('calibration_ready',False),
                ('max_feedback_age',.5),('output_topic','/motion_mode_debug')]:
                self.declare_parameter(name,value)
            self.latest = {};self.command = None
            self.model = Ranger4WSKinematics()
            self.pub = self.create_publisher(String,self.get_parameter('output_topic').value,1)
            for topic,kind,msg_type in [('/actuator_state','wheels',ActuatorStateArray),
                    ('/motion_state','mode',MotionState),('/odom','odom',Odometry)]:
                self.create_subscription(msg_type,topic,lambda msg,k=kind:self.capture(k,msg),1)
            self.create_subscription(Twist,'/cmd_vel_safe',lambda msg:setattr(self,'command',(msg,time.monotonic())),1)
            self.create_timer(.05,self.publish_state,clock=Clock(clock_type=ClockType.STEADY_TIME))
            self.get_logger().info('Measured feedback bridge running')

        def capture(self,kind,msg):
            stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
            self.latest[kind]=(msg,stamp,time.monotonic())

        def publish_state(self):
            now=self.get_clock().now().nanoseconds*1e-9;wall=time.monotonic()
            age=float(self.get_parameter('max_feedback_age').value)
            result={'feedback_valid':False,'feedback_source':'ranger_ros_measured',
                    'stamp_ros_s':now,'wheel_steering_rad':[],'mode':'UNKNOWN',
                    'executed':{},'steering_transition_stop':False}
            try:
                if not self.get_parameter('calibration_ready').value:
                    raise ValueError('Wheel order, units, sign and zero require calibration')
                for kind in ('wheels','mode','odom'):
                    _,stamp,received=self.latest[kind]
                    if not (-.05 <= now-stamp <= age and 0 <= wall-received <= age):
                        raise ValueError(kind+' feedback expired or clock regressed')
                wheels,mode,odom=(self.latest[k][0] for k in ('wheels','mode','odom'))
                angles=calibrated_angles(wheels.states,*[self.get_parameter(k).value for k in ('wheel_ids','angle_scales','angle_offsets')])
                mode_name=MODES[int(mode.motion_mode)]
                vel=odom.twist.twist;vx,vy,wz=vel.linear.x,vel.linear.y,vel.angular.z
                if not all(math.isfinite(v) for v in (vx,vy,wz)):
                    raise ValueError('Non-finite measured body velocity')
                aligning=False
                if self.command and wall-self.command[1] <= age:
                    msg=self.command[0]
                    if not all(math.isfinite(v) for v in (msg.linear.x,msg.linear.y,msg.angular.z)):
                        raise ValueError('Non-finite body command')
                    command=ranger_mini_v2_command(msg.linear.x,msg.linear.y,msg.angular.z)
                    self.model.angles=angles[:]
                    targets,_=self.model._target_wheels(*command[:3],command[3])
                    error=max(abs(math.atan2(math.sin(a-b),math.cos(a-b))) for a,b in zip(targets,angles))
                    aligning=(command[3]!='STOPPED' and (mode_name!=command[3] or error>.025)
                              and math.hypot(vx,vy)<.02 and abs(wz)<.02)
                result.update(feedback_valid=True,stamp_ros_s=min(self.latest[k][1] for k in ('wheels','mode','odom')),
                    wheel_steering_rad=angles,mode=mode_name,executed={'vx':vx,'vy':vy,'wz':wz},
                    steering_transition_stop=aligning,kinematic_model='measured wheel angles and odometry')
            except (KeyError,ValueError,TypeError) as error:
                result['reason']=str(error)
            self.pub.publish(String(data=json.dumps(result,allow_nan=False)))

    rclpy.init(args=args);node=FeedbackBridge()
    try:rclpy.spin(node)
    except (KeyboardInterrupt,ExternalShutdownException):pass
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()

if __name__=='__main__':main()
