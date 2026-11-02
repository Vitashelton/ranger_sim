#!/usr/bin/env python3
"""One sensor-closed-loop service-marker episode in the existing lab Gazebo.

Controller input: /odom, /scan, rendered /camera/image, and registered task
prior. Dynamic proxy poses are evaluator/event-manager data only. No VLM or
LLM quality is evaluated by this coded-marker task.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rangermini_dynamic_semantic.command_semantics_mppi import MPPIConfig
from rangermini_dynamic_semantic.observation_mppi import ObservationMPPI, ObservationConfig
from rangermini_dynamic_semantic.task_observation import (
    TaskScope, Evidence, EvidenceStore, QueryBudget, ViewConfig, view_score, project_target,
)


def yaw(q):
    return math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))


class Episode(Node):
    def __init__(self, args):
        super().__init__('unified_service_episode', parameter_overrides=[
            rclpy.parameter.Parameter('use_sim_time', value=True)])
        self.args = args
        self.pose = self.scan = self.image = None
        self.mode = 'STOPPED'
        self.started = self.arrival = None
        self.done = False
        self.last_plan = -1e9
        self.previous_active_mode = None
        self.switches = 0
        self.path_length = 0.
        self.previous_position = None
        self.pending = None
        self.pose_history = deque(maxlen=200)
        self.last_rgb = None
        self.query_id = 0
        self.last_query = -1e9
        self.output = Path(args.output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.trace = (self.output/'trace.jsonl').open('w')
        self.scope = TaskScope('ranger_lab_corridor_v1', 1, 'confirm_service_marker',
                               args.task, 'magenta_service_marker')
        self.store = EvidenceStore([args.task], ttl_s=2)
        self.budget = QueryBudget(60, 60*640*480*3, 60)
        target_x = 3.24 if args.task == 'lab_614' else 7.65
        # Registered map prior, not an online query of simulator target pose.
        self.hypotheses = ((target_x, -3.84, .65, 1.),)
        self.goal = (target_x, -2.85)
        self.cfg = MPPIConfig(samples_per_behavior=args.samples, horizon=16,
                              max_speed=.3, random_seed=args.seed+17,
                              allow_reverse=False, lidar_pitch=0.)
        self.obs = ObservationConfig(observation_weight=args.weight,
                    initial_switch_cost=args.c0, internal_switch_cost=args.c1,
                    lateral_candidates=args.method != 'A-Observe')
        self.planner = ObservationMPPI(self.cfg, self.obs)
        self.command = self.create_publisher(Twist, '/cmd_vel_safe', 10)
        self.create_subscription(Odometry, '/odom', self.on_pose, 10)
        self.create_subscription(LaserScan, '/scan', self.on_scan, 10)
        self.create_subscription(Image, '/camera/image', self.on_image, 10)
        self.create_subscription(String, '/motion_mode_debug', self.on_mode, 10)
        self.create_timer(.02, self.tick)
        (self.output/'manifest.json').write_text(json.dumps({
            'evidence_type':'gazebo_sensor_closed_loop_coded_marker',
            'arguments':vars(args), 'scope':asdict(self.scope),
            'controller':asdict(self.cfg), 'observation':asdict(self.obs),
            'view':asdict(ViewConfig()), 'registered_target_prior':self.hypotheses,
            'confirmation':{'min_magenta_pixels':80,'frame_max_age_s':.5,
                            'evidence_ttl_s':2,'response_latency_s':.3,
                            'color_rule':'r,b>60; g<0.45*min(r,b); abs(r-b)<0.35*max(r,b)'},
            'simulation_scan':'horizontal_2d_proxy',
            'semantic_model':'fixed_color_code_not_vlm',
            'localization':'gazebo_pose_derived_odometry',
            'world_sha256':hashlib.sha256(Path(args.world).read_bytes()).hexdigest(),
            'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [
                Path(__file__), Path(__file__).parents[1]/'rangermini_dynamic_semantic/observation_mppi.py',
                Path(__file__).parents[1]/'rangermini_dynamic_semantic/task_observation.py']},
        },indent=2))

    def now(self):
        return self.get_clock().now().nanoseconds*1e-9

    def on_pose(self, msg):
        self.pose = (msg.pose.pose.position.x, msg.pose.pose.position.y,
                     yaw(msg.pose.pose.orientation))
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9
        self.pose_history.append((stamp,self.pose))

    def on_scan(self, msg):
        self.scan = msg

    def on_image(self, msg):
        self.image = msg

    def on_mode(self, msg):
        try:
            self.mode = json.loads(msg.data)['mode']
        except (KeyError, ValueError):
            pass

    def publish(self, action):
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = action
        self.command.publish(msg)

    def finish(self, status, reason):
        self.publish((0.,0.,0.))
        self.done = True
        result = {'status':status,'reason':reason,
                  'elapsed_sim_s':self.now()-self.started if self.started is not None else None,
                  'arrival_sim_s':self.arrival-self.started if self.arrival is not None else None,
                  'confirmation_sim_s':self.now()-self.started if status=='SUCCEEDED' else None,
                  'path_length_m':self.path_length, 'active_mode_switches':self.switches,
                  'queries':self.budget.queries,'image_bytes':self.budget.image_bytes,
                  'evidence_wait_s':self.budget.wait_s,'final_pose':self.pose,
                  'evidence_state':self.store.state(self.scope,self.now()),
                  'task':self.args.task,'method':self.args.method,'seed':self.args.seed}
        (self.output/'result.json').write_text(json.dumps(result,indent=2))
        if self.last_rgb is not None:
            height, width = self.last_rgb.shape[:2]
            (self.output/'last_camera.ppm').write_bytes(
                f'P6\n{width} {height}\n255\n'.encode()+self.last_rgb.tobytes())
        self.trace.flush()
        print(json.dumps(result),flush=True)

    def tick(self):
        if self.done:
            self.publish((0.,0.,0.))
            return
        now = self.now()
        if self.pose is None or self.scan is None or self.image is None or now <= 0:
            self.publish((0.,0.,0.))
            return
        if self.started is None:
            self.started = now
        if now-self.started > self.args.deadline:
            self.finish('TIMEOUT','target_not_confirmed_before_deadline')
            return
        if now-self.last_plan < .12:
            return
        self.last_plan = now
        if self.previous_position is not None:
            self.path_length += math.hypot(self.pose[0]-self.previous_position[0],
                                           self.pose[1]-self.previous_position[1])
        self.previous_position = self.pose[:2]
        distance = math.hypot(self.pose[0]-self.goal[0],self.pose[1]-self.goal[1])
        if distance <= .42 and self.arrival is None:
            self.arrival = now
        frame_time = self.image.header.stamp.sec + self.image.header.stamp.nanosec*1e-9
        encoding = self.image.encoding.lower()
        pixels = 0
        centroid = None
        if encoding in ('rgb8','bgr8') and 0 <= now-frame_time <= .5:
            rows = np.frombuffer(bytes(self.image.data),dtype=np.uint8).reshape(self.image.height,self.image.step)
            rgb = rows[:,:self.image.width*3].reshape(self.image.height,self.image.width,3)
            if encoding == 'bgr8':
                rgb = rgb[:,:,::-1]
            self.last_rgb = rgb.copy()
            red, green, blue = [rgb[:,:,i].astype(np.float32) for i in range(3)]
            mask = ((red>60)&(blue>60)&
                (green<.45*np.minimum(red,blue))&
                (np.abs(red-blue)<.35*np.maximum(red,blue)))
            pixels = int(np.count_nonzero(mask))
            if pixels:
                py, px = np.nonzero(mask)
                centroid = (float(np.mean(px)),float(np.mean(py)))
        evidence_event = None
        if self.pending is not None and now >= self.pending.received_at:
            e = self.pending
            self.budget.settle(e.request_id, e.received_at-e.captured_at)
            accepted, reason = self.store.admit(e,self.scope,now)
            evidence_event = {'accepted':accepted,'reason':reason,'envelope':asdict(e)}
            self.pending = None
        if distance <= .42 and self.store.state(self.scope,now) == 'confirmed':
            self.trace.write(json.dumps({'timestamp':now,'pose':self.pose,'pixels':pixels,
                                        'evidence_event':evidence_event,'terminal':True})+'\n')
            self.finish('SUCCEEDED','region_reached_and_recent_coded_target_evidence')
            return
        if distance <= .42 and self.pending is None and now-self.last_query >= .5 and now-frame_time <= .5:
            self.query_id += 1
            rid = f'frame-{self.query_id}'
            if not self.budget.reserve(rid,len(self.image.data),1.):
                self.finish('STOPPED','observation_budget_exhausted')
                return
            # Receipt/capture timestamps and capture pose travel together.
            pose_stamp,capture_pose = min(self.pose_history,key=lambda entry:abs(entry[0]-frame_time))
            projection = project_target(capture_pose,self.hypotheses[0])
            geometry_ok = (abs(pose_stamp-frame_time)<=.10 and projection is not None and
                (centroid is None or math.hypot(centroid[0]-projection[0],centroid[1]-projection[1])<=60))
            self.store.register_request(rid,self.scope,frame_time,capture_pose)
            self.pending = Evidence(rid,self.scope,frame_time,now+.3,capture_pose,
                                    'confirmed' if pixels>=80 else 'uncertain',geometry_ok)
            self.last_query = now
        weight = self.args.weight
        if self.args.method == 'G-MPPI' or (self.args.method=='S-Observe' and self.arrival is None):
            weight = 0.
        self.planner.obs = ObservationConfig(observation_weight=weight,
            initial_switch_cost=self.args.c0,internal_switch_cost=self.args.c1,
            lateral_candidates=self.args.method!='A-Observe')
        scan = (self.scan.ranges,self.scan.angle_min,self.scan.angle_increment)
        # Geometric baseline has the same nominal destination-facing heading.
        plan = self.planner.plan(self.pose,self.goal,scan,current_mode=self.mode,
                                 goal_yaw=-math.pi/2,hypotheses=self.hypotheses)
        self.publish(plan['action'])
        active = plan['command_mode']
        if active!='STOPPED':
            if self.previous_active_mode is not None and active!=self.previous_active_mode:
                self.switches += 1
            self.previous_active_mode = active
        row = {'timestamp':now,'elapsed':now-self.started,'pose':self.pose,
               'distance':distance,'pixels':pixels,'frame_age_s':now-frame_time,
               'image_encoding':encoding,'rgb_channel_max':self.last_rgb.max(axis=(0,1)).tolist() if self.last_rgb is not None else None,
               'evidence_event':evidence_event,'evidence_state':self.store.state(self.scope,now),
               'plan':plan}
        self.trace.write(json.dumps(row)+'\n')
        self.trace.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--world',required=True)
    p.add_argument('--task',choices=['lab_614','lab_second'],default='lab_614')
    p.add_argument('--method',choices=['G-MPPI','S-Observe','A-Observe','O-MPPI'],default='O-MPPI')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--samples',type=int,default=16)
    p.add_argument('--weight',type=float,default=3)
    p.add_argument('--c0',type=float,default=.28)
    p.add_argument('--c1',type=float,default=.28)
    p.add_argument('--deadline',type=float,default=45)
    args, ros_args = p.parse_known_args()
    rclpy.init(args=ros_args)
    node = Episode(args)
    started_wall = time.monotonic()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node,timeout_sec=.1)
            if time.monotonic()-started_wall > 120:
                node.finish('INFRASTRUCTURE_FAILURE','wall_timeout_or_missing_sensors')
    finally:
        node.publish((0.,0.,0.))
        for _ in range(3):
            rclpy.spin_once(node,timeout_sec=.05)
        node.trace.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':
    main()
