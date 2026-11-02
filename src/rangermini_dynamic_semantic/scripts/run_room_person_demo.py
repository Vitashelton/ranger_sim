#!/usr/bin/env python3
"""Known-room, finite-viewpoint entry/search/approach capability probe.

No target coordinates are read from SDF. This probe uses preset task semantics;
it is not a VLM identity-recognition result or an open-world exploration method.
"""
import argparse,json,math,os,subprocess,sys,time
from pathlib import Path
from dataclasses import asdict,replace
import rclpy
PKG=Path(__file__).resolve().parents[1]
sys.path.insert(0,'/home/USER/ranger_sim_ws/artifacts/group_meeting_20261013/scripts')
import run_view_demo as demo
from run_unified_gazebo_suite import stop
from rangermini_dynamic_semantic.observation_mppi import ObservationMPPI,ObservationConfig

demo.INSTRUCTION='进入614实验室寻找工作人员，找到后保持安全距离并朝向他完成观察，不要倒车。'

class RoomPersonDemo(demo.ViewDemo):
    def __init__(self,args):
        args.scene='furnished';args.target_class='person';args.semantic_provider='preset'
        super().__init__(args)
        self.cfg=replace(self.cfg,max_speed=args.max_speed)
        self.wall_started=time.monotonic()
        registry=json.loads((PKG/'config/room_search_registry.json').read_text())
        self.region=registry['rooms']['614']
        self.waypoints=self.region['viewpoints'];self.waypoint=0
        self.phase='ENTER';self.ready=False;self.started=None;self.confirming_since=None
        self.events=(self.path/'task_events.jsonl').open('w')
        self.goal=tuple(self.waypoints[0]);self.target=None;self.goal_yaw=-math.pi/2
        self.preference={'provider':'preset','preference':dict(demo.PREFERENCE),'latency_ms':0,
                         'label':'受控入室找人任务；语义预设，未调用VLM'}
        self.planner=ObservationMPPI(self.cfg,ObservationConfig(observation_weight=3. if args.method=='view' else 0.,
                              arrival_radius=.18,lateral_candidates=args.method=='view'))
        (self.path/'preference.json').write_text(json.dumps(self.preference,ensure_ascii=False,indent=2))
        manifest=json.loads((self.path/'manifest.json').read_text())
        manifest.update({'task':'known_room_person_search','map_id':registry['map_id'],
            'known_room_registry':self.region,'target_coordinates_from_sdf':False,
            'target_source':'actual_person_detection_and_depth','person_standoff_m':2.6,
            'planner_config':asdict(self.cfg)})
        (self.path/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))

    def event(self,kind,**fields):
        self.events.write(json.dumps({'timestamp':self.now(),'elapsed':self.now()-self.started,'event':kind,
                                     'phase':self.phase,'pose':self.pose,**fields},ensure_ascii=False)+'\n');self.events.flush()

    def valid_grounding(self):
        g=self.ground_target()
        if g is None:return None
        x,y,_=g['point'];poly=self.region['polygon']
        if not (min(p[0] for p in poly)<=x<=max(p[0] for p in poly) and
                min(p[1] for p in poly)<=y<=max(p[1] for p in poly)):return None
        return g

    def tick(self):
        if self.done:return
        if self.pose is None or self.scan is None or self.rgb is None or self.depth is None:
            self.publish((0.,0.,0.));return
        now=self.now()
        if not self.ready:
            self.started=now;self.ready=True;self.event('TASK_STARTED',goal=self.goal)
        if now-self.started>80:
            self.finish('TASK_TIMEOUT');return
        if now-self.last_plan<.12:return
        self.last_plan=now
        g=self.valid_grounding()
        if self.phase!='ENTER' and g is not None:
            first=self.target is None;self.target=g
            if first:
                (self.path/'initial_grounding.json').write_text(json.dumps(g,indent=2));self.event('PERSON_CANDIDATE_OBSERVED',grounding=g)
        distance=math.hypot(self.goal[0]-self.pose[0],self.goal[1]-self.pose[1])
        if self.phase in ('ENTER','SEARCH') and distance<=.18:
            if self.phase=='ENTER':
                self.phase='SEARCH';self.event('ENTERED_ROOM')
            self.waypoint+=1
            if self.waypoint>=2 and self.target is not None:
                self.phase='REOBSERVE';self.reobserve_start=now;self.goal=tuple(self.pose[:2])
                self.event('REOBSERVE_LAST_SEEN_REGION',prior_age_s=now-self.target['capture_time'])
            elif self.waypoint<len(self.waypoints):
                self.goal=tuple(self.waypoints[self.waypoint]);self.event('SEARCH_VIEWPOINT',goal=self.goal)
            elif self.target is not None:
                self.phase='REOBSERVE';self.reobserve_start=now;self.goal=tuple(self.pose[:2])
                self.event('REOBSERVE_LAST_SEEN_REGION',prior_age_s=now-self.target['capture_time'])
            else:
                self.finish('SEARCH_EXHAUSTED_UNCONFIRMED');return
        goal_yaw=-math.pi/2
        fresh=self.target is not None and now-self.target['capture_time']<=1.5
        if self.phase=='REOBSERVE':
            if g is not None:
                self.phase='APPROACH';self.event('PERSON_REOBSERVED')
            elif now-self.reobserve_start>12:
                self.finish('REOBSERVATION_UNCONFIRMED');return
            else:
                tx,ty,_=self.target['point'];goal_yaw=math.atan2(ty-self.pose[1],tx-self.pose[0])
        if self.phase=='APPROACH':
            if not fresh:
                self.publish((0.,0.,0.))
                if self.target is None or now-self.target['capture_time']>5:
                    self.finish('PERSON_CONFIRMATION_UNAVAILABLE')
                return
            tx,ty,_=self.target['point'];dx,dy=tx-self.pose[0],ty-self.pose[1];pd=math.hypot(dx,dy)
            goal_yaw=math.atan2(dy,dx)
            self.goal=(tx-2.6*dx/max(pd,1e-6),ty-2.6*dy/max(pd,1e-6))
            heading=abs(math.atan2(math.sin(goal_yaw-self.pose[2]),math.cos(goal_yaw-self.pose[2])))
            if g is not None and 2.35<=pd<=2.9 and heading<=.25:
                self.publish((0.,0.,0.))
                if self.confirming_since is None:
                    self.confirming_since=now;self.event('HOLD_FOR_CURRENT_FRAME')
                elif (now-self.confirming_since>=.6 and g['capture_time']>=self.confirming_since and
                      abs(math.atan2(math.sin(g['capture_pose'][2]-self.pose[2]),math.cos(g['capture_pose'][2]-self.pose[2])))<.1):
                    self.event('PERSON_OBSERVATION_CONFIRMED',distance_m=pd,heading_error_rad=heading,grounding=g)
                    self.finish('PERSON_FOUND_AND_OBSERVED')
                return
            self.confirming_since=None
        # Search follows registered viewpoint priors; target coordinates are
        # added only after a current image/depth observation inside the room.
        hypotheses=(tuple(self.target['point'])+(1.,),) if fresh and self.args.method=='view' else ()
        plan=self.planner.plan(self.pose,self.goal,(self.scan.ranges,self.scan.angle_min,self.scan.angle_increment),
            current_mode=self.mode,goal_yaw=goal_yaw,hypotheses=hypotheses)
        self.publish(plan['action'])
        self.trace.write(json.dumps({'timestamp':now,'elapsed':now-self.started,'pose':self.pose,'phase':self.phase,
            'goal':self.goal,'goal_distance':math.hypot(self.goal[0]-self.pose[0],self.goal[1]-self.pose[1]),
            'pixels':int(self.mask.sum()),'target_grounding':self.target,'detection':self.detection,'plan':plan})+'\n');self.trace.flush()

    def finish(self,status):
        super().finish(status)
        p=self.path/'result.json';a=json.loads(p.read_text());a.update({'task':'known_room_person_search','phase':self.phase,
            'elapsed_wall_s':time.monotonic()-self.wall_started,'max_speed_mps':self.cfg.max_speed,
            'semantic_identity_confirmed_by_vlm':False,'confirmation':'current COCO person detection, RGB-D range and camera heading; no biometric identity claim'})
        p.write_text(json.dumps(a,ensure_ascii=False,indent=2));self.events.flush()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--controller',action='store_true')
    p.add_argument('--world');p.add_argument('--method',default='view',choices=['view','geometric'])
    p.add_argument('--max-speed',type=float,default=.35,help='Simulation-only speed limit in m/s; prior footage used 0.18.')
    args=p.parse_args()
    if not (0 < args.max_speed <= .6):p.error('--max-speed must be in (0, 0.6] for this simulation probe')
    if args.controller:
        rclpy.init();node=RoomPersonDemo(args);start=time.monotonic()
        try:
            while rclpy.ok() and not node.done:
                rclpy.spin_once(node,timeout_sec=.05)
                if time.monotonic()-start>150:node.finish('WALL_TIMEOUT')
        finally:
            node.publish((0.,0.,0.));node.trace.close();node.frame_log.close();node.overview_log.close();node.events.close()
            node.pool.shutdown(wait=False);node.destroy_node();rclpy.shutdown()
        return
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    import shutil
    world=out/'world.sdf';shutil.copy2(PKG/'worlds/ranger_lab_corridor_room_search.sdf',world)
    env=dict(os.environ,ROS_DOMAIN_ID='99',RMW_IMPLEMENTATION='rmw_fastrtps_cpp',IGN_PARTITION='ranger_room_person',IGN_IP='127.0.0.1',ROS_LOG_DIR=str(out/'ros_logs'))
    env.pop('CYCLONEDDS_URI',None);children=[]
    def spawn(cmd,name):
        log=(out/(name+'.log')).open('w');proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);children.append((proc,log));return proc
    try:
        spawn(['ign','gazebo','-s','-r',str(world)],'gazebo');time.sleep(2)
        spawn(['ros2','run','ros_gz_bridge','parameter_bridge','/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
            '/model/rangermini_2_0/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist','/model/rangermini_2_0/pose@geometry_msgs/msg/PoseStamped[ignition.msgs.Pose',
            '/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan','/camera/image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/camera/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image','/scene/navigation@sensor_msgs/msg/Image[ignition.msgs.Image',
            '--ros-args','-r','/model/rangermini_2_0/cmd_vel:=/cmd_vel_gazebo'],'bridge')
        spawn(['ros2','run','rangermini_dynamic_semantic','gazebo_odom_adapter'],'odom')
        spawn(['ros2','run','rangermini_dynamic_semantic','ranger_motion_adapter','--ros-args','-p','input_topic:=/cmd_vel_safe','-p','output_topic:=/cmd_vel_gazebo'],'adapter')
        proc=spawn(['python3',str(Path(__file__).resolve()),'--controller','--world',str(world),'--method',args.method,'--max-speed',str(args.max_speed),'--output',str(out/'episode')],'controller')
        proc.wait(timeout=165)
        result=out/'episode/result.json';print(result.read_text() if result.exists() else 'NO_RESULT')
    finally:stop(children)
if __name__=='__main__':main()
