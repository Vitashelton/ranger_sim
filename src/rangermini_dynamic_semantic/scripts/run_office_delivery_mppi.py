#!/usr/bin/env python3
"""Office A delivery experiment: geometry/VLM x Ackermann/Ranger 4WS."""
import argparse, json, math, os, sys, time, hashlib, subprocess
from pathlib import Path

PKG=Path(__file__).resolve().parents[1]
ROOT=PKG.parents[1]
sys.path.insert(0,str(PKG));sys.path.insert(0,str(ROOT/'artifacts/group_meeting_20261013/scripts'))
import run_view_demo as demo
from generate_office_delivery_world import make_world, LAYOUTS
from rangermini_dynamic_semantic.observation_mppi import ObservationMPPI,ObservationConfig
from rangermini_dynamic_semantic.command_semantics_mppi import MPPIConfig
from rangermini_dynamic_semantic.visual_behavior_selector import select_visual_trajectory

INSTRUCTION=('把资料送到 Office A（619 室）门内的交接区：沿走廊通行，从敞开的门洞进入，'
    '在门内交接位停稳。途中经过等候同事和器材推车。只在已生成的可行路线中选择，不要输出底盘控制量。')
# The task ends at a fixed, unobstructed handoff zone immediately inside the
# office. It tests corridor navigation plus door entry without requiring a
# separate object-detection task on a desk.
GOAL=(12.80,1.15); GOAL_YAW=math.pi/2
PALETTE=[(255,80,55),(40,210,255),(95,235,75),(235,70,235),(30,145,255),(255,205,45),
         (195,105,255),(90,230,205),(225,145,80),(100,155,255),(190,230,80),(255,130,180)]

def wrap(a): return math.atan2(math.sin(a),math.cos(a))

def rect_circle_gap(pose,center,radius,length=.751,width=.5):
    x,y,yaw=pose;dx,dy=center[0]-x,center[1]-y;c,s=math.cos(yaw),math.sin(yaw)
    bx,by=c*dx+s*dy,-s*dx+c*dy
    return max(0.,math.hypot(max(abs(bx)-length/2,0.),max(abs(by)-width/2,0.))-radius)

def rect_rect_gap(a,b,alen,awid,blen,bwid):
    def axes(box):return [(math.cos(box[2]),math.sin(box[2])),(-math.sin(box[2]),math.cos(box[2]))]
    def proj(box,length,width,axis):
        x,y,yaw=box;u=(math.cos(yaw),math.sin(yaw));v=(-u[1],u[0])
        c=x*axis[0]+y*axis[1]
        r=length/2*abs(u[0]*axis[0]+u[1]*axis[1])+width/2*abs(v[0]*axis[0]+v[1]*axis[1])
        return c-r,c+r
    sep=0.
    for axis in axes(a)+axes(b):
        alo,ahi=proj(a,alen,awid,axis);blo,bhi=proj(b,blen,bwid,axis)
        sep=max(sep,blo-ahi,alo-bhi)
    return max(0.,sep)

class OfficeDeliveryPilot(demo.ViewDemo):
    def __init__(self,args):
        super().__init__(args)
        self.layout=args.layout;self.condition=args.condition
        self.clearance_threshold=float(args.clearance_threshold)
        self.trolley_clearance_threshold=float(args.trolley_clearance_threshold)
        self.goal=GOAL;self.goal_yaw=GOAL_YAW;self.initial_yaw=0.
        # First approach on the open-leaf side, then enter directly to the
        # indoor handoff zone. Both selectors share these task subgoals.
        self.passage_points=[(12.80,-1.15),(12.80,1.15)]
        # The second point clears the open door leaf. Its tighter tolerance is
        # needed because the Ranger footprint is longer than the waypoint
        # radius used for ordinary corridor subgoals.
        self.passage_radii=[.38,.16,.18]
        self.passage_index=0
        self.fourws=args.condition.endswith('4WS');self.semantic=args.condition.startswith('VLM')
        self.cfg=MPPIConfig(samples_per_behavior=12,horizon=60,dt=.12,max_speed=.35,
            min_turn_radius=.4764,wheelbase=.494,body_length=.751,body_width=.5,
            lidar_pitch=0.,random_seed=args.seed,allow_reverse=False)
        self.planner=ObservationMPPI(self.cfg,ObservationConfig(observation_weight=0.,
            arrival_radius=.24,collision_margin=.035,lateral_candidates=self.fourws,
            ranger_4ws_modes=self.fourws,reference_weight=18.,
            enforce_approach_progress=False,stop_action_penalty=2.0))
        self.reference=None;self.reference_start=None;self.pending=None
        self.pending_candidates={};self.query_index=0;self.last_plan=-1e9
        self.candidate_snapshot_saved=False
        self.vlm_errors=0;self.vlm_fallback=False
        self.last_progress_time=None;self.last_progress_pose=None
        self.last_saved_rgb=-1e9;self.last_saved_overview=-1e9;self.layout_truth=LAYOUTS[self.layout]
        self.events=(self.path/'selection_events.jsonl').open('w')
        manifest={'instruction':INSTRUCTION,'condition':self.condition,'layout':self.layout,'seed':args.seed,
            'start_pose':[3.,-2.1,0.],'goal_pose':[GOAL[0],GOAL[1],GOAL_YAW],
            'service_target':{'object':'indoor document handoff zone','room':'Office A / room 619',
                'approach_pose':[GOAL[0],GOAL[1],GOAL_YAW],
                'doorway_plane_y_m':-.30,'doorway_crossing_required':True},
            'controller':'receding-horizon MPPI with optional DeepSeek selection of projected feasible rollouts',
            'motion_model':'RangerMini four-wheel rigid-body pure-rolling kinematics; finite steering rate; stationary mode transition',
            'ranger_4ws':self.fourws,'semantic_candidate_selection':self.semantic,
            'visible_payload':'non-colliding document packet and sheets attached to base_link',
            'person_proxy_radius_m':.18,'body_size_m':[.751,.5],
            'person_proxy_clearance_reference_m':self.clearance_threshold,
            'trolley_proxy_clearance_reference_m':self.trolley_clearance_threshold,
            'world_sha256':hashlib.sha256(Path(args.world).read_bytes()).hexdigest(),
            'layout_truth_evaluator_only':self.layout_truth}
        (self.path/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))

    def on_rgb(self,msg):
        if msg.encoding.lower() not in ('rgb8','bgr8'):return
        arr=demo.np.frombuffer(bytes(msg.data),dtype=demo.np.uint8).reshape(msg.height,msg.step)
        rgb=arr[:,:msg.width*3].reshape(msg.height,msg.width,3)
        if msg.encoding.lower()=='bgr8':rgb=rgb[:,:,::-1]
        self.rgb=rgb.copy();self.frame_time=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        self.mask=demo.np.zeros(self.rgb.shape[:2],dtype=bool)
        if self.frame_time-self.last_saved_rgb>=1.:
            demo.PILImage.fromarray(self.rgb).save(self.path/'frames'/f'{self.frame_index:05d}.jpg',quality=90)
            self.frame_log.write(json.dumps({'frame':self.frame_index,'sim_time':self.frame_time,'pose':self.pose})+'\n');self.frame_log.flush()
            self.frame_index+=1;self.last_saved_rgb=self.frame_time

    def on_overview(self,msg):
        if msg.encoding.lower() not in ('rgb8','bgr8'):return
        arr=demo.np.frombuffer(bytes(msg.data),dtype=demo.np.uint8).reshape(msg.height,msg.step)
        rgb=arr[:,:msg.width*3].reshape(msg.height,msg.width,3)
        if msg.encoding.lower()=='bgr8':rgb=rgb[:,:,::-1]
        t=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        if t-self.last_saved_overview>=.5:
            demo.PILImage.fromarray(rgb).save(self.path/'overview_frames'/f'{self.overview_index:05d}.jpg',quality=88)
            self.overview_log.write(json.dumps({'frame':self.overview_index,'sim_time':t,'pose':self.pose})+'\n');self.overview_log.flush()
            self.overview_index+=1;self.last_saved_overview=t

    def project(self,trajectory):
        if self.pose is None:return []
        x,y,psi=self.pose;f=640/(2*math.tan(1.396/2));out=[]
        for px,py,*_ in trajectory:
            dx,dy=px-x,py-y;forward=math.cos(psi)*dx+math.sin(psi)*dy-.34
            lateral=-math.sin(psi)*dx+math.cos(psi)*dy
            if forward<=.08:continue
            u=320-lateral*f/forward;v=240+(.43-.04)*f/forward
            if 0<=u<640 and 0<=v<480:out.append((int(u),int(v)))
        return out

    def candidate_view(self,candidates):
        image=self.rgb.copy();metadata=[];lookup={}
        for i,c in enumerate(candidates):
            # A stationary rollout is not a route choice. In particular, do
            # not let the visual selector satisfy a clearance instruction by
            # selecting STOP far from the delivery point.
            if c.get('motion_behavior')=='STOP' and math.dist(self.pose[:2],self.goal)>.32:
                continue
            cid=c['candidate_id'];color=PALETTE[i%len(PALETTE)];pixels=self.project(c['trajectory'])
            if len(pixels)<2:continue
            demo.cv2.polylines(image,[demo.np.asarray(pixels,demo.np.int32)],False,color,4,lineType=demo.cv2.LINE_AA)
            x,y=pixels[min(len(pixels)-1,3)]
            demo.cv2.putText(image,cid,(x,y),demo.cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,demo.cv2.LINE_AA)
            metadata.append({'candidate_id':cid,'feasible':True,'motion_behavior':c['motion_behavior'],
                'geometric_cost':round(c['cost'],3),'scan_clearance_m':round(c['min_clearance_m'],3),'legend_rgb':color})
            lookup[cid]=c
        if not metadata:return None,metadata,lookup
        y0=18
        for item in metadata:
            label=item['candidate_id']+' '+item['motion_behavior']
            demo.cv2.putText(image,label,(8,y0),
                demo.cv2.FONT_HERSHEY_SIMPLEX,.42,item['legend_rgb'],1,demo.cv2.LINE_AA);y0+=16
        ok,buf=demo.cv2.imencode('.jpg',image[:,:,::-1],[int(demo.cv2.IMWRITE_JPEG_QUALITY),92])
        if not ok:return None,metadata,lookup
        demo.PILImage.fromarray(image).save(self.path/f'selection_{self.query_index:02d}.jpg',quality=94)
        (self.path/f'selection_{self.query_index:02d}_candidates.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
        full_candidates=[]
        for item in metadata:
            candidate=lookup[item['candidate_id']]
            full_candidates.append({**item,'trajectory':candidate['trajectory'],
                'modes':candidate.get('modes',[])})
        (self.path/f'selection_{self.query_index:02d}_rollouts.json').write_text(
            json.dumps(full_candidates,ensure_ascii=False,separators=(',',':')))
        return bytes(buf),metadata,lookup

    def start_query(self,plan,now):
        self.query_index+=1
        jpg,metadata,lookup=self.candidate_view(plan.get('candidate_population',[]))
        if jpg is None or not metadata:
            self.query_index-=1
            return False
        self.pending_candidates=lookup
        self.pending=self.pool.submit(select_visual_trajectory,jpg,INSTRUCTION,metadata,timeout_s=90)
        self.query_started=now
        self.events.write(json.dumps({'event':'query_started','query':self.query_index,'sim_time':now,
            'pose':self.pose,'candidate_ids':list(lookup)},ensure_ascii=False)+'\n');self.events.flush()
        return True

    def reference_segment(self,now):
        if not self.reference:return None
        i=max(0,int((now-self.reference_start)/self.cfg.dt))
        if i>=len(self.reference)-4:return None
        return self.reference[i:]

    def tick(self):
        if self.done or self.pose is None or self.scan is None or self.rgb is None:return
        now=self.now()
        if self.started is None:self.started=now
        if now-self.started>600:self.finish('TIMEOUT');return
        dist=math.dist(self.pose[:2],self.goal);yawerr=abs(wrap(self.pose[2]-self.goal_yaw))
        if dist<=.32 and yawerr<=math.radians(18):self.finish('ARRIVED');return
        if (self.last_progress_pose is None or math.dist(self.pose[:2],self.last_progress_pose)>.05):
            self.last_progress_pose=tuple(self.pose[:2]);self.last_progress_time=now
        elif self.last_progress_time is not None and now-self.last_progress_time>15.0 and not (
                self.semantic and (self.pending is not None or self.reference_segment(now) is None)):
            self.finish('STUCK');return
        while self.passage_index<len(self.passage_points) and math.dist(
                self.pose[:2],self.passage_points[self.passage_index])<=self.passage_radii[self.passage_index]:
            self.passage_index+=1
        subgoal=(self.passage_points[self.passage_index]
                 if self.passage_index<len(self.passage_points) else self.goal)
        subgoal_yaw=self.goal_yaw if self.passage_index==len(self.passage_points) else None
        if now-self.last_plan<.12:return
        self.last_plan=now;ref=self.reference_segment(now)
        args=dict(current_mode=self.mode,goal_yaw=subgoal_yaw,
            current_steering_angles=self.mode_payload.get('wheel_steering_rad'),
            current_steering_transition=self.mode_payload.get('steering_transition_stop',False),
            reference_trajectory=ref)
        scan=(self.scan.ranges,self.scan.angle_min,self.scan.angle_increment)
        plan=self.planner.plan(self.pose,subgoal,scan,**args)
        need_selection=self.semantic and (self.reference is None or ref is None)
        if not self.semantic and not self.candidate_snapshot_saved:
            self.save_rollout_snapshot(plan,1)
            self.candidate_snapshot_saved=True
        if need_selection and self.pending is None:
            if not self.start_query(plan,now):
                self.semantic=False;self.vlm_fallback=True
        if need_selection and self.pending is not None and not self.pending.done():
            # Hold the robot while the low-rate visual decision is outstanding;
            # otherwise the chassis would execute a geometric route before the
            # selected rollout returns, making the selection causally irrelevant.
            self.publish((0.,0.,0.))
            self.events.flush()
            return
        if self.pending is not None and self.pending.done():
            try:
                response=self.pending.result();cid=response['decision']['candidate_id'];chosen=self.pending_candidates[cid]
                traj=chosen['trajectory'];offset=int(demo.np.argmin([math.dist(self.pose[:2],p[:2]) for p in traj]))
                self.reference=traj[offset:];self.reference_start=now;self.preference=response
                self.last_progress_pose=tuple(self.pose[:2]);self.last_progress_time=now
                self.events.write(json.dumps({'event':'selected','query':self.query_index,'sim_time':now,
                    'pose':self.pose,'candidate_id':cid,'candidate_behavior':chosen['motion_behavior'],
                    'reason':response['decision']['reason'],'latency_s':response['latency_s'],
                    'trajectory_offset':offset},ensure_ascii=False)+'\n');self.events.flush()
                (self.path/f'selection_{self.query_index:02d}_response.json').write_text(json.dumps(response,ensure_ascii=False,indent=2))
                self.pending=None;self.pending_candidates={};ref=self.reference_segment(now)
                args['reference_trajectory']=ref
                plan=self.planner.plan(self.pose,subgoal,scan,**args)
            except Exception as exc:
                self.vlm_errors+=1
                self.events.write(json.dumps({'event':'selection_error','query':self.query_index,'sim_time':now,
                    'error':str(exc)},ensure_ascii=False)+'\n');self.events.flush()
                self.pending=None;self.pending_candidates={}
                if self.vlm_errors>=2:
                    self.semantic=False;self.vlm_fallback=True
                    self.last_progress_pose=tuple(self.pose[:2]);self.last_progress_time=now
                    self.events.write(json.dumps({'event':'fallback_to_geometry','sim_time':now,
                        'reason':'two visual selector failures'},ensure_ascii=False)+'\n');self.events.flush()
                else:
                    self.publish((0.,0.,0.))
                    return
        self.publish(plan['action'])
        compact={k:v for k,v in plan.items() if k not in
            ('candidate_population','candidate_summary','selected_trajectory','predicted_executed_controls')}
        self.trace.write(json.dumps({'timestamp':now,'elapsed':now-self.started,'pose':self.pose,
            'goal_distance':dist,'goal_yaw_error_rad':yawerr,'local_subgoal':subgoal,
            'subgoal_index':self.passage_index,'actuation':self.mode_payload,
            'reference_active':ref is not None,'plan':compact})+'\n');self.trace.flush()

    def finish(self,status):
        self.publish((0.,0.,0.));self.done=True;self.trace.flush();self.frame_log.flush()
        rows=[json.loads(s) for s in (self.path/'trace.jsonl').read_text().splitlines()]
        length=sum(math.dist(a['pose'][:2],b['pose'][:2]) for a,b in zip(rows,rows[1:]))
        poserr=math.dist(self.pose[:2],self.goal) if self.pose else None
        yawerr=abs(wrap(self.pose[2]-self.goal_yaw)) if self.pose else None
        person=self.layout_truth['person'];cart=self.layout_truth['cart'];poses=[r['pose'] for r in rows]
        pgap=min((rect_circle_gap(p,person,.18) for p in poses),default=None)
        cgap=min((rect_rect_gap(p,(*cart,0.),.751,.5,.70,.42) for p in poses),default=None)
        modes=[r.get('actuation',{}).get('mode','UNKNOWN') for r in rows]
        ev=(self.path/'selection_events.jsonl').read_text().splitlines()
        doorway_crossed=any(p[1]>.65 and 12.30<p[0]<13.28 for p in poses)
        room_pose_error=math.dist(self.pose[:2],GOAL) if self.pose else None
        result={'status':status,'condition':self.condition,'layout':self.layout,'seed':self.args.seed,
            'elapsed_sim_s':self.now()-self.started if self.started else 0.,'path_length_m':length,'final_pose':self.pose,
            'delivery_position_error_m':poserr,'delivery_heading_error_deg':math.degrees(yawerr) if yawerr is not None else None,
            'doorway_crossed_into_room_619':doorway_crossed,
            'handoff_zone_position_error_m':room_pose_error,
            'person_proxy_min_clearance_m':pgap,'trolley_proxy_min_clearance_m':cgap,
            'person_clearance_threshold_m':self.clearance_threshold,
            'person_clearance_compliant':pgap is not None and pgap>=self.clearance_threshold,
            'trolley_clearance_threshold_m':self.trolley_clearance_threshold,
            'trolley_clearance_compliant':cgap is not None and cgap>=self.trolley_clearance_threshold,
            'mode_counts':{m:modes.count(m) for m in sorted(set(modes))},
            'mode_switches':sum(a!=b for a,b in zip(modes,modes[1:])),
            'visual_queries':self.query_index,'visual_selections':sum(json.loads(s).get('event')=='selected' for s in ev),
            'visual_selector_errors':self.vlm_errors,'visual_fallback_to_geometry':self.vlm_fallback,
            'frame_count':self.frame_index,'overview_frame_count':self.overview_index,
            'task_success':status=='ARRIVED' and doorway_crossed and poserr is not None and poserr<=.32 and
                yawerr<=math.radians(18)}
        (self.path/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(json.dumps(result,ensure_ascii=False),flush=True)

    def save_rollout_snapshot(self,plan,index):
        candidates=plan.get('candidate_population',[])
        compact=[{'candidate_id':c['candidate_id'],'motion_behavior':c['motion_behavior'],
            'feasible':c.get('feasible',True),'geometric_cost':c.get('cost'),
            'scan_clearance_m':c.get('min_clearance_m'),'trajectory':c.get('trajectory',[]),
            'modes':c.get('modes',[])} for c in candidates]
        (self.path/f'selection_{index:02d}_rollouts.json').write_text(
            json.dumps(compact,ensure_ascii=False,separators=(',',':')))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--condition',choices=['GEO_ACK','GEO_4WS','VLM_ACK','VLM_4WS'],default='GEO_4WS')
    p.add_argument('--layout',choices=sorted(LAYOUTS),default='east_weave');p.add_argument('--seed',type=int,default=17)
    p.add_argument('--clearance-threshold',type=float,default=.45,
        help='person-proxy clearance reference in metres; diagnostic only')
    p.add_argument('--trolley-clearance-threshold',type=float,default=.35,
        help='trolley-proxy clearance reference in metres; diagnostic only')
    p.add_argument('--ros-domain-id',type=int,default=99)
    p.add_argument('--ign-partition',default='ranger_group_meeting')
    p.add_argument('--controller',action='store_true');p.add_argument('--world')
    for name,default in [('method','geometric'),('scene','furnished'),('target-class','person'),('semantic-provider','preset')]:p.add_argument('--'+name,default=default)
    args=p.parse_args()
    if not args.controller:
        demo.__file__=__file__;demo.create_world=lambda path,scene:make_world(args.layout,path)
        original=demo.subprocess.Popen
        # The active shell may be sourced from ranger_ws. Build a clean ROS
        # environment for this simulation from Humble plus this workspace's
        # overlay, so ros2 run cannot silently resolve the hardware package.
        clean_base={'HOME':os.environ.get('HOME','/tmp'),'USER':os.environ.get('USER','zbx'),
            'LOGNAME':os.environ.get('USER','zbx'),'PATH':'/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
            'LANG':'C.UTF-8'}
        setup_cmd=("source /opt/ros/humble/setup.bash && "
                   f"source {ROOT/'install/local_setup.bash'} && env -0")
        setup=subprocess.run(['bash','--noprofile','--norc','-c',setup_cmd],env=clean_base,
            stdout=subprocess.PIPE,check=True).stdout
        sim_env=dict((k.decode(),v.decode()) for k,v in
            (entry.split(b'=',1) for entry in setup.split(b'\0') if b'=' in entry))
        passthrough=('DISPLAY','XAUTHORITY','QT_QPA_PLATFORM','DEEPSEEK_API_KEY',
            'IGN_GAZEBO_RESOURCE_PATH','IGN_PARTITION','IGN_IP','RMW_IMPLEMENTATION','ROS_DOMAIN_ID',
            'HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY','http_proxy','https_proxy','all_proxy','no_proxy',
            'SSL_CERT_FILE','REQUESTS_CA_BUNDLE')
        class SupervisedProcess:
            def __init__(self,process):self.process=process
            def __getattr__(self,name):return getattr(self.process,name)
            def wait(self,timeout=None):return self.process.wait(timeout=max(timeout or 0,720))
        def spawn(command,**kwargs):
            inherited=kwargs.get('env') or os.environ
            child_env=sim_env.copy()
            for key in passthrough:
                if key in inherited:child_env[key]=inherited[key]
            child_env['ROS_DOMAIN_ID']=str(args.ros_domain_id)
            child_env['IGN_PARTITION']=args.ign_partition
            kwargs['env']=child_env
            if isinstance(command,list) and '--controller' in command:
                command += ['--condition',args.condition,'--layout',args.layout,'--seed',str(args.seed),
                    '--clearance-threshold',str(args.clearance_threshold),
                    '--trolley-clearance-threshold',str(args.trolley_clearance_threshold)]
                return SupervisedProcess(original(command,**kwargs))
            return original(command,**kwargs)
        demo.subprocess.Popen=spawn;os.environ['IGN_GAZEBO_RESOURCE_PATH']=str(PKG/'models/lab_assets')
        demo.orchestrate(args)
        if not (Path(args.output)/'episode/result.json').exists():raise RuntimeError('episode has no terminal result')
        return
    demo.rclpy.init();node=OfficeDeliveryPilot(args);wall=time.monotonic()
    try:
        while demo.rclpy.ok() and not node.done:
            demo.rclpy.spin_once(node,timeout_sec=.05)
            if time.monotonic()-wall>660:node.finish('WALL_TIMEOUT')
    finally:
        node.publish((0.,0.,0.));node.trace.close();node.frame_log.close();node.overview_log.close();node.events.close()
        node.pool.shutdown(wait=False);node.destroy_node();demo.rclpy.shutdown()

if __name__=='__main__':main()
