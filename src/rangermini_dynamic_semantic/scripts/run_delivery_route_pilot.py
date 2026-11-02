#!/usr/bin/env python3
"""Real furnished-Gazebo pilot: identical route bank, three route selectors.

One semantic selection at departure. Low-level MPPI continually uses current
laser geometry. This is a delivery-route experiment, not completed handover.
"""
import argparse
import copy
from dataclasses import asdict,replace
import io
import json
import math
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

PKG=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PKG));sys.path.insert(0,str(PKG.parents[1]/'artifacts/group_meeting_20261013/scripts'))
import run_view_demo as demo
from rangermini_dynamic_semantic.observation_mppi import ObservationMPPI,ObservationConfig
from rangermini_dynamic_semantic.visual_behavior_selector import select_visual_route,locate_visual_task_target

INSTRUCTION='把车载资料送到前面的实验室。经过箱子时，避开站在旁边的同学，别从他身边挤过去。'
ROUTES={'LEFT':[(4.80,-1.45),(6.45,-1.45),(7.25,-2.10)],
        'RIGHT':[(4.80,-2.90),(6.45,-2.90),(7.25,-2.10)]}
TASK_INSTRUCTIONS={
    'person':INSTRUCTION,
    'trolley':'把车载资料送到前面的实验室。经过箱子时，尽量远离器材推车一侧，给取用和搬运器材留出空间。',
}
COLORS={'LEFT':(72,155,116),'RIGHT':(214,117,52)}


def create_world(path,scene,person_side='north',paired_objects=False,person_range='near'):
    root=ET.parse(PKG/'worlds/ranger_lab_corridor_furnished.sdf');world=root.getroot().find('world')
    robot=world.find("model[@name='rangermini_2_0']")
    robot.find('pose').text='3.0 -2.1 .01 0 0 0'
    robot.find("link/sensor[@name='mid360s_proxy']/pose").text='.30 0 .35 0 0 0'
    robot.find('link/collision/geometry/box/size').text='.751 .500 .20'
    world.find("model[@name='corridor_box']/pose").text='5.4 -2.1 .02 0 0 0'
    person=copy.deepcopy(world.find("model[@name='lab_student']"));person.set('name','waiting_student')
    person.find('pose').text=('4.9 -0.95 .02 0 0 3.141592653589793' if person_side=='north' else '4.9 -3.30 .02 0 0 0')
    if person_range=='remote':
        pose=person.find('pose').text.split();pose[0]='8.8';person.find('pose').text=' '.join(pose)
    link=person.find('link')
    collision=ET.SubElement(link,'collision',name='body_proxy')
    ET.SubElement(collision,'pose').text='0 0 .85 0 0 0'
    cylinder=ET.SubElement(ET.SubElement(collision,'geometry'),'cylinder')
    ET.SubElement(cylinder,'radius').text='.18';ET.SubElement(cylinder,'length').text='1.7'
    world.append(person)
    if paired_objects:
        # Relocate the existing trolley assembly; preserve original world.
        target_y=-3.30 if person_side=='north' else -.95
        for model in world.findall('model'):
            if model.get('name')=='service_trolley' or model.get('name','').startswith('trolley_'):
                pose=list(map(float,model.findtext('pose').split()))
                pose[0]+=4.9-8.9;pose[1]+=target_y-(-1.1)
                model.find('pose').text=' '.join(map(str,pose))
    camera=world.find("model[@name='navigation_overhead_camera']")
    camera.find('pose').text='5.2 -2.1 8 0 1.57079632679 -1.57079632679'
    root.write(path,encoding='unicode',xml_declaration=True)


class DeliveryPilot(demo.ViewDemo):
    def __init__(self,args):
        super().__init__(args)
        self.instruction=TASK_INSTRUCTIONS[args.task_target]
        if args.person_range=='remote':self.instruction='把车载资料送到前面的实验室。沿途给等候同学留出空间，别从他身边挤过去。'
        self.detector=None;self.mask=None
        self.initial_yaw=0.;self.goal=ROUTES['LEFT'][-1]
        self.cfg=replace(self.cfg,random_seed=args.seed,max_speed=.40,horizon=36,samples_per_behavior=16)
        self.bank={k:ObservationMPPI(self.cfg,ObservationConfig(observation_weight=0.,arrival_radius=.18,lateral_candidates=True)) for k in ROUTES}
        self.route_id=None;self.stage=0;self.route_planner=None;self.initial_plans=None
        self.events=(self.path/'selection_events.jsonl').open('w')
        self.person_detector=demo.cv2.dnn.readNetFromONNX(str(PKG/'models/perception/yolov8n.onnx'))
        m=json.loads((self.path/'manifest.json').read_text())
        m.update(instruction=self.instruction,task_target=args.task_target,paired_objects=args.paired_objects,person_range=args.person_range,method=args.selector,config=asdict(self.cfg),goal=self.goal,
            person_side=args.person_side,
            target_geometry_source='registered_route_guides; fixed_rule_only_uses_detected_person_RGBD',
            semantic_provider_requested='deepseek' if args.selector in ('vlm','grounded','image_rule') else args.selector,
            evidence_type='actual_gazebo_delivery_route_pilot',route_bank=ROUTES,
            navigation_used='existing_ObservationMPPI_with_RangerMini_4WS_kinematic_adapter',
            robot_motion_model='four-wheel-steering rigid-body pure-rolling constraints; RangerMini wheelbase/track; finite steering rate; stop during mode changes',
            person_clearance_requirement_m=.60,
            semantic_calls='one asynchronous selection at departure; waits stopped before departure',
            full_delivery_or_handover_measured=False,semantic_entity_truth_used_by_controller=False,
            localization_source='Gazebo model pose through odom adapter; not a localization estimator')
        (self.path/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2))

    def on_rgb(self,msg):
        if msg.encoding.lower() not in ('rgb8','bgr8'):return
        a=demo.np.frombuffer(bytes(msg.data),dtype=demo.np.uint8).reshape(msg.height,msg.step)
        self.rgb=a[:,:msg.width*3].reshape(msg.height,msg.width,3).copy()
        if msg.encoding.lower()=='bgr8':self.rgb=self.rgb[:,:,::-1].copy()
        self.frame_time=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        self.mask=demo.np.zeros(self.rgb.shape[:2],bool)
        if self.frame_time-self.last_frame>=.12:
            self.last_frame=self.frame_time
            demo.PILImage.fromarray(self.rgb).save(self.path/'frames'/f'{self.frame_index:05d}.jpg',quality=90)
            self.frame_log.write(json.dumps(dict(frame=self.frame_index,sim_time=self.frame_time,pose=self.pose))+'\n');self.frame_log.flush();self.frame_index+=1

    def project(self,trajectory):
        x,y,psi=self.pose;f=640/(2*math.tan(1.396/2));points=[]
        for px,py,*_ in trajectory:
            dx,dy=px-x,py-y;forward=math.cos(psi)*dx+math.sin(psi)*dy-.34
            lateral=-math.sin(psi)*dx+math.cos(psi)*dy
            if forward<=.05:continue
            u=320-lateral*f/forward;v=240+(.43-.04)*f/forward
            if 0<=u<640 and 0<=v<480:points.append((int(u),int(v)))
        return points

    def fixed_rule(self,candidates):
        # Stronger than a keyword-only baseline: detect person, depth-project,
        # then select the candidate route with greater person clearance.
        padded=demo.np.full((640,640,3),114,demo.np.uint8);padded[80:560]=self.rgb
        self.person_detector.setInput(demo.cv2.dnn.blobFromImage(padded,1/255.,(640,640),swapRB=False))
        prediction=self.person_detector.forward()[0].T
        persons=[r for r in prediction if int(demo.np.argmax(r[4:]))==0 and float(r[4])>=.25]
        grounded=[]
        for r in persons:
            cx,cy,w,h=map(float,r[:4]);cy-=80
            x0,x1=max(0,int(cx-w*.3)),min(640,int(cx+w*.3));y0,y1=max(0,int(cy-h*.3)),min(480,int(cy+h*.3))
            region=self.depth[y0:y1,x0:x1];z=region[demo.np.isfinite(region)&(region>.2)&(region<8)]
            if len(z)<20:continue
            d=float(demo.np.median(z));f=640/(2*math.tan(1.396/2))
            bx,by=.34+d,-(cx-320)*d/f;x,y,psi=self.pose
            grounded.append((x+math.cos(psi)*bx-math.sin(psi)*by,y+math.sin(psi)*bx+math.cos(psi)*by))
        if grounded:
            clear={k:min(math.dist(p,q) for p in grounded for q in ROUTES[k][:2]) for k in candidates}
            chosen=max(clear,key=clear.get)
        else:clear={};chosen=min(candidates,key=lambda k:candidates[k]['predicted_score'])
        return dict(provider='fixed_rule_yolo_depth',decision={'candidate_id':chosen,'reason':'maximize person clearance; geometry fallback if person ungrounded'},persons_rgbd=grounded,route_person_clearance=clear)

    def ground_target_preference(self,response):
        box=response['evidence']['bbox']
        if box is None:raise ValueError('task target not visually grounded')
        x0,y0,x1,y1=box;x0,x1=int(x0*640),min(640,int(x1*640));y0,y1=int(y0*480),min(480,int(y1*480))
        region=self.depth[y0:y1,x0:x1];yy,xx=demo.np.where(demo.np.isfinite(region)&(region>.2)&(region<8))
        if len(xx)<20:raise ValueError('insufficient valid target depth')
        demo.np.save(self.path/'grounding_depth_roi.npy',region)
        depths=region[yy,xx];near=depths<=demo.np.quantile(depths,.20)
        depths=depths[near];xx=xx[near]+x0
        f=640/(2*math.tan(1.396/2));bx=.34+depths;by=-(xx-320)*depths/f
        x,y,psi=self.pose;px=float(demo.np.median(x+math.cos(psi)*bx-math.sin(psi)*by));py=float(demo.np.median(y+math.sin(psi)*bx+math.cos(psi)*by))
        # Dense registered guide, fixed-heading rectangular chassis proxy;
        # all prefixes must also pass the shared current-scan MPPI check.
        radius=.18 if self.args.task_target=='person' else .41
        scores={}
        for route,plan in self.initial_plans.items():
            if not plan['selected_trajectory'] or plan['selected_behavior']=='STOP':continue
            points=[self.pose[:2]]+ROUTES[route];gaps=[]
            for a,b in zip(points,points[1:]):
                for t in demo.np.linspace(0,1,max(2,int(math.dist(a,b)/.05)+1)):
                    qx=a[0]+t*(b[0]-a[0]);qy=a[1]+t*(b[1]-a[1])
                    gaps.append(max(0.,math.hypot(max(abs(px-qx)-.751/2,0),max(abs(py-qy)-.500/2,0))-radius))
            scores[route]=min(gaps)
        if not scores:raise ValueError('no currently feasible grounded route')
        # Resolve exact target-score ties with the already shared MPPI objective.
        best=max(scores.values());tied=[k for k,v in scores.items() if best-v<=1e-9]
        route=min(tied,key=lambda k:self.initial_plans[k]['predicted_score'])
        response.update(grounded_target_xy=[px,py],depth_grounding='nearest-depth quintile; median unprojected XY; no simulator object pose',
            route_target_proxy_gaps=scores,decision=dict(candidate_id=route,reason='maximize task-target proxy gap; objective breaks exact score ties'))
        demo.PILImage.fromarray(self.rgb).save(self.path/'grounding_rgb.jpg',quality=95)
        return response

    def finish(self,status):
        if self.done:return
        super().finish(status)
        p=self.path/'result.json';r=json.loads(p.read_text())
        r.update(method=self.args.selector,seed=self.args.seed,route_id=self.route_id,
                 initial_plans=self.initial_plans,semantic=self.preference,
                 api_called=self.args.selector in ('vlm','grounded','image_rule') and self.future is not None,
                 navigation_arrived=status=='ARRIVED',full_delivery_success=False)
        p.write_text(json.dumps(r,ensure_ascii=False,indent=2))

    def tick(self):
        if self.done:return
        now=self.now()
        if self.pose is None or self.scan is None or self.rgb is None or self.depth is None:
            self.publish((0.,0.,0.));return
        if not self.ready:
            self.publish((0.,0.,0.))
            if self.initial_plans is None:
                scan=(self.scan.ranges,self.scan.angle_min,self.scan.angle_increment)
                plans={k:planner.plan(self.pose,ROUTES[k][0],scan,current_mode=self.mode,goal_yaw=0.,current_steering_angles=self.mode_payload.get('wheel_steering_rad'),current_steering_transition=self.mode_payload.get('steering_transition_stop',False)) for k,planner in self.bank.items()}
                candidates={k:v for k,v in plans.items() if v['selected_trajectory'] and v['selected_behavior']!='STOP'}
                self.initial_plans=plans
                if not candidates:self.finish('NO_FEASIBLE_CANDIDATE');return
                overlay=self.rgb.copy()
                for k,plan in candidates.items():
                    pixels=self.project(plan['selected_trajectory'])
                    plan['projection_pixels']=pixels
                    if len(pixels)>=2:demo.cv2.polylines(overlay,[demo.np.asarray(pixels,demo.np.int32)],False,COLORS[k],3)
                demo.PILImage.fromarray(self.rgb).save(self.path/'selection_rgb.jpg',quality=95)
                demo.PILImage.fromarray(overlay).save(self.path/'selection_overlay.jpg',quality=95)
                (self.path/'selection_input.json').write_text(json.dumps(dict(instruction=self.instruction,capture_time=self.frame_time,capture_pose=self.pose,candidates=plans),ensure_ascii=False,indent=2))
                if self.args.selector in ('vlm','grounded','image_rule'):
                    if any(len(v['projection_pixels'])<2 for v in candidates.values()):
                        self.finish('CANDIDATE_PROJECTION_UNVISIBLE');return
                    b=io.BytesIO();demo.PILImage.fromarray(overlay).save(b,format='JPEG',quality=95)
                    description=[dict(candidate_id=k,feasible=True,color='green' if k=='LEFT' else 'orange') for k,v in candidates.items()]
                    if self.args.selector in ('grounded','image_rule'):
                        b=io.BytesIO();demo.PILImage.fromarray(self.rgb).save(b,format='JPEG',quality=95)
                        self.future=self.pool.submit(locate_visual_task_target,b.getvalue(),self.instruction,self.args.task_target)
                    else:self.future=self.pool.submit(select_visual_route,b.getvalue(),self.instruction,description)
                elif self.args.selector=='fixed':self.preference=self.fixed_rule(candidates)
                else:
                    chosen=min(candidates,key=lambda k:candidates[k]['predicted_score'])
                    self.preference=dict(provider='geometric_mppi',decision=dict(candidate_id=chosen,reason='minimum MPPI objective'))
            if self.args.selector in ('vlm','grounded','image_rule'):
                if not self.future.done():return
                try:
                    self.preference=self.future.result()
                    if self.args.selector=='grounded':self.preference=self.ground_target_preference(self.preference)
                    elif self.args.selector=='image_rule':
                        box=self.preference['evidence']['bbox']
                        if box is None:raise ValueError('task target not visible')
                        chosen='RIGHT' if (box[0]+box[2])/2<.5 else 'LEFT'
                        if self.initial_plans[chosen]['selected_behavior']=='STOP':raise ValueError('image-rule route has no feasible prefix')
                        self.preference['decision']=dict(candidate_id=chosen,reason='task-aware image-side rule using same object locator; no depth or metric distance')
                except Exception as exc:
                    self.preference=dict(provider='deepseek',error=str(exc));self.finish('VLM_FAILED');return
            self.route_id=self.preference['decision']['candidate_id'];self.route_planner=self.bank[self.route_id]
            (self.path/'selection_output.json').write_text(json.dumps(self.preference,ensure_ascii=False,indent=2))
            self.events.write(json.dumps(dict(time=now,route_id=self.route_id,selection=self.preference),ensure_ascii=False)+'\n');self.events.flush()
            self.started=now;self.ready=True
        if now-self.started>45:self.finish('TIMEOUT');return
        if now-self.last_plan<.12:return
        self.last_plan=now
        waypoint=ROUTES[self.route_id][self.stage];distance=math.dist(self.pose[:2],waypoint)
        if distance<=.20:
            if self.stage==2:self.finish('ARRIVED');return
            self.stage+=1;waypoint=ROUTES[self.route_id][self.stage]
        plan=self.route_planner.plan(self.pose,waypoint,(self.scan.ranges,self.scan.angle_min,self.scan.angle_increment),current_mode=self.mode,goal_yaw=0.,current_steering_angles=self.mode_payload.get('wheel_steering_rad'),current_steering_transition=self.mode_payload.get('steering_transition_stop',False))
        self.publish(plan['action'])
        self.trace.write(json.dumps(dict(timestamp=now,elapsed=now-self.started,pose=self.pose,pixels=0,goal_distance=math.dist(self.pose[:2],self.goal),waypoint=waypoint,stage=self.stage,route_id=self.route_id,actuation=self.mode_payload,plan=plan))+'\n');self.trace.flush()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--selector',choices=['geometric','fixed','vlm','grounded','image_rule'],default='geometric');p.add_argument('--seed',type=int,default=17)
    p.add_argument('--task-target',choices=list(TASK_INSTRUCTIONS),default='person')
    p.add_argument('--person-range',choices=['near','remote'],default='near')
    p.add_argument('--paired-objects',action='store_true')
    p.add_argument('--person-side',choices=['north','south'],default='north')
    p.add_argument('--controller',action='store_true');p.add_argument('--world')
    for name,default in [('method','view'),('scene','furnished'),('target-class','person'),('semantic-provider','preset')]:p.add_argument('--'+name,default=default)
    args=p.parse_args()
    if not args.controller:
        demo.__file__=__file__;demo.create_world=lambda path,scene:create_world(path,scene,args.person_side,args.paired_objects,args.person_range)
        original=demo.subprocess.Popen
        def spawn(command,**kwargs):
            if isinstance(command,list) and '--controller' in command:command += ['--selector',args.selector,'--seed',str(args.seed),'--person-side',args.person_side,'--task-target',args.task_target,'--person-range',args.person_range]
            if isinstance(command,list) and '--controller' in command and args.paired_objects:command.append('--paired-objects')
            return original(command,**kwargs)
        demo.subprocess.Popen=spawn
        os.environ['IGN_GAZEBO_RESOURCE_PATH']=str(PKG/'models/lab_assets')
        demo.orchestrate(args)
        if not (Path(args.output)/'episode/result.json').exists():raise RuntimeError('no result; inspect retained controller log')
        return
    demo.rclpy.init();node=DeliveryPilot(args);start=time.monotonic()
    try:
        while demo.rclpy.ok() and not node.done:
            demo.rclpy.spin_once(node,timeout_sec=.05)
            if time.monotonic()-start>75:node.finish('WALL_OR_SENSOR_TIMEOUT')
    finally:
        node.publish((0.,0.,0.));node.trace.close();node.events.close();node.frame_log.close();node.overview_log.close()
        node.pool.shutdown(wait=False);node.destroy_node();demo.rclpy.shutdown()


if __name__=='__main__':main()
