#!/usr/bin/env python3
"""Furnished-scene pilot: identical approach, then arrival/wait/active RGB-D check.

Deterministic execution policy, no VLM and no Nav2. Reuses the recorded demo's
sensor frontend, local planner and isolated process orchestration.
"""
import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'artifacts/group_meeting_20261013/scripts'))
import run_view_demo as demo
from rangermini_dynamic_semantic.observation_mppi import ObservationMPPI, ObservationConfig


class RevalidationPilot(demo.ViewDemo):
    def __init__(self, args):
        super().__init__(args)
        self.cfg = replace(self.cfg, random_seed=args.seed)
        self.arrival_time = None
        self.last_evidence_stamp = None
        self.support = []
        self.reference = None
        self.phase = 'APPROACH'
        self.events = (self.path / 'task_events.jsonl').open('w')
        (self.path/'confirmation_frames').mkdir()
        self.recheck = ObservationMPPI(self.cfg, ObservationConfig(
            observation_weight=0., arrival_radius=.16, lateral_candidates=True))
        manifest = json.loads((self.path / 'manifest.json').read_text())
        manifest.update(policy=args.policy, seed=args.seed, config=asdict(self.cfg),
            evidence_type='actual_gazebo_rgbd_closed_loop_capability_pilot',
            semantic_provider='deterministic_policy_no_api', nav2_used=False,
            target_identity='COCO tv class and proximity to initial RGB-D estimate; no instance identity',
            confirmation='3 distinct post-arrival RGB-D frames spanning >=0.4s, age<=0.35s, distance<=0.65m',
            recheck_budget_sim_s=8., memory_scope='one target fact, not full C1 task graph')
        (self.path / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    def event(self, kind, **extra):
        self.events.write(json.dumps(dict(sim_time=self.now(), kind=kind, **extra))+'\n')
        self.events.flush()

    def finish(self, status):
        if self.done: return
        self.event('TERMINAL', status=status, evidence_frames=len(self.support))
        super().finish(status)
        p=self.path/'result.json'
        result=json.loads(p.read_text())
        result.update(policy=self.args.policy, seed=self.args.seed,
            navigation_arrived=self.arrival_time is not None,
            target_confirmed=status=='TARGET_RECONFIRMED',
            post_arrival_support_frames=len(self.support),
            confirmation_stamps=self.support,
            arrival_elapsed_sim_s=self.arrival_time-self.started if self.arrival_time is not None else None,
            final_detection=self.detection, final_phase=self.phase,
            deepseek_api_called=False, nav2_used=False)
        p.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        (self.path/'task_memory.json').write_text(json.dumps(dict(
            fact_id='observed_monitor', version=2 if result['target_confirmed'] else 1,
            status='supported_by_current_rgbd' if result['target_confirmed'] else 'unconfirmed',
            initial_evidence=self.reference, post_arrival_support=self.support), indent=2))

    def tick(self):
        if self.done:return
        if self.pose is None or self.scan is None or self.rgb is None or self.depth is None:
            self.publish((0.,0.,0.)); return
        now=self.now()
        grounding=self.ground_target()
        if not self.ready:
            self.publish((0.,0.,0.))
            if grounding is None:return
            self.reference=grounding; self.target=grounding
            self.preference={'provider':'preset','preference':dict(demo.PREFERENCE),'label':'deterministic policy; no VLM'}
            self.started=now;self.ready=True
            self.planner=ObservationMPPI(self.cfg,ObservationConfig(observation_weight=6.,arrival_radius=.16,lateral_candidates=True))
            self.event('INITIAL_FACT_FROM_RGBD', grounding=grounding)
            (self.path/'initial_grounding.json').write_text(json.dumps(grounding,indent=2))
        if now-self.started>35: self.finish('APPROACH_TIMEOUT');return
        if now-self.last_plan<.12:return
        self.last_plan=now
        distance=math.hypot(self.goal[0]-self.pose[0],self.goal[1]-self.pose[1])
        if grounding is not None:self.target=grounding
        if self.arrival_time is None and distance<=.16:
            self.arrival_time=now;self.phase='POST_ARRIVAL_RECHECK'
            self.event('NAVIGATION_ARRIVAL', pose=self.pose, distance=distance)
            if self.args.policy=='arrival':self.finish('NAVIGATION_ONLY');return
        if self.arrival_time is not None:
            if grounding is not None and grounding['capture_time']>self.arrival_time and 0<=now-grounding['capture_time']<=.35:
                stamp=grounding['capture_time']
                delta=math.dist(grounding['point'],self.reference['point'])
                if stamp!=self.last_evidence_stamp and delta<=.65:
                    self.last_evidence_stamp=stamp;self.support.append(stamp)
                    stem=f'confirmation_frames/{len(self.support):03d}'
                    demo.PILImage.fromarray(self.rgb).save(self.path/(stem+'.jpg'),quality=95)
                    demo.np.save(self.path/(stem+'_depth.npy'),self.depth)
                    self.event('POST_ARRIVAL_RGBD_SUPPORT',grounding=grounding,position_delta_m=delta,
                               rgb_path=stem+'.jpg',depth_path=stem+'_depth.npy',depth_capture_time=self.depth_time)
            if len(self.support)>=3 and self.support[-1]-self.support[0]>=.4:
                self.finish('TARGET_RECONFIRMED');return
            if now-self.arrival_time>8:self.finish('TARGET_UNCONFIRMED');return
            if self.args.policy=='wait':
                self.publish((0.,0.,0.));plan={'action':[0.,0.,0.],'command_mode':'STOPPED','behavior':'WAIT_FOR_EVIDENCE'}
            else:
                target=self.reference['point']
                heading=math.atan2(target[1]-self.pose[1],target[0]-self.pose[0])
                plan=self.recheck.plan(self.pose,self.goal,(self.scan.ranges,self.scan.angle_min,self.scan.angle_increment),current_mode=self.mode,goal_yaw=heading)
                self.publish(plan['action'])
        else:
            plan=self.planner.plan(self.pose,self.goal,(self.scan.ranges,self.scan.angle_min,self.scan.angle_increment),current_mode=self.mode,goal_yaw=self.initial_yaw,hypotheses=(tuple(self.target['point'])+(1.,),))
            self.publish(plan['action'])
        self.trace.write(json.dumps(dict(timestamp=now,elapsed=now-self.started,pose=self.pose,pixels=int(self.mask.sum()),goal_distance=distance,target_grounding=self.target,detection=self.detection,phase=self.phase,plan=plan))+'\n')
        self.trace.flush()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--policy',choices=['arrival','wait','inspect'],default='inspect')
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--controller',action='store_true')
    p.add_argument('--world')
    p.add_argument('--method',default='view')
    p.add_argument('--scene',default='furnished')
    p.add_argument('--target-class',default='tv')
    p.add_argument('--semantic-provider',default='preset')
    args=p.parse_args()
    args.method='view';args.scene='furnished';args.target_class='tv';args.semantic_provider='preset'
    if not args.controller:
        # Keep the proven orchestrator, but ensure its child executes this pilot.
        demo.__file__=__file__
        original=demo.subprocess.Popen
        def spawn(command,**kwargs):
            if isinstance(command,list) and '--controller' in command:
                command += ['--policy',args.policy,'--seed',str(args.seed)]
            return original(command,**kwargs)
        demo.subprocess.Popen=spawn
        os.environ['IGN_GAZEBO_RESOURCE_PATH']=str(demo.PACKAGE/'models/lab_assets')
        demo.orchestrate(args)
        if not (Path(args.output)/'episode/result.json').exists():
            raise RuntimeError('controller produced no terminal result; inspect retained logs')
        return
    demo.rclpy.init();node=RevalidationPilot(args);wall=time.monotonic()
    try:
        while demo.rclpy.ok() and not node.done:
            demo.rclpy.spin_once(node,timeout_sec=.05)
            if time.monotonic()-wall>75:node.finish('SENSOR_OR_WALL_TIMEOUT')
    finally:
        node.publish((0.,0.,0.));node.events.close();node.trace.close();node.frame_log.close();node.overview_log.close()
        node.pool.shutdown(wait=False);node.destroy_node();demo.rclpy.shutdown()


if __name__=='__main__':main()
