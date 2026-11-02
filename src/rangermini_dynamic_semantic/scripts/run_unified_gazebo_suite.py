#!/usr/bin/env python3
"""Generate task overlays without altering the established lab layout.

Default is one pilot episode. Suites expand only after the pilot succeeds.
Every child process is isolated in a process group and stopped on exit.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent


def model(name, xyz, size, color, collision=True):
    element = ET.Element('model',name=name)
    ET.SubElement(element,'static').text = 'true'
    ET.SubElement(element,'pose').text = ' '.join(map(str,(*xyz,0,0,0)))
    link = ET.SubElement(element,'link',name='link')
    for kind in (['collision','visual'] if collision else ['visual']):
        item = ET.SubElement(link,kind,name=kind)
        ET.SubElement(ET.SubElement(ET.SubElement(item,'geometry'),'box'),'size').text = ' '.join(map(str,size))
        if kind=='visual':
            material = ET.SubElement(item,'material')
            for key in ('ambient','diffuse'):
                ET.SubElement(material,key).text = ' '.join(map(str,(*color,1)))
    return element


def create_world(path, task, condition, seed):
    source = PACKAGE/'worlds/ranger_lab_corridor_firedoor_half_open.sdf'
    root = ET.parse(source)
    world = root.getroot().find('world')
    robot = world.find("model[@name='rangermini_2_0']")
    robot.find('pose').text = f'{.75+.05*seed} {-2.1+.04*seed} .01 0 0 {0.03*seed}'
    # Deliberately labeled horizontal 2-D baseline sensor, separate from real
    # MID-360S. Historical pitched scan and its worlds are left untouched.
    robot.find("link/sensor[@name='mid360s_proxy']/pose").text = '.30 0 .35 0 0 0'
    robot.find('link/collision/geometry/box/size').text = '.751 .500 .20'
    target_x = 3.24 if task=='lab_614' else 7.65
    world.append(model('service_marker',(target_x,-3.84,.65),(.30,.04,.25),(1,0,1),False))
    if condition in ('C1','F-occluded'):
        width = .65 if condition=='C1' else 3.0
        world.append(model('view_blocker',(target_x,-3.5,.65),(width,.18,1.0),(.2,.3,.35)))
    if condition=='C2':
        world.append(model('pedestrian_proxy',(target_x-.7,-3.45,.7),(.45,.35,1.4),(.2,.3,.35)))
        # Evaluator/script moves this entity. Its poses do not enter the planner.
        plugin = ET.SubElement(world,'plugin',filename='libignition-gazebo-triggered-publisher-system.so',
                               name='ignition::gazebo::systems::TriggeredPublisher')
        # Movement is applied by the independent ROS event driver below.
        world.remove(plugin)
    if condition=='C3':
        world.append(model('doorway_cart',(target_x-.6,-3.05,.4),(.55,.55,.8),(.3,.3,.3)))
    root.write(path,encoding='unicode',xml_declaration=True)
    return {'task':task,'condition':condition,'seed':seed,
            'base_world_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'horizontal_scan_proxy':True,'target_code':'magenta',
            'target_asset_pose_evaluator_only':[target_x,-3.84,.65]}


def stop(children):
    for process, log in reversed(children):
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGINT)
    deadline = time.monotonic()+5
    for process, log in reversed(children):
        try:
            process.wait(timeout=max(.1,deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait(timeout=2)
        log.close()


def run_episode(root, task, condition, method, seed, args):
    name = f'{task}_{condition}_{method}_s{seed}_n{args.samples}_wo{args.weight}_c{args.c0}_{args.c1}'
    work = root/name
    work.mkdir(parents=True,exist_ok=False)
    world = work/'world.sdf'
    meta = create_world(world,task,condition,seed)
    (work/'scene_manifest.json').write_text(json.dumps(meta,indent=2))
    children = []
    env = dict(os.environ,ROS_DOMAIN_ID='99',RMW_IMPLEMENTATION='rmw_fastrtps_cpp',
               IGN_PARTITION='unified_ranger_lab',IGN_IP='127.0.0.1',
               ROS_LOG_DIR=str(work/'ros_logs'))
    env.pop('CYCLONEDDS_URI',None)
    def spawn(command,label):
        log = (work/(label+'.log')).open('w')
        process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                                   env=env,start_new_session=True)
        children.append((process,log))
        return process
    print(f'START {name}',flush=True)
    try:
        spawn(['ign','gazebo','-s','-r',str(world)],'gazebo')
        time.sleep(2)
        spawn(['ros2','run','ros_gz_bridge','parameter_bridge',
            '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
            '/model/rangermini_2_0/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist',
            '/model/rangermini_2_0/pose@geometry_msgs/msg/PoseStamped[ignition.msgs.Pose',
            '/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan',
            '/camera/image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/world/ranger_lab_corridor/set_pose@ros_gz_interfaces/srv/SetEntityPose',
            '--ros-args','-r','/model/rangermini_2_0/cmd_vel:=/cmd_vel_gazebo'],'bridge')
        spawn(['ros2','run','rangermini_dynamic_semantic','gazebo_odom_adapter'],'odom')
        spawn(['ros2','run','rangermini_dynamic_semantic','ranger_motion_adapter',
               '--ros-args','-p','input_topic:=/cmd_vel_safe','-p','output_topic:=/cmd_vel_gazebo'],'adapter')
        if condition=='C2':
            spawn(['python3',str(HERE/'unified_pedestrian_event.py'),
                   '--x',str(3.24 if task=='lab_614' else 7.65)],'event')
        process = spawn(['python3',str(HERE/'run_unified_gazebo_episode.py'),
            '--output',str(work/'episode'),'--world',str(world),'--task',task,
            '--method',method,'--seed',str(seed),'--samples',str(args.samples),
            '--weight',str(args.weight),'--c0',str(args.c0),'--c1',str(args.c1),
            '--deadline',str(args.deadline)],'controller')
        process.wait(timeout=140)
        result_file = work/'episode/result.json'
        result = json.loads(result_file.read_text()) if result_file.exists() else {
            'status':'INFRASTRUCTURE_FAILURE','reason':'no_terminal_result'}
        result['directory'] = str(work)
        print('END '+json.dumps(result),flush=True)
        return result
    finally:
        stop(children)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--tasks',nargs='+',default=['lab_614'])
    p.add_argument('--conditions',nargs='+',default=['C0'])
    p.add_argument('--methods',nargs='+',default=['O-MPPI'])
    p.add_argument('--seeds',nargs='+',type=int,default=[0])
    p.add_argument('--samples',type=int,default=16)
    p.add_argument('--weight',type=float,default=3)
    p.add_argument('--c0',type=float,default=.28)
    p.add_argument('--c1',type=float,default=.28)
    p.add_argument('--deadline',type=float,default=45)
    args = p.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True,exist_ok=False)
    (root/'suite_manifest.json').write_text(json.dumps(vars(args),indent=2))
    results = []
    for task in args.tasks:
        for condition in args.conditions:
            for seed in args.seeds:
                for method in args.methods:
                    result = run_episode(root,task,condition,method,seed,args)
                    results.append(result)
                    (root/'results.json').write_text(json.dumps(results,indent=2))
                    # Infrastructure failures are not research failures: fix
                    # the pipeline before consuming the declared main seeds.
                    if result['status']=='INFRASTRUCTURE_FAILURE':
                        return 2
    return 0


if __name__=='__main__':
    raise SystemExit(main())
