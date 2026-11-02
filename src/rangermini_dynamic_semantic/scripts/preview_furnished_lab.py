#!/usr/bin/env python3
"""No-motion Gazebo visual preview and genuine YOLOv8n image inference."""
import argparse,json,os,subprocess,time,sys
from pathlib import Path
import cv2,numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from run_unified_gazebo_suite import stop

PKG=Path(__file__).resolve().parents[1]
CLASSES='person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee skis snowboard sports_ball kite baseball_bat baseball_glove skateboard surfboard tennis_racket bottle wine_glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch potted_plant bed dining_table toilet tv laptop mouse remote keyboard cell_phone microwave oven toaster sink refrigerator book clock vase scissors teddy_bear hair_drier toothbrush'.split()

def detect(path,weights,out):
    img=cv2.imread(str(path));h,w=img.shape[:2];scale=640/max(h,w)
    nw,nh=round(w*scale),round(h*scale);dx,dy=(640-nw)//2,(640-nh)//2
    padded=np.full((640,640,3),114,np.uint8);padded[dy:dy+nh,dx:dx+nw]=cv2.resize(img,(nw,nh))
    net=cv2.dnn.readNetFromONNX(str(weights));net.setInput(cv2.dnn.blobFromImage(padded,1/255.,(640,640),swapRB=True))
    start=time.perf_counter();pred=net.forward()[0].T;ms=(time.perf_counter()-start)*1000
    boxes=[];scores=[];labels=[]
    for row in pred:
        cls=int(np.argmax(row[4:]));score=float(row[4+cls])
        if score<.25:continue
        cx,cy,bw,bh=row[:4];x=(cx-bw/2-dx)/scale;y=(cy-bh/2-dy)/scale
        boxes.append([round(x),round(y),round(bw/scale),round(bh/scale)]);scores.append(score);labels.append(cls)
    kept=[]
    for cls in set(labels):
        indices=[i for i,c in enumerate(labels) if c==cls]
        nms=cv2.dnn.NMSBoxes([boxes[i] for i in indices],[scores[i] for i in indices],.25,.45)
        kept.extend(indices[int(i)] for i in np.asarray(nms).reshape(-1))
    records=[]
    for i in kept:
        x,y,bw,bh=boxes[i];x0,y0=max(0,x),max(0,y);x1,y1=min(w-1,x+bw),min(h-1,y+bh)
        records.append({'class':CLASSES[labels[i]],'confidence':scores[i],'bbox_xyxy':[x0,y0,x1,y1]})
        cv2.rectangle(img,(x0,y0),(x1,y1),(50,185,60),2)
        cv2.putText(img,f'{CLASSES[labels[i]]} {scores[i]:.2f}',(x0,max(18,y0-7)),cv2.FONT_HERSHEY_SIMPLEX,.55,(30,160,40),2)
    cv2.imwrite(str(out/'robot_detected.jpg'),img)
    (out/'detections.json').write_text(json.dumps({'source':str(path),'model':'YOLOv8n COCO pretrained','threshold':.25,'inference_ms':ms,'detections':records,'supplied_simulator_model_names':False},indent=2))
    return records

class Capture(Node):
    def __init__(self,out):
        super().__init__('furnished_lab_camera_preview')
        self.out=out;self.count={'robot':0,'overview':0}
        for topic,name in [('/camera/image','robot'),('/scene/overview','overview')]:
            self.create_subscription(Image,topic,lambda msg,n=name:self.receive(msg,n),10)
    def receive(self,msg,name):
        if msg.encoding.lower() not in ('rgb8','bgr8'):return
        a=np.frombuffer(bytes(msg.data),np.uint8).reshape(msg.height,msg.step)[:,:msg.width*3].reshape(msg.height,msg.width,3)
        if msg.encoding.lower()=='rgb8':a=a[:,:,::-1]
        self.count[name]+=1
        if self.count[name]<=60:
            cv2.imwrite(str(self.out/f'{name}_{self.count[name]:03d}.jpg'),a)
        cv2.imwrite(str(self.out/f'{name}_raw.jpg'),a)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--capture',action='store_true')
    p.add_argument('--world',default=str(PKG/'worlds/ranger_lab_corridor_furnished.sdf'));args=p.parse_args()
    out=Path(args.output)
    if args.capture:
        rclpy.init();node=Capture(out);start=time.monotonic()
        try:
            while time.monotonic()-start<40:
                rclpy.spin_once(node,timeout_sec=.1)
                if min(node.count.values())>=35:break
            (out/'capture.json').write_text(json.dumps(node.count))
        finally:node.destroy_node();rclpy.shutdown()
        if min(node.count.values())<1:raise RuntimeError('Missing actual camera image')
        return
    out.mkdir(parents=True,exist_ok=False)
    children=[];env=dict(os.environ,ROS_DOMAIN_ID='99',RMW_IMPLEMENTATION='rmw_fastrtps_cpp',IGN_PARTITION='ranger_furnished_preview',IGN_IP='127.0.0.1',ROS_LOG_DIR=str(out/'ros_logs'),IGN_GAZEBO_RESOURCE_PATH=str(PKG/'models/lab_assets'))
    env.pop('CYCLONEDDS_URI',None)
    def spawn(cmd,name):
        log=(out/f'{name}.log').open('w');proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);children.append((proc,log));return proc
    try:
        spawn(['ign','gazebo','-s','-r',args.world],'gazebo');time.sleep(2)
        spawn(['ros2','run','ros_gz_bridge','parameter_bridge','/camera/image@sensor_msgs/msg/Image[ignition.msgs.Image','/scene/overview@sensor_msgs/msg/Image[ignition.msgs.Image'],'bridge')
        cap=spawn(['python3',str(Path(__file__).resolve()),'--output',str(out),'--capture'],'capture');cap.wait(timeout=50)
        if cap.returncode:raise RuntimeError('Camera preview failed; see capture.log')
    finally:stop(children)
    records=detect(out/'robot_raw.jpg',PKG/'models/perception/yolov8n.onnx',out)
    print(json.dumps({'output':str(out),'detections':records},ensure_ascii=False))

if __name__=='__main__':main()
