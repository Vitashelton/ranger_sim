#!/usr/bin/env python3
"""Make the paper scene as a minimal, collision-consistent furnished-world copy."""
from pathlib import Path
import copy,json,hashlib,math
import xml.etree.ElementTree as E
ROOT=Path(__file__).resolve().parents[3];PKG=ROOT/'src/rangermini_dynamic_semantic'
source=PKG/'worlds/ranger_lab_corridor_furnished.sdf'
target=PKG/'worlds/ranger_lab_corridor_paper.sdf'
tree=E.parse(source);world=tree.getroot().find('world')
# Office C is the third near-side office (617); Office A is the first far office (619).
# Keep the fire-door hinges and corridor width; provide wall length for its open north leaf.
def set_box(model,pose,size):
 model.find('pose').text=' '.join(map(str,pose))
 for node in model.findall('.//geometry/box/size'):node.text=' '.join(map(str,size))
wall=world.find("model[@name='north_corridor_wall_3']")
set_box(wall,(10.445,-.3,1.2,0,0,0),(3.79,.18,2.4))  # x = 8.55 .. 12.34
wall=world.find("model[@name='north_corridor_wall_4']")
set_box(wall,(13.595,-.3,1.2,0,0,0),(.71,.18,2.4))  # x = 13.24 .. 13.95
for model in world.findall('model'):
 if model.get('name','').startswith('door_619_') or model.get('name')=='room_sign_619':
  pose=list(map(float,model.findtext('pose').split()));pose[0]+=1.00;model.find('pose').text=' '.join(map(str,pose))
# Close the room-side separation and the back-wall gap, with real collision geometry.
template=world.find("model[@name='far_office_divider_0']")
for name,pose,size in [
 ('office_C_A_solid_partition',(10.53,1.8,1.2,0,0,0),(.18,4.2,2.4)),
 ('office_C_A_backwall_connector',(10.305,3.9,1.2,0,0,0),(.45,.18,2.4))]:
 model=copy.deepcopy(template);model.set('name',name);set_box(model,pose,size);world.append(model)
# Strengthen the existing C boundary to the same 0.18m wall thickness.
set_box(world.find("model[@name='north_office_divider_3_0']"),(10.08,1.8,1.2,0,0,0),(.18,4.2,2.4))
tree.write(target,encoding='unicode',xml_declaration=True)
out=ROOT/'artifacts/paper_scene_20261014';out.mkdir(parents=True,exist_ok=True)
# Actual camera preview near C/A; this is separate from experiment initial pose.
preview=E.parse(target);pw=preview.getroot().find('world')
pw.find("model[@name='rangermini_2_0']/pose").text='9.0 -2.1 .01 0 0 0'
pw.find("model[@name='scene_overview_camera']/pose").text='10.5 0 12 0 1.57079632679 -1.57079632679'
preview.write(out/'preview_world.sdf',encoding='unicode',xml_declaration=True)
north_leaf_max_x=10.305+1.71;office_a_opening_min_x=11.79+1.00-.45
assert office_a_opening_min_x>north_leaf_max_x
manifest=dict(source=str(source),paper_world=str(target),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
 paper_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),room_mapping={'Office C':'617','Office A':'619'},
 changes=['A door/frame/sign moved +1.00m inside same room','corridor wall lengthened up to new A opening',
 'added full-height solid A west partition and closed C/A back-wall gap','C existing divider thickened to .18m'],
 fire_door_hinges_unchanged=True,corridor_width_unchanged=True,door_a_opening_x=[12.34,13.24],
 north_fire_door_open_leaf_max_x=north_leaf_max_x,door_leaf_to_opening_longitudinal_gap_m=office_a_opening_min_x-north_leaf_max_x,
 evidence='SDF geometry check; actual Gazebo preview follows')
(out/'scene_changes.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print(target)

