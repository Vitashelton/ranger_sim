#!/usr/bin/env python3
"""Room-depth extension of the established furnished laboratory floor."""
from pathlib import Path
import copy,json,math,xml.etree.ElementTree as E
from run_unified_gazebo_suite import model
ROOT=Path(__file__).resolve().parents[1]

def main():
    tree=E.parse(ROOT/'worlds/ranger_lab_corridor_furnished.sdf');w=tree.getroot().find('world')
    # Top view: north leaf lies along the corridor; south leaf is half open.
    # Leaf geometry is locally along Y, so horizontal requires a 90-degree yaw.
    for name,angle in [('fire_door_north_leaf',90),('fire_door_south_leaf',-45)]:
        p=w.find(f"model[@name='{name}']/pose");a=p.text.split()
        a[5]=str(math.radians(angle));p.text=' '.join(a)
    def pose(name,xyz,yaw=None):
        p=w.find(f"model[@name='{name}']/pose");a=p.text.split();a[:3]=map(str,xyz)
        if yaw is not None:a[5]=str(yaw)
        p.text=' '.join(a)
    def size(name,xyz):
        m=w.find(f"model[@name='{name}']")
        for s in m.findall('.//geometry/box/size'):s.text=' '.join(map(str,xyz))
    pose('south_lab_outer_wall_0',(5.04,-11.4,1.2))
    for name,x in [('south_lab_divider_1_0',5.49),('south_lab_divider_2_0',10.08)]:
        pose(name,(x,-7.65,1.2));size(name,(.108,7.5,2.4))
    w.append(model('south_lab_west_wall',(0,-7.65,1.2),(.108,7.5,2.4),(.72,.74,.76)))
    pose('south_rooms_floor',(7.92,-7.65,.005));size('south_rooms_floor',(15.84,7.5,.02))
    pose('building_floor',(9.18,-3.2625,-.02));size('building_floor',(19.26,16.275,.04))
    for num,x,front in [('614',3.24,1.6),('616',7.65,6.65)]:
        pose('lab_table_'+num,(front,-6.2,.02))
        pose('lab_monitor_'+num,(front,-6.2,.82))
        pose('lab_chair_'+num,(front-.25,-5.2,.02),math.pi)
        cx=1.1 if num=='614' else 9.0
        pose('lab_console_'+num,(cx,-10.1,0))
        for old,new,xyz in [('lab_table_','lab_rear_table_',(x+.7,-9.2,.02)),
                            ('lab_monitor_','lab_rear_monitor_',(x+.7,-9.2,.82)),
                            ('lab_chair_','lab_rear_chair_',(x+1.0,-8.0,.02))]:
            m=copy.deepcopy(w.find(f"model[@name='{old+num}']"));m.set('name',new+num)
            a=m.find('pose').text.split();a[:3]=map(str,xyz);m.find('pose').text=' '.join(a);w.append(m)
        # The room entrance does not directly reveal the target person.
        w.append(model('lab_view_partition_'+num,(front,-7.6,.75),(1.9,.08,1.5),(.62,.66,.69)))
    pose('lab_student',(1.75,-8.9,.02),0)
    other=copy.deepcopy(w.find("model[@name='lab_student']"));other.set('name','lab_student_616')
    other.find('pose').text='6.4 -9.3 .02 0 0 0';w.append(other)
    pose('scene_overview_camera',(7,-14,17),math.pi/2)
    camera=w.find("model[@name='scene_overview_camera']/pose");a=camera.text.split();a[4]='1.02';camera.text=' '.join(a)
    pose('navigation_overhead_camera',(3.0,-6.3,9),-math.pi/2)
    w.find("model[@name='navigation_overhead_camera']/link/sensor/camera/horizontal_fov").text='1.2'
    pose('rangermini_2_0',(3.24,-2.0,.01),-math.pi/2)
    out=ROOT/'worlds/ranger_lab_corridor_room_search.sdf';tree.write(out,encoding='unicode',xml_declaration=True)
    registry={'map_id':'ranger_lab_room_search_v1','rooms':{
        '614':{'polygon':[[.054,-3.99],[5.436,-3.99],[5.436,-11.31],[.054,-11.31]],'door':[3.24,-3.9],
               'viewpoints':[[3.24,-4.8],[3.24,-6.3],[3.1,-8.1]]},
        '616':{'polygon':[[5.544,-3.99],[10.026,-3.99],[10.026,-11.31],[5.544,-11.31]],'door':[7.65,-3.9],
               'viewpoints':[[7.65,-4.8],[7.65,-6.3],[7.7,-8.6]]}},
        'scope':'Known room regions and candidate viewpoints; no target object or person coordinates are supplied to the controller.'}
    (ROOT/'config/room_search_registry.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2))
    print(out)
if __name__=='__main__':main()
