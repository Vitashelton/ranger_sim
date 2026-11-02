#!/usr/bin/env python3
"""Generate the fixed Office A delivery layouts used in the 2x2 study."""
import xml.etree.ElementTree as ET
from pathlib import Path

PKG=Path(__file__).resolve().parents[1]
LAYOUTS={
    # Each layout places the waiting student, cart and parcel box at different
    # longitudinal positions, yielding a human-scale S-shaped passage.
    "east_weave": {"person":(5.7,-.96),"cart":(8.65,-3.28)},
    "west_weave": {"person":(8.55,-3.22),"cart":(5.75,-.96)},
    "door_approach": {"person":(10.10,-3.22),"cart":(7.35,-.96)},
    "person_clearance": {"person":(6.50,-2.10),"cart":(8.65,-1.05)},
    "person_clearance_mirror": {"person":(6.50,-3.15),"cart":(8.65,-1.05)},
    "doorway_delivery": {"person":(10.10,-2.10),"cart":(7.35,-1.05)},
}

def make_world(layout,output):
    if layout not in LAYOUTS: raise ValueError(f"unknown layout: {layout}")
    root=ET.parse(PKG/'worlds/ranger_lab_corridor_paper.sdf'); world=root.getroot().find('world')
    robot=world.find("model[@name='rangermini_2_0']")
    robot.find('pose').text='3.0 -2.10 .01 0 0 0'
    # A visible, non-colliding document packet makes the service payload
    # legible in Gazebo frames. It is not a handoff actuator or task-completion
    # sensor; delivery is still scored at the marked receiving pose.
    base=robot.find("link[@name='base_link']")
    if base is not None and base.find("visual[@name='delivery_packet']") is None:
        packet=ET.SubElement(base,'visual',name='delivery_packet')
        ET.SubElement(packet,'pose').text='-0.015 0 0.1325 0 0 0'
        geom=ET.SubElement(packet,'geometry');box=ET.SubElement(geom,'box')
        ET.SubElement(box,'size').text='.30 .22 .065'
        material=ET.SubElement(packet,'material')
        ET.SubElement(material,'ambient').text='.56 .34 .14 1'
        ET.SubElement(material,'diffuse').text='.72 .49 .25 1'
        sheet=ET.SubElement(base,'visual',name='document_sheets')
        ET.SubElement(sheet,'pose').text='-0.015 0 0.1675 0 0 0'
        sg=ET.SubElement(sheet,'geometry');sb=ET.SubElement(sg,'box')
        ET.SubElement(sb,'size').text='.26 .18 .004'
        sm=ET.SubElement(sheet,'material')
        ET.SubElement(sm,'ambient').text='.92 .92 .86 1'
        ET.SubElement(sm,'diffuse').text='.98 .98 .92 1'
    target=LAYOUTS[layout]
    # The source furnished world includes a box near the S-shaped route. It is
    # removed in this controlled study so the human and trolley remain the two
    # task-relevant dynamic-clearance proxies rather than an accidental choke.
    box=world.find("model[@name='corridor_box']")
    if box is not None:world.remove(box)
    person=world.find("model[@name='lab_student']")
    person.find('pose').text=f"{target['person'][0]} {target['person'][1]} .02 0 0 0"
    person.find('static').text='true'
    link=person.find('link')
    if link.find("collision[@name='paper_person_proxy']") is None:
        collision=ET.SubElement(link,'collision',name='paper_person_proxy')
        ET.SubElement(collision,'pose').text='0 0 .85 0 0 0'
        cylinder=ET.SubElement(ET.SubElement(collision,'geometry'),'cylinder')
        ET.SubElement(cylinder,'radius').text='.18'; ET.SubElement(cylinder,'length').text='1.7'
    dx=target['cart'][0]-8.9; dy=target['cart'][1]-(-1.1)
    for model in world.findall('model'):
        if model.get('name')=='service_trolley' or model.get('name','').startswith('trolley_'):
            pose=list(map(float,model.findtext('pose').split()));pose[0]+=dx;pose[1]+=dy
            model.find('pose').text=' '.join(map(str,pose))
    root.write(output,encoding='unicode',xml_declaration=True)
    return dict(layout=layout,obstacles=target,goal=(12.80,1.15,1.57079632679),
                service_target={'object':'indoor document handoff zone','room':'Office A / room 619',
                    'approach_pose':(12.80,1.15,1.57079632679)},
                instruction='把资料送到 Office A（619 室）门内交接区，沿走廊通行并从敞开的门洞进入，在门内交接位停稳。')

