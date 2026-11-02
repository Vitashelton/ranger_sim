#!/usr/bin/env python3
"""Mechanism-focused scenes reusing the furnished lab and proximity fixture.

Actors respond only to measured robot proximity, never to the method name.
Scene generation does not constitute a completed navigation experiment.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from grounded_delivery_scene import create_world as create_base_world
import continuous_delivery_scene as continuous

SCENARIOS = ('heading_constrained_passage', 'static_detour', 'multiroom_service', 'collection_delivery', 'collection_return_delivery', 'heading_passing', 'offset_portal', 'encounter_entry', 'delivery_aisle_open', 'delivery_aisle_narrow', 'office_handoff')
SPECS = {
    'static_detour': dict(start=(7., -2.1, 0.), cart=(2., -.95),
        person=(2.85, -6.05), cart_trigger=.1, person_trigger=.1,
        parked=(9., -2.1), purpose='Static equipment detour without a heading-retention requirement; measure path, time and unnecessary mode changes'),
    'collection_delivery': dict(start=(3., -2.1, 0.), cart=(8.6, -2.1),
        person=(4.4, -3.4), cart_trigger=3.5, person_trigger=.1,
        parked=None, purpose='Registered collection stop, forward departure reorientation, corridor encounter, and delivery to Office 619'),
    'office_handoff': dict(start=(12.8, -1.15, math.pi/2), cart=(3., -3.1),
        person=(2.85, -6.05), cart_trigger=.1, person_trigger=.1,
        parked=None, purpose='Existing narrow doorway and desk-referenced, heading-retaining receiving approach'),
    'heading_passing': dict(start=(3., -2.1, 0.), cart=(8.6, -2.1),
        person=(2.85, -6.05), cart_trigger=4., person_trigger=.1,
        parked=None, purpose='Observed oncoming equipment and heading-preserving lateral clearance'),
    'offset_portal': dict(start=(11.0, -1.65, 0.), cart=(8.8, -.95),
        person=(2.85, -6.05), cart_trigger=.1, person_trigger=.1,
        parked=None, purpose='Offset approach through a clear office portal with a constrained terminal heading'),
    'encounter_entry': dict(start=(9.0, -2.1, 0.), cart=(12.7, -2.1),
        person=(12.8, .45), cart_trigger=3.6, person_trigger=2.4,
        parked=(11.95, -1.95), purpose='Encounter terminal configuration coupled to subsequent occupied doorway entry'),
}

SPECS['heading_constrained_passage'] = dict(SPECS['static_detour'],
    purpose='Same static detour geometry, retain zero body heading through x<=10 m, then release for doorway entry')

SPECS['multiroom_service'] = dict(SPECS['collection_delivery'],
    person=(6.2,-3.4), cart_travel_s=36.,
    purpose='One compiled 617 pickup, 614 exchange, 619 delivery task; trolley completes passage without parking at a service doorway')

SCENARIOS += ('multiroom_service_office_contact',)
SPECS['multiroom_service_office_contact'] = dict(SPECS['multiroom_service'],
    person=(2.85,-6.05),
    purpose='Multiroom service with the receiving person stationed inside office 614 and a moving trolley in the corridor')

# Static application trace isolates service sequencing and room maneuvers.
# Dynamic-interaction outcomes remain separate, including failed attempts.
SCENARIOS += ('multiroom_service_static',)
SPECS['multiroom_service_static'] = dict(SPECS['multiroom_service'],
    cart=(2., -.95), person=(2.85, -6.05), cart_travel_s=0.,
    purpose='Static furnished multiroom service application; no dynamic-interaction performance claim')

# Appearance is counterbalanced without changing geometry, object IDs or goals.
SCENARIOS += ('object_reference_a','object_reference_b')
for name in ('object_reference_a','object_reference_b'):
    SPECS[name] = dict(start=(3.,-2.1,0.),cart=(2.,-.95),person=(2.85,-6.05),
        cart_trigger=.1,person_trigger=.1,cart_travel_s=0.,parked=None,
        purpose='Counterbalanced visual referring objects with identical geometry and navigation objective')

SPECS['collection_return_delivery'] = dict(SPECS['collection_delivery'], start=(6.,-2.1,0.), cart=(10.,-2.1),
    purpose='Return to a registered collection station behind the initial heading, then deliver to Office 619 without reverse retreat')

for name, width in [('delivery_aisle_open', 1.90), ('delivery_aisle_narrow', 1.50)]:
    SPECS[name] = dict(start=(10.4, -2.75, 0.), cart=(12.7, -2.1),
        person=(2.85, -6.05), cart_trigger=4., person_trigger=.1,
        parked=None, aisle_width=width, aisle_x=(11.0,12.0),
        purpose='One document-delivery route: pass equipment aisle, then enter Office 619')

def add_equipment_rack(world, name, xmin, xmax, ymin, ymax):
    model=ET.SubElement(world, 'model', name=name)
    ET.SubElement(model, 'static').text='true'
    ET.SubElement(model, 'pose').text=f'{(xmin+xmax)/2} {(ymin+ymax)/2} .45 0 0 0'
    link=ET.SubElement(model,'link',name='rack')
    for tag in ['collision','visual']:
        item=ET.SubElement(link,tag,name=name+'_'+tag)
        geometry=ET.SubElement(ET.SubElement(item,'geometry'),'box')
        ET.SubElement(geometry,'size').text=f'{xmax-xmin} {ymax-ymin} .9'
        if tag=='visual':
            material=ET.SubElement(item,'material')
            ET.SubElement(material,'ambient').text='.39 .43 .47 1'
            ET.SubElement(material,'diffuse').text='.48 .52 .56 1'


def create_world(output, scenario):
    spec=SPECS[scenario]
    metadata=create_base_world(output, 'headon')
    tree=ET.parse(output);world=tree.getroot().find('world')
    # Both fire-door leaves are parked against their corridor walls. The
    # furnished source SDF leaves the north leaf across the corridor and the
    # south leaf diagonal; their 1.71 m collision shapes disconnect the
    # approach when the Ranger changes heading. Normalize both open poses in
    # every research variant, rather than letting this map artifact dominate
    # a planner comparison.
    for name,yaw in (('fire_door_north_leaf',math.pi/2),
                     ('fire_door_south_leaf',-math.pi/2)):
        leaf=world.find(f"model[@name='{name}']/pose")
        leaf_pose=list(map(float,leaf.text.split()))
        leaf_pose[5]=yaw
        leaf.text=' '.join(map(str,leaf_pose))
    x,y,yaw=spec['start']
    world.find("model[@name='rangermini_2_0']/pose").text=f'{x} {y} .01 0 0 {yaw}'
    x,y=spec['cart'];world.find("model[@name='service_trolley']/pose").text=f'{x} {y} .12 0 0 0'
    x,y=spec['person'];world.find("model[@name='lab_student']/pose").text=f'{x} {y} .02 0 0 {-math.pi/2}'
    if spec['parked'] is not None:
        station=copy.deepcopy(world.find("model[@name='service_trolley']"))
        station.set('name','entry_equipment_station')
        x,y=spec['parked'];station.find('pose').text=f'{x} {y} .12 0 0 0'
        world.append(station)
    if scenario in ('object_reference_a','object_reference_b'):
        palette=((.80,.08,.06,1.),(.05,.22,.82,1.))
        if scenario.endswith('_b'):
            palette=palette[::-1]
        for index,(y,color) in enumerate(zip((-.95,-3.25),palette)):
            station=copy.deepcopy(world.find("model[@name='service_trolley']"))
            station.set('name','entry_equipment_station'+('' if index==0 else '_2'))
            station.find('pose').text=f'6.5 {y} .12 0 0 0'
            station.find('static').text='true'
            base=ET.SubElement(station,'link',name='reference_payload_link')
            item=ET.SubElement(base,'visual',name='reference_payload')
            ET.SubElement(item,'pose').text='0 0 .76 0 0 0'
            ET.SubElement(ET.SubElement(ET.SubElement(item,'geometry'),'box'),'size').text='.38 .30 .22'
            material=ET.SubElement(item,'material')
            for tag in ('ambient','diffuse'):
                ET.SubElement(material,tag).text=' '.join(map(str,color))
            world.append(station)
    if scenario == 'offset_portal':
        # Preserve the offset entry and requested final heading while making
        # the aperture feasible for the footprint and the planner's margin.
        center=12.79;half=.75
        for side,sign in (('-0.46',-1),('0.46',1)):
            jamb=world.find(f"model[@name='door_619_jamb_{side}']")
            pose=list(map(float,jamb.findtext('pose').split()))
            pose[0]=center+sign*half
            jamb.find('pose').text=' '.join(map(str,pose))
        # The furnished corridor wall is split around the original doorway.
        # Move those physical segments to the new jambs as well.
        left=world.find("model[@name='north_corridor_wall_3']")
        right=world.find("model[@name='north_corridor_wall_4']")
        for wall,xmin,xmax in ((left,8.55,center-half-.0225),
                               (right,center+half+.0225,13.95)):
            pose=list(map(float,wall.findtext('pose').split()));pose[0]=(xmin+xmax)/2
            wall.find('pose').text=' '.join(map(str,pose))
            for shape in ('collision','visual'):
                wall.find(f"link/{shape}/geometry/box/size").text=f'{xmax-xmin:.6f} 0.18 2.4'
        lintel=world.find("model[@name='door_619_lintel']")
        pose=list(map(float,lintel.findtext('pose').split()));pose[0]=center
        lintel.find('pose').text=' '.join(map(str,pose))
        for shape in ('collision','visual'):
            lintel.find(f"link/{shape}/geometry/box/size").text='1.58 0.18 0.07'
        leaf=world.find("model[@name='door_619_open_leaf']")
        pose=list(map(float,leaf.findtext('pose').split()));pose[0]=center-half+.025
        leaf.find('pose').text=' '.join(map(str,pose))
    if 'aisle_width' in spec:
        xmin,xmax=spec['aisle_x'];half=spec['aisle_width']/2
        add_equipment_rack(world,'task_aisle_rack_north',xmin,xmax,-2.1+half,-.39)
        add_equipment_rack(world,'task_aisle_rack_south',xmin,xmax,-3.81,-2.1-half)
    tree.write(output,encoding='unicode',xml_declaration=True)
    metadata.update(scenario=scenario,protocol_version='motion_planning_research_v1',
        start_pose=list(spec['start']),trolley_initial=list(spec['cart']),
        person_initial=list(spec['person']),parked_equipment_pose=spec['parked'],
        purpose=spec['purpose'],scene_spec=spec,
        actor_trigger_source='independent measured robot proximity; identical for all methods',
        validation_status='generated_scene; no outcome asserted')
    metadata['fire_door_leaf_pose'] = {'north_yaw_rad': math.pi/2,
        'south_yaw_rad': -math.pi/2,
        'basis':'both open leaves aligned with the corridor walls'}
    if scenario == 'offset_portal':
        metadata['portal_clear_width_m']=1.50
        metadata['portal_change_reason']='Original 0.875 m opening plus a parked trolley at the portal prevented a feasible approach under the 0.751 m footprint and 0.10 m safety margin. Portal widened and immediate doorway clutter removed.'
    if scenario in ('multiroom_service', 'multiroom_service_office_contact', 'multiroom_service_static', 'collection_delivery', 'collection_return_delivery'):
        metadata['service_protocol'] = dict(version='collection_delivery_v2',
            collection_pose=[4.4,-2.1,-math.pi/2], departure_pose=[4.4,-2.1,0.],
            corridor_pose=[6.,-2.1,0.], collection_verification='stationary pose acknowledgement',
            fire_door_south_leaf_yaw=-math.pi/2,
            task_source='registered service locations; not autonomous task decomposition')
    if 'aisle_width' in spec:
        metadata['task_protocol'] = dict(version='delivery_aisle_v3',
            aisle_x=list(spec['aisle_x']), aisle_y=[-2.1-spec['aisle_width']/2,-2.1+spec['aisle_width']/2],
            actor_direction='negative_x', priority_rule='robot footprint must not enter aisle before the trolley rear clears its entrance',
            clearance_evaluation='independent acknowledged actor poses plus registered static collision boxes',
            completion='verified indoor arrival within episode limit, no evaluated overlap, plus instruction compliance')
    return metadata

def fixture(args):
    spec=SPECS[args.scenario]
    continuous.CART_START=spec['cart'];continuous.PERSON_START=spec['person']
    continuous.CART_TRIGGER_DISTANCE=spec['cart_trigger']
    continuous.PERSON_TRIGGER_DISTANCE=spec['person_trigger']
    # Existing fixture compares robot distance to these evaluation-only locations.
    continuous.PORTAL_APPROACH=(12.8,-1.15)
    def actors(now,triggers):
        cart_dt=0. if triggers['trolley'] is None else max(0.,now-triggers['trolley'])
        person_dt=0. if triggers['person'] is None else max(0.,now-triggers['person'])
        cart=spec['cart'];person=spec['person']
        if args.scenario not in ('offset_portal', 'office_handoff'):
            cart=(cart[0]-.22*min(spec.get('cart_travel_s',25.),cart_dt),cart[1])
        if args.scenario=='encounter_entry':
            person=(person[0],person[1]-.28*min(13.75,person_dt))
        return cart,person
    continuous.actor_positions=actors
    continuous.fixture(args)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--scenario',choices=SCENARIOS,required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--generate',action='store_true')
    args=ap.parse_args()
    if args.generate:
        result=create_world(args.output,args.scenario)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    else:fixture(args)
