from pathlib import Path
import shutil, zipfile, json, math, xml.etree.ElementTree as E
import copy
from PIL import Image,ImageDraw,ImageFont

ROOT=Path('/home/USER/ranger_sim_ws')
PKG=ROOT/'src/rangermini_dynamic_semantic'
AS=PKG/'models/lab_assets'
AS.mkdir(parents=True,exist_ok=True)
PERCEPTION=PKG/'models/perception';PERCEPTION.mkdir(exist_ok=True)
if not (PERCEPTION/'yolov8n.onnx').exists():shutil.copy2('/home/USER/yolov8n.onnx',PERCEPTION/'yolov8n.onnx')
for name in ['cafe_table','person_standing','control_console','cardboard_box']:
    source=Path('/home/USER/.gazebo/models')/name
    if source.exists():shutil.copytree(source,AS/name,dirs_exist_ok=True)
for archive,name in [('/tmp/ranger_office_chair.zip','OfficeChairBlack'),('/tmp/ranger_monitor.zip','MonitorAndKeyboard')]:
    dest=AS/name;dest.mkdir(exist_ok=True)
    if not Path(archive).exists() and (dest/'model.sdf').exists():continue
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            if '..' in Path(entry.filename).parts:raise ValueError('unsafe archive')
        z.extractall(dest)
for dae in AS.rglob('*.dae'):
    data=E.parse(dae)
    if data.getroot().tag.startswith('{'):
        E.register_namespace('',data.getroot().tag[1:].split('}')[0])
    for node in data.getroot().iter():
        if node.tag.endswith('init_from') and node.text:
            matches=list(AS.rglob(Path(node.text).name))
            if matches:node.text=str(matches[0])
    data.write(dae,encoding='unicode',xml_declaration=True)
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
TEX=AS/'textures';TEX.mkdir(exist_ok=True)
def texture(number,kind):
    p=TEX/f'room_{number}.png';im=Image.new('RGB',(768,384),'#193e59');d=ImageDraw.Draw(im)
    d.rounded_rectangle((12,12,755,371),radius=16,outline='#d8e4eb',width=7)
    d.text((50,18),number,font=ImageFont.truetype(FONT,158),fill='white')
    d.text((54,225),kind,font=ImageFont.truetype(FONT,65),fill='#dee9ee');im.save(p)
    return p
def obj_plane(name,w,h,image):
    p=TEX/f'{name}.obj'
    (TEX/f'{name}.mtl').write_text(f'newmtl sign\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nmap_Kd {image}\n')
    p.write_text(f'mtllib {name}.mtl\nusemtl sign\nv {-w/2} 0 {-h/2}\nv {-w/2} 0 {h/2}\nv {w/2} 0 {h/2}\nv {w/2} 0 {-h/2}\nvt 1 0\nvt 1 1\nvt 0 1\nvt 0 0\nvn 0 1 0\nf 1/1/1 2/2/1 3/3/1 4/4/1\n')
    return p
tree=E.parse(PKG/'worlds/ranger_lab_corridor_firedoor_half_open.sdf');world=tree.getroot().find('world')
for name,angle in [('fire_door_north_leaf',90),('fire_door_south_leaf',-45)]:
    leaf=world.find(f"model[@name='{name}']")
    pose=leaf.find('pose').text.split();pose[-1]=str(math.radians(angle));leaf.find('pose').text=' '.join(pose)
def element(tag,text,parent):E.SubElement(parent,tag).text=str(text)
def box(name,pos,size,color=(.7,.7,.7),collision=True,rot=0):
    m=E.SubElement(world,'model',name=name);element('static','true',m);element('pose',' '.join(map(str,(*pos,0,0,rot))),m)
    l=E.SubElement(m,'link',name='body')
    for tag in (['collision','visual'] if collision else ['visual']):
        v=E.SubElement(l,tag,name=tag);element('size',' '.join(map(str,size)),E.SubElement(E.SubElement(v,'geometry'),'box'))
        if tag=='visual':
            mat=E.SubElement(v,'material')
            for k in ['ambient','diffuse']:element(k,' '.join(map(str,(*color,1))),mat)
    return m
def asset(name,source,pos,yaw=0):
    m=E.parse(AS/source/'model.sdf').getroot().find('model');m.set('name',name)
    old=m.find('pose')
    if old is not None:m.remove(old)
    element('pose',' '.join(map(str,(*pos,0,0,yaw))),m)
    s=m.find('static')
    if s is None:element('static','true',m)
    else:s.text='true'
    for uri in m.findall('.//uri'):
        if uri.text.startswith('model://'):
            uri.text='file://'+str(AS/uri.text[len('model://'):])
    world.append(m);return m
def plaque(name,pos,image,w=.50,h=.25,yaw=0):
    m=box(name,pos,(w,.025,h),(.18,.25,.31),False,rot=yaw)
    vis=E.SubElement(m.find('link'),'visual',name='printed_sign');element('pose','0 .014 0 0 0 0',vis)
    element('uri','file://'+str(obj_plane(name,w,h,image)),E.SubElement(E.SubElement(vis,'geometry'),'mesh'))
    mat=E.SubElement(vis,'material');element('diffuse','1 1 1 1',mat);element('ambient','1 1 1 1',mat)
    metal=E.SubElement(E.SubElement(mat,'pbr'),'metal')
    element('albedo_map','file://'+str(image),metal);element('metalness',0,metal);element('roughness',.8,metal)
def door(x,y,num,kind,side):
    # Same existing opening; thin jambs and a leaf rotated into the room.
    for dx in [-.46,.46]:box(f'door_{num}_jamb_{dx}',(x+dx,y,1.05),(.045,.18,2.1),(.18,.20,.22))
    box(f'door_{num}_lintel',(x,y,2.09),(.98,.18,.07),(.18,.20,.22))
    hinge=(x-.435,y);angle=-math.pi/2 if side=='south' else math.pi/2
    cx=hinge[0]+.42*math.cos(angle);cy=hinge[1]+.42*math.sin(angle)
    leaf=box(f'door_{num}_open_leaf',(cx,cy,1.02),(.84,.045,2.04),(.48,.28,.12),rot=angle)
    link=leaf.find('link')
    for tag in ['visual','collision']:
        v=E.SubElement(link,tag,name='handle');element('pose','.32 .055 -.08 0 0 0',v)
        element('size','.12 .03 .025',E.SubElement(E.SubElement(v,'geometry'),'box'))
        if tag=='visual':
            mat=E.SubElement(v,'material');element('diffuse','.75 .76 .78 1',mat);element('ambient','.75 .76 .78 1',mat)
    plaque('room_sign_'+num,(x+.68,y+.11 if side=='south' else y-.11,1.10),texture(num,kind),yaw=0 if side=='south' else math.pi)

# Change furniture visuals, not the corridor topology or width.
for m in list(world.findall('model')):
    if m.get('name','').startswith(('desk_','lab_bench_')):world.remove(m)
for x,num in [(1.98,'613'),(5.04,'615'),(8.10,'617'),(11.79,'619'),(14.4,'621')]:
    door(x,-.30,num,'教师办公室','north')
    asset('office_desk_'+num,'cafe_table',(x,2.1,.02))
    asset('office_chair_'+num,'OfficeChairBlack',(x+.3,1.25,.02))
    asset('office_monitor_'+num,'MonitorAndKeyboard',(x,2.1,.82),math.pi/2)
for x,num in [(3.24,'614'),(7.65,'616')]:
    door(x,-3.90,num,'学生实验室','south')
    asset('lab_table_'+num,'cafe_table',(x,-5.35,.02))
    asset('lab_chair_'+num,'OfficeChairBlack',(x+.55,-4.70,.02),math.pi)
    asset('lab_monitor_'+num,'MonitorAndKeyboard',(x,-5.35,.82),-math.pi/2)
    asset('lab_console_'+num,'control_console',(x+1.05,-6.55,0),math.pi)
# Observable objects for the four task families, away from this stationary probe.
asset('lab_student','person_standing',(2.85,-6.05,.02),math.pi)
asset('corridor_box','cardboard_box',(5.4,-1.35,.02))
# Recognizable service trolley: open shelves, uprights, handle and four wheels.
cart=box('service_trolley',(8.9,-1.1,.12),(.70,.42,.05),(.55,.58,.62))
for z in [.52,.9]:box(f'trolley_shelf_{z}',(8.9,-1.1,z),(.70,.42,.035),(.7,.73,.76))
for dx in [-.31,.31]:
    for dy in [-.17,.17]:
        box(f'trolley_post_{dx}_{dy}',(8.9+dx,-1.1+dy,.51),(.025,.025,.8),(.5,.55,.6))
        wheel=box(f'trolley_wheel_{dx}_{dy}',(8.9+dx,-1.1+dy,.065),(.045,.10,.10),(.06,.06,.06))
# Glass showcase with a frame and shelves, providing a compositional-clearance target.
box('glass_showcase_base',(6.0,-.68,.14),(1.2,.35,.28),(.25,.29,.32))
for dx in [-.58,.58]:box(f'glass_showcase_frame_{dx}',(6+dx,-.68,.85),(.03,.35,1.45),(.45,.49,.52))
for z in [.5,1,1.55]:box(f'glass_showcase_shelf_{z}',(6,-.68,z),(1.17,.35,.025),(.6,.7,.75))
glass=box('glass_showcase_pane',(6,-.87,.92),(1.15,.008,1.23),(.65,.8,.85))
element('transparency',.72,glass.find('link/visual'))
# Render-only overview camera, separate from the robot camera.
camera=E.SubElement(world,'model',name='scene_overview_camera');element('static','true',camera)
element('pose','8 -10 11 0 1.0 1.57079632679',camera)
sensor=E.SubElement(E.SubElement(camera,'link',name='camera'),'sensor',name='overview',type='camera')
element('always_on','true',sensor);element('update_rate',10,sensor);element('topic','/scene/overview',sensor)
c=E.SubElement(sensor,'camera');element('horizontal_fov',1.5,c)
im=E.SubElement(c,'image');element('width',1280,im);element('height',720,im)
clip=E.SubElement(c,'clip');element('near',.1,clip);element('far',60,clip)
navcamera=copy.deepcopy(camera);navcamera.set('name','navigation_overhead_camera')
navcamera.find('pose').text='3.6 -3.5 6 0 1.57079632679 -1.57079632679'
navcamera.find('link/sensor/topic').text='/scene/navigation'
navcamera.find('link/sensor/camera/horizontal_fov').text='1.1'
world.append(navcamera)
robot=world.find("model[@name='rangermini_2_0']")
robot.find('pose').text='3.24 -1.75 .01 0 0 -1.57079632679'
robot.find("link/sensor[@name='mid360s_proxy']/pose").text='.30 0 .35 0 0 0'
path=PKG/'worlds/ranger_lab_corridor_furnished.sdf';tree.write(path,encoding='unicode',xml_declaration=True)
manifest={'base_world':'ranger_lab_corridor_firedoor_half_open.sdf','topology_changed':False,'corridor_clear_width_m':3.42,
'models':{'OfficeChairBlack':'https://fuel.gazebosim.org/1.0/OpenRobotics/models/OfficeChairBlack','MonitorAndKeyboard':'https://fuel.gazebosim.org/1.0/OpenRobotics/models/MonitorAndKeyboard'},
'notes':'Gazebo classic assets copied locally; model names are not supplied to detector. Door numbers are scene labels, not OCR outputs.'}
(AS/'ASSET_SOURCES.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
shutil.copy2(__file__,PKG/'scripts/generate_furnished_lab.py')
print(path)
