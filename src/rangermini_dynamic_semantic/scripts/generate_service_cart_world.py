"""Add a movable cart to the existing laboratory; room geometry stays fixed."""
from pathlib import Path
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parents[1]
tree = ET.parse(root/'worlds/ranger_lab_corridor_firedoor_half_open.sdf')
world = tree.getroot().find('world')
cart = ET.fromstring('''<model name="service_cart"><static>true</static>
<pose>2.0 -0.90 0.0 0 0 0</pose><link name="cart_body">
<collision name="cart_collision"><pose>0 0 0.42 0 0 0</pose><geometry><box><size>0.65 0.50 0.80</size></box></geometry></collision>
<visual name="cart_visual"><pose>0 0 0.42 0 0 0</pose><geometry><box><size>0.65 0.50 0.80</size></box></geometry>
<material><ambient>0.85 0.45 0.1 1</ambient><diffuse>0.85 0.45 0.1 1</diffuse></material></visual>
</link></model>''')
world.append(cart)
output = root/'worlds/ranger_lab_service_cart.sdf'
ET.indent(tree, space='  ')
tree.write(output, encoding='unicode',xml_declaration=True)
print(output)
