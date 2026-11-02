"""Build the adapter against installed official Nav2 libraries; solver is unchanged."""
import hashlib,json,shlex,subprocess
from pathlib import Path
import xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent
cmd='source /opt/ros/humble/setup.bash && cmake -S '+shlex.quote(str(HERE))+' -B '+shlex.quote(str(HERE/'build'))+' && cmake --build '+shlex.quote(str(HERE/'build'))+' -j2'
subprocess.run(['bash','-c',cmd],check=True)
version=ET.parse('/opt/ros/humble/share/nav2_mppi_controller/package.xml').getroot().findtext('version')
files=[HERE/'bridge.cpp',HERE/'CMakeLists.txt',HERE/'build/libranger_nav2_mppi.so',Path('/opt/ros/humble/lib/libmppi_controller.so'),Path('/opt/ros/humble/lib/libmppi_critics.so')]
metadata={'upstream':'https://github.com/ros-navigation/navigation2','installed_package':'nav2_mppi_controller','version':version,'motion_model':'Omni','optimizer_and_critics_modified':False,
 'vy_max_m_s':.35,'vy_std_m_s':.18,'regenerate_noises':False,
 'reference':'Shared registered local reference and heading; retained heading encoded in path orientations',
 'execution':'Nearest native-command conversion followed by shared Ranger finite-rate adapter; raw/native/executed commands retained',
 'seed':'Episode seed applied to xtensor default RNG before optimizer initialization; fixed-observation fresh-process first solve verified by check_seed.py. Closed-loop ROS/Gazebo replay is not bitwise deterministic.',
 'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
(HERE/'provenance.json').write_text(json.dumps(metadata,indent=2))
print(json.dumps(metadata,indent=2))
