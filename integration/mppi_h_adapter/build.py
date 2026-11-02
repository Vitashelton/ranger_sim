#!/usr/bin/env python3
"""Compile upstream solver, with explicit, recorded platform limit substitutions."""
from pathlib import Path
import shutil,subprocess,json,hashlib
root=Path(__file__).resolve().parent
up=root.parent/'mppi_h_upstream'
build=root/'build';build.mkdir(exist_ok=True)
inc=build/'include';shutil.copytree(up/'src/control/mppi_h/include',inc,dirs_exist_ok=True)
for mode, name in [('mode1_mppi_3d','mppi_3d_core.hpp'),('mode2_mppi_4d','mppi_4d_core.hpp')]:
 p=inc/mode/name
 p.write_text(p.read_text().replace('const int random_seed_ = 623;', 'const int random_seed_ = ranger_mppi_seed;'))
changes=[]
for name in ['common_type.hpp','mode1_mppi_3d/mppi_3d_setting.hpp','mode2_mppi_4d/mppi_4d_setting.hpp']:
 p=inc/name;old=p.read_text();new=old
 new=new.replace('= -2.0;', '= -0.35;').replace('= +2.0;', '= +0.35;')
 if name=='common_type.hpp':new=new.replace('= -1.57;', '= -0.9;').replace('= +1.57;', '= +0.9;')
 if name=='common_type.hpp':
  new='#include "ranger_projection.hpp"\n'+new
  new=new.replace('omega = std::max(omega_min, std::min(omega_max, omega));','omega = std::max(omega_min, std::min(omega_max, omega));\n            ranger_project(vx, vy, omega);')
 elif name.startswith('mode2'):
  new=new.replace('return cmd3d;','cmd3d.clamp();\n    return cmd3d;')
 if name!='common_type.hpp':
  new=new.replace('param.controller.ref_velocity, 2)', 'param.controller.ref_velocity * std::min(1.,std::hypot(state.x-goal_state.x,state.y-goal_state.y)/.5), 2)')
  marker='return cost;'
  idx=new.rfind(marker)
  new=new[:idx]+"if (std::hypot(state.x-goal_state.x,state.y-goal_state.y)<.35) { double e=std::remainder(state.yaw-goal_state.yaw,2*M_PI); cost += param.controller.weight_angular_error*e*e; }\n        "+new[idx:]
 p.write_text(new)
 changes.append({'file':name,'before_sha256':hashlib.sha256(old.encode()).hexdigest(),'after_sha256':hashlib.sha256(new.encode()).hexdigest(),'change':'Ranger speed limits and nearest native-mode projection in rollout and command conversion; shared by 3D and 4D'})
cmd=['g++','-O3','-DNDEBUG','-std=c++17','-shared','-fPIC','-fopenmp','-include','chrono','-include','numeric','-I/usr/include/eigen3','-I'+str(root/'compat'),'-I'+str(inc),str(root/'bridge.cpp')]
cmd += [str(up/'src/control/mppi_h/src'/n) for n in ['mppi_h_core.cpp','mode1_mppi_3d/mppi_3d_core.cpp','mode2_mppi_4d/mppi_4d_core.cpp']]
cmd+=['-o',str(build/'libmppi_h.so')]
subprocess.run(cmd,check=True)
(root/'provenance.json').write_text(json.dumps(dict(upstream_url='https://github.com/MizuhoAOKI/mppi_swerve_drive_ros',upstream_commit=subprocess.check_output(['git','-C',str(up),'rev-parse','HEAD'],text=True).strip(),solver_cpp_modified=False,header_substitutions=changes,grid_interface='row-major grid view: nearest and bilinear interpolation',random_seed='episode seed, default upstream 623 exposed as parameter',threads=1,seed_header_change='const seed 623 exposed through constructor-time bridge variable',binary_sha256=hashlib.sha256((build/'libmppi_h.so').read_bytes()).hexdigest(),build_command=cmd),indent=2))
