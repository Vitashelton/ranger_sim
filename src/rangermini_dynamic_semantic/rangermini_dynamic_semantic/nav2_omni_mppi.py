"""Unmodified installed Nav2 Omni MPPI optimizer, with observed-costmap input.

The adapter retains lateral sampling. Body commands use the common Ranger
execution adapter. A nearest native-command conversion resolves mixed Twist
requests without changing the upstream optimizer or disabling lateral samples.
"""
import ctypes as C
import math
from pathlib import Path
import time
import numpy as np
from .interaction_mppi import InteractionMPPI
from .command_semantics_mppi import ranger_command
ROOT=Path(__file__).resolve().parents[3]
DP=C.POINTER(C.c_double)
UP=C.POINTER(C.c_ubyte)

class Nav2OmniMPPI(InteractionMPPI):
    def __init__(self,config,prediction=False,static_boxes=None,samples=1000):
        super().__init__(config,prediction=False,static_boxes=static_boxes)
        self.samples=samples
        self._runtime_nav2_lib=None
        self._runtime_nav2_handle=None
        self._runtime_grid_cache=None

    def _initialize(self):
        if self._runtime_nav2_handle:return
        lib=C.CDLL(str(ROOT/'integration/nav2_mppi_adapter/build/libranger_nav2_mppi.so'))
        lib.nav2_mppi_create.argtypes=[C.c_int,C.c_int,C.c_double,C.c_double,C.c_int]
        lib.nav2_mppi_create.restype=C.c_void_p
        lib.nav2_mppi_error.restype=C.c_char_p
        lib.nav2_mppi_solve.argtypes=[C.c_void_p,DP,DP,UP,C.c_int,C.c_int,C.c_double,C.c_double,C.c_double,DP,C.c_int,DP,DP]
        handle=lib.nav2_mppi_create(self.samples,self.cfg.horizon,self.cfg.dt,self.cfg.max_speed,self.cfg.random_seed)
        if not handle:raise RuntimeError(lib.nav2_mppi_error().decode())
        self._runtime_nav2_lib=lib;self._runtime_nav2_handle=handle

    def bank(self,state,main_goal,scan,tracks,stamp,behavior_ids=None,**kwargs):
        started=time.perf_counter()
        candidate=self.behavior_plan(state,main_goal,scan,tracks,stamp,'CONTINUE',**kwargs)
        self.bank_time_ms=(time.perf_counter()-started)*1000
        return [candidate]

    def behavior_plan(self,state,main_goal,scan,tracks,stamp,behavior,**kwargs):
        self._initialize();started=time.perf_counter()
        condition=getattr(self,'active_motion_condition',None)
        resolution=.05;n=161;x0=state[0]-n*resolution/2;y0=state[1]-n*resolution/2
        xx,yy=np.meshgrid(x0+(np.arange(n)+.5)*resolution,y0+(np.arange(n)+.5)*resolution)
        grid=np.zeros((n,n),dtype=np.uint8)
        for x,y,yaw,hx,hy in self.static_boxes:
            dx,dy=xx-x,yy-y;c,s=math.cos(yaw),math.sin(yaw)
            grid[(np.abs(c*dx+s*dy)<=hx)&(np.abs(-s*dx+c*dy)<=hy)]=254
        points=np.asarray(self._scan_points_world(state,scan),float).reshape(-1,2)
        if len(points):
            ij=np.floor((points-[x0,y0])/resolution).astype(int)
            ij=ij[(ij[:,0]>=0)&(ij[:,0]<n)&(ij[:,1]>=0)&(ij[:,1]<n)]
            grid[ij[:,1],ij[:,0]]=254
        for track in tracks:
            if getattr(track,'registered_box',None) is not None:continue
            grid[(xx-track.position[0])**2+(yy-track.position[1])**2<=track.radius**2]=254
        # Task input adapter only: preserve the public optimizer/critics.
        # Preferred object clearance is a NONLETHAL graded layer. Hard object
        # geometry and the installed inflation/footprint checker stay intact.
        if condition:
            relevant=set(condition.get('relevant_object_ids',()))
            preferred=float(condition.get('preferred_clearance_m',.65))
            for track in tracks:
                if track.evidence_id not in relevant:continue
                box=getattr(track,'registered_box',None)
                if box is None:
                    gap=np.hypot(xx-track.position[0],yy-track.position[1])-track.radius-.25
                else:
                    bx,by,angle,hx,hy=box;dx,dy=xx-bx,yy-by
                    gap=np.hypot(np.maximum(np.abs(dx*math.cos(angle)+dy*math.sin(angle))-hx,0.),
                                 np.maximum(np.abs(-dx*math.sin(angle)+dy*math.cos(angle))-hy,0.))-.25
                preference=np.clip((preferred-gap)/max(preferred,.01),0.,1.)
                soft=np.asarray(200.*preference,dtype=np.uint8)
                grid=np.maximum(grid,soft)
        # The same registered route reference supplied to the native planner.
        target=tuple(self.route_goal) if getattr(self,'route_goal',None) is not None else tuple(main_goal)
        frozen_path=getattr(self,'reference_path',None)
        if frozen_path:
            # A near staging guide is not the terminal task goal. Retain the
            # shared route ahead. Start from the nearest point on that frozen
            # path; prepending the current pose to its original start makes the
            # controller drive backwards along the path after it has progressed.
            nearest=min(range(len(frozen_path)),
                        key=lambda i:math.dist(state[:2],frozen_path[i][:2]))
            coarse=list(frozen_path[nearest:])
            if math.dist(coarse[0][:2],state[:2])>.05:
                coarse.insert(0,state[:2])
            target=tuple(coarse[-1][:2])
        else:
            coarse=[state[:2],target]
        if math.dist(coarse[0],state[:2])>.05:coarse.insert(0,state[:2])
        if math.dist(coarse[-1],target)>.05:coarse.append(target)
        desired=kwargs.get('goal_yaw') if getattr(self,'reference_path',None) else (
            getattr(self,'route_yaw',None) if getattr(self,'route_goal',None) is not None
            else kwargs.get('goal_yaw'))
        retained=condition and (condition.get('retain_heading') or
                                condition.get('heading_policy')=='preserve_current_heading')
        path=[]
        oriented=list(getattr(self,'reference_pose_path',None) or [])
        if oriented:
            # Preserve the SE(2) reference: a lateral segment's heading is
            # not its travel tangent, and coincident poses may encode a turn.
            # Dropping these orientations invalidates PathAlign's input.
            if math.dist(oriented[0][:2],state[:2])>.05:
                oriented.insert(0,tuple(state))
            segments=zip(oriented,oriented[1:])
        else:
            segments=zip(coarse,coarse[1:])
        for a,b in segments:
            distance=math.dist(a[:2],b[:2]);steps=max(1,math.ceil(distance/.05))
            yaw=(condition['heading'] if retained else math.atan2(b[1]-a[1],b[0]-a[0]))
            turn=math.atan2(math.sin(b[2]-a[2]),math.cos(b[2]-a[2])) if oriented else 0.
            if oriented:steps=max(steps,math.ceil(abs(turn)/.1))
            for k in range(steps):
                alpha=k/steps
                heading=condition['heading'] if retained else a[2]+alpha*turn if oriented else yaw
                path.append((a[0]+alpha*(b[0]-a[0]),a[1]+alpha*(b[1]-a[1]),heading))
        if not path:path=[(*state[:2],state[2])]
        terminal_yaw=condition['heading'] if retained else desired if desired is not None else path[-1][2]
        path.append((*target,terminal_yaw))
        # Limit reference extent to the populated local costmap, as the normal
        # Nav2 controller path handler does, without replacing its geometry.
        near=[]
        for point in path:
            if abs(point[0]-state[0])>3.7 or abs(point[1]-state[1])>3.7:break
            near.append(point)
        path=np.ascontiguousarray(near if len(near)>=2 else path[:2],dtype=np.float64)
        pose=np.ascontiguousarray(state,dtype=np.float64)
        velocity=np.ascontiguousarray(kwargs.get('current_body_velocity',(0.,0.,0.)),dtype=np.float64)
        command=np.zeros(4);trajectory=np.zeros((self.cfg.horizon,3))
        count=self._runtime_nav2_lib.nav2_mppi_solve(self._runtime_nav2_handle,pose.ctypes.data_as(DP),
            velocity.ctypes.data_as(DP),grid.ctypes.data_as(UP),n,n,resolution,x0,y0,path.ctypes.data_as(DP),
            len(path),command.ctypes.data_as(DP),trajectory.ctypes.data_as(DP))
        if count<0:
            message=self._runtime_nav2_lib.nav2_mppi_error().decode()
            return dict(behavior_id='CONTINUE',goal=list(target),goal_yaw=terminal_yaw,raw_goal=list(target),guide_projected=False,
                feasible=False,plan=dict(action=[0.,0.,0.],selected_behavior='STOP',command_mode='STOPPED',
                candidate_count=self.samples,selected_trajectory=[],predicted_score=float('inf'),
                cost_terms={},candidate_summary=[],candidate_population=[],planning_time_ms=(time.perf_counter()-started)*1000,
                public_solver_error=message))
        if not np.isfinite(command).all():raise RuntimeError('Nav2 produced nonfinite command')
        score=float(command[3]);command=command[:3]
        raw=command.copy()
        # Same native command semantics as the Ranger Mini V2 ROS driver:
        # nonzero lateral command selects parallel steering and ignores yaw.
        # Do not replace it with a nearest-mode chooser invented by the adapter.
        # A near-zero translational residue from the optimizer must not turn
        # an intended in-place rotation into a low-speed Ackermann command.
        if math.hypot(command[0], command[1]) < 0.01 and abs(command[2]) > 0.05:
            command[0] = command[1] = 0.
        elif abs(command[1]) > 1e-4:
            command[2] = 0.
        elif abs(command[0]) <= 1e-4:
            command[0] = command[1] = 0.
        peak=math.hypot(*command[:2])
        if peak>self.cfg.max_speed:command[:2]*=self.cfg.max_speed/peak
        mapped=ranger_command(*command)
        plan=dict(action=command.tolist(),selected_behavior='NAV2_OMNI',command_mode=mapped[3],
            initial_mode=kwargs.get('current_mode'),candidate_count=self.samples,
            planning_time_ms=(time.perf_counter()-started)*1000,predicted_score=score,cost_terms={'nav2_min_sample_cost':score},
            selected_trajectory=trajectory[:count].tolist(),candidate_summary=[],candidate_population=[],
            predicted_executed_controls=[],terminal_steering_angles=kwargs.get('current_steering_angles'),
            terminal_steering_transition=False,terminal_mode=mapped[3],public_solver='Nav2 MPPI 1.1.20 Omni',
            public_raw_command=raw.tolist(),public_native_request=command.tolist(),public_projected_command=list(mapped[:3]),public_reference_path=path.tolist(),
            applied_motion_condition=condition,public_lateral_sampling_enabled=True,public_rng_seed=self.cfg.random_seed,public_rng_seed_interface='xtensor default engine seeded before optimizer initialization')
        plan['public_task_condition_adapter']='heading in common reference; graded relevant-object clearance costs capped at 200/254; original optimizer and critics'
        plan['public_path_policy']='shared SE(2) route with heading and in-place turns preserved; cropped at local costmap extent'
        plan['public_command_mapping']='Near-zero translation spin deadband, then Ranger Mini V2 lateral-priority semantics and finite steering execution'
        return dict(behavior_id='CONTINUE',goal=list(target),goal_yaw=terminal_yaw,raw_goal=list(target),guide_projected=False,feasible=True,plan=plan)

    def note_execution(self,*args):
        pass
