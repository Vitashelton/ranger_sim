"""Official MPPI-H solver through a C ABI; ROS2/metric-grid/Ranger adapters only."""
import ctypes as C
import math
from pathlib import Path
import time
import heapq
import numpy as np
import cv2
from .interaction_mppi import InteractionMPPI
from .command_semantics_mppi import ranger_command

ROOT=Path(__file__).resolve().parents[3]
DP=C.POINTER(C.c_double)

def occupancy_path(clearance, start, goal, res):
    """8-connected A* reference on the observed inflated occupancy grid."""
    ny,nx=clearance.shape
    free=clearance>.30
    def nearest(p):
        x,y=np.clip(np.rint(p).astype(int),[0,0],[nx-1,ny-1])
        if free[y,x]:return int(y),int(x)
        yy,xx=np.nonzero(free)
        if not len(xx):return None
        i=np.argmin((xx-x)**2+(yy-y)**2)
        return int(yy[i]),int(xx[i])
    origin=nearest(start);destination=nearest(goal)
    if origin is None or destination is None:return []
    q=[(0.,0.,origin)];cost={origin:0.};parent={}
    while q:
        _,g,node=heapq.heappop(q)
        if g>cost[node]+1e-9:continue
        if node==destination:break
        for dy,dx in ((0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)):
            y,x=node[0]+dy,node[1]+dx
            if not(0<=y<ny and 0<=x<nx) or not free[y,x]:continue
            if dx and dy and not(free[node[0],x] and free[y,node[1]]):continue
            nxt=y,x;new=g+math.hypot(dx,dy)*(1+.08/max(.02,clearance[y,x]-.27))
            if new>=cost.get(nxt,float('inf')):continue
            cost[nxt]=new;parent[nxt]=node
            heapq.heappush(q,(new+math.hypot(destination[0]-y,destination[1]-x),new,nxt))
    if destination not in cost:return []
    path=[destination]
    while path[-1]!=origin:path.append(parent[path[-1]])
    path.reverse()
    def visible(a,b):
        yy=np.rint(np.linspace(a[0],b[0],int(math.dist(a,b)*2)+2)).astype(int)
        xx=np.rint(np.linspace(a[1],b[1],len(yy))).astype(int)
        return bool(free[yy,xx].all())
    smooth=[path[0]];i=0
    while i<len(path)-1:
        j=len(path)-1
        while j>i+1 and not visible(path[i],path[j]):j-=1
        smooth.append(path[j]);i=j
    return [(float(x)*res,float(y)*res) for y,x in smooth]

class UpstreamMPPI(InteractionMPPI):
    def __init__(self, config, prediction=True, static_boxes=None, variant='mppi-h', samples=3000):
        super().__init__(config,prediction,static_boxes)
        self.variant=variant
        self.lib=C.CDLL(str(ROOT/'integration/mppi_h_adapter/build/libmppi_h.so'))
        self.lib.mppi_create.argtypes=[C.c_int,C.c_int,C.c_int,C.c_double,C.c_int]
        self.lib.mppi_create.restype=C.c_void_p
        self.lib.mppi_solve.argtypes=[C.c_void_p,DP,DP,C.c_int,C.c_int,C.c_double,C.c_double,C.c_double,DP,DP,DP,DP,DP]
        self.lib.mppi_solve.restype=C.c_int
        self.handle=self.lib.mppi_create({'mppi-h':0,'mppi-3d':1,'mppi-4d':2}[variant],samples,config.horizon,config.dt,config.random_seed)
        self.samples=samples
        self.last_projection=None

    def bank(self,state,main_goal,scan,tracks,stamp,behavior_ids=None,**kwargs):
        start=time.perf_counter()
        bank=[self.behavior_plan(state,main_goal,scan,tracks,stamp,'CONTINUE',**kwargs)]
        self.bank_time_ms=(time.perf_counter()-start)*1000
        return bank

    def behavior_plan(self,state,main_goal,scan,tracks,stamp,behavior,**kwargs):
        begin=time.perf_counter()
        goal=main_goal
        desired_yaw=kwargs.get('goal_yaw')
        direction=math.atan2(goal[1]-state[1],goal[0]-state[0])
        # Upstream reference yaw follows the reference path. Terminal heading
        # is imposed only in the capture region, not throughout a long route.
        reference_yaw=direction if math.dist(state[:2],goal)>.35 else (desired_yaw if desired_yaw is not None else direction)
        res=.06;n=151
        x0=state[0]-(n//2)*res;y0=state[1]-(n//2)*res
        xx,yy=np.meshgrid(x0+np.arange(n)*res,y0+np.arange(n)*res)
        pts=np.c_[xx.ravel(),yy.ravel()]
        start=np.asarray(state[:2]);end=np.asarray(goal)
        segment=end-start;length2=max(float(segment@segment),1e-9)
        alpha=np.clip((pts-start)@segment/length2,0,1)
        distance=np.linalg.norm(pts-(start+alpha[:,None]*segment),axis=1)
        points=np.asarray(self._scan_points_world(state,scan)).reshape(-1,2)
        occupied=np.ones((n,n),dtype=np.uint8)
        if len(points):
            ij=np.rint((points-[x0,y0])/res).astype(int)
            ij=ij[(ij[:,0]>=0)&(ij[:,0]<n)&(ij[:,1]>=0)&(ij[:,1]<n)]
            occupied[ij[:,1],ij[:,0]]=0
        clearance=cv2.distanceTransform(occupied,cv2.DIST_L2,cv2.DIST_MASK_PRECISE).ravel()*res
        # Nav2-style inscribed-radius inflation for the point-costmap solver.
        radius=self.cfg.body_width/2
        collision=np.where(clearance<=radius+.035,100.,np.where(clearance<.65,100.*np.exp(-8.*(clearance-radius-.035)),0.))
        yaw=np.full(n*n,reference_yaw)
        local=occupancy_path(clearance.reshape(n,n),((state[0]-x0)/res,(state[1]-y0)/res),
            ((goal[0]-x0)/res,(goal[1]-y0)/res),res)
        path=[(x+x0,y+y0) for x,y in local] or getattr(self,'reference_path',None)
        if path and len(path)>1:
            best=np.full(len(pts),float('inf'))
            for pa,pb in zip(path,path[1:]):
                pa=np.asarray(pa);d=np.asarray(pb)-pa;den=float(d@d)
                if den<1e-8:continue
                t=np.clip((pts-pa)@d/den,0,1)
                dist=np.linalg.norm(pts-(pa+t[:,None]*d),axis=1)
                mask=dist<best;best[mask]=dist[mask];yaw[mask]=math.atan2(d[1],d[0])
            distance=best
        if math.dist(state[:2],goal)<.35 and desired_yaw is not None:
            yaw.fill(desired_yaw)
        arrays=[np.ascontiguousarray(a,dtype=np.float64) for a in (state,(*goal,desired_yaw if desired_yaw is not None else state[2]),collision,distance,yaw)]
        out=np.zeros(6);traj=np.zeros((self.cfg.horizon,3))
        count=self.lib.mppi_solve(self.handle,arrays[0].ctypes.data_as(DP),arrays[1].ctypes.data_as(DP),n,n,x0,y0,res,
            *[a.ctypes.data_as(DP) for a in arrays[2:]],out.ctypes.data_as(DP),traj.ctypes.data_as(DP))
        raw=tuple(float(v) for v in out[:3])
        # Project a free 4WIDS body twist to the closest Ranger-native command.
        # Keep this conversion outside the unchanged upstream optimizer and log
        # BOTH raw/projected commands. Angular residual is measured at wheelbase/2.
        vx,vy,w=raw
        candidates=[(vx,vy,0.),(vx,0.,w),(0.,0.,w)]
        projected=[]
        for u in candidates:
            p=ranger_command(*u)
            error=(p[0]-vx)**2+(p[1]-vy)**2+(.247*(p[2]-w))**2
            projected.append((error,p,u))
        _,command,native=min(projected,key=lambda v:v[0])
        action=list(native)
        magnitude=math.hypot(*action[:2])
        if magnitude>self.cfg.max_speed:
            action=[v*self.cfg.max_speed/magnitude for v in action]
        if not all(math.isfinite(v) for v in action):
            action=[0.,0.,0.]
        plan=dict(action=action,selected_behavior=self.variant,command_mode=command[3],initial_mode=command[3],
            candidate_count=self.samples,rejected=0,planning_time=(time.perf_counter()-begin)*1000,
            predicted_score=float(out[3]),cost_terms={'upstream_objective':float(out[3])},selected_trajectory=traj[:count].tolist(),
            candidate_population=[],candidate_summary=[],upstream_raw_command=raw,
            upstream_reference_path=path,upstream_sampling_dimension=int(out[5]),upstream_solver_ms=float(out[4]),
            terminal_steering_angles=kwargs.get('current_steering_angles'),terminal_steering_transition=False,
            terminal_mode=command[3],predicted_executed_controls=[])
        return dict(behavior_id='CONTINUE',goal=list(goal),raw_goal=list(goal),guide_projected=False,feasible=True,plan=plan)

    def note_execution(self,*args):
        pass
