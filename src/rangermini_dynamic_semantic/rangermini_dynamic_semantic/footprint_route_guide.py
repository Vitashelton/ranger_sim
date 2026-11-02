"""Static footprint-aware A* reference for local MPPI; no semantic decisions."""
import heapq
import math
import numpy as np


class FootprintRouteGuide:
    def __init__(self, graph, resolution=.06):
        self.graph = graph
        self.resolution = resolution
        self.boxes = np.asarray(graph.static_boxes)
        self.cache = {}
        self.pose_grid = None
        self.guide_yaw = None
        self.pose_path = []

    def free(self, points, yaw, margin=.10):
        pts = np.asarray(points).reshape(-1, 2)
        x,y,a,hx,hy=self.boxes.T
        dx=x[None,:]-pts[:,0:1];dy=y[None,:]-pts[:,1:2]
        c,s=math.cos(yaw),math.sin(yaw)
        ca,sa=np.cos(a),np.sin(a)
        co,si=np.abs(np.cos(yaw-a)),np.abs(np.sin(yaw-a))
        hl,hw=.751/2,.500/2
        sep=np.maximum.reduce([
            np.abs(dx*c+dy*s)-hl-hx*co-hy*si,
            np.abs(-dx*s+dy*c)-hw-hx*si-hy*co,
            np.abs(dx*ca+dy*sa)-hx-hl*co-hw*si,
            np.abs(-dx*sa+dy*ca)-hy-hl*si-hw*co])
        return np.all(sep>=margin,axis=1)

    def segment(self, start, end, yaw):
        count=max(2,math.ceil(math.dist(start,end)/.04)+1)
        return bool(self.free(np.linspace(start,end,count),yaw).all())

    def guide(self, pose, target, target_yaw=None):
        # The reference must cover the requested task, including room goals.
        # The former corridor-only upper bound (-0.69 m) made every interior
        # pose goal unreachable to this search, silently dropping the route.
        bounds = (min(-3.4, math.floor((min(pose[1], target[1])-.6)/.6)*.6),
                  max(-.69, math.ceil((max(pose[1], target[1])+.6)/.6)*.6))
        if bounds != getattr(self, '_y_bounds', None):
            self._y_bounds = bounds
            self.cache.clear()
            self.pose_grid = None
        self.guide_yaw = None
        self.pose_path = []
        self.target_yaw = target_yaw
        if target_yaw is not None:
            guide,path = self._pose_guide(pose,target)
        else:
            guide,path = self._fixed_guide(pose,target)
            if guide is None and not self.segment(pose[:2],target,pose[2]):
                guide,path = self._pose_guide(pose,target)
        # Grid cells approximate the route, not the task's arrival pose.
        # Preserve its exact endpoint for every downstream planner. Otherwise
        # grid error plus controller tolerance can leave the robot outside
        # the unchanged measured-arrival threshold.
        if path and math.dist(path[-1][:2],target)>1e-6:
            yaw=self.pose_path[-1][2] if self.pose_path else pose[2]
            if self.segment(path[-1][:2],target,yaw):
                path=list(path)+[tuple(target)]
                if self.pose_path:self.pose_path.append((*target,yaw))
        return guide,path

    def _fixed_guide(self, pose, target):
        start=np.asarray(pose[:2]);yaw=pose[2]
        if self.segment(start,target,yaw):
            return None, []
        # Grid is confined to the registered corridor; doorway capture remains
        # the shared entry controller. Heading-specific footprint, not a point.
        yaw_key=round(yaw/.10)*.10
        if yaw_key not in self.cache:
            xs=np.arange(.48,17.6,self.resolution);ys=np.arange(*self._y_bounds,self.resolution)
            xx,yy=np.meshgrid(xs,ys)
            points=np.c_[xx.ravel(),yy.ravel()]
            valid=self.free(points,yaw_key,margin=.12).reshape(len(ys),len(xs))
            self.cache[yaw_key]=(xs,ys,valid)
        xs,ys,valid=self.cache[yaw_key]
        def closest(p):
            available=np.argwhere(valid)
            order=np.argsort((xs[available[:,1]]-p[0])**2+(ys[available[:,0]]-p[1])**2)
            for k in order[:20]:
                y,x=map(int,available[k]);point=(xs[x],ys[y])
                if self.segment(p,point,yaw):return y,x
            return None
        origin,destination=closest(start),closest(target)
        if origin is None or destination is None:return None,[]
        queue=[(0.,0.,origin)];cost={origin:0.};parent={}
        while queue:
            _,distance,node=heapq.heappop(queue)
            if distance>cost[node]+1e-9:continue
            if node==destination:break
            for dy,dx in ((0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)):
                y,x=node[0]+dy,node[1]+dx
                if not(0<=y<len(ys) and 0<=x<len(xs)) or not valid[y,x]:continue
                if dx and dy and not(valid[node[0],x] and valid[y,node[1]]):continue
                nxt=(y,x);alternative=distance+math.hypot(dx,dy)
                if alternative>=cost.get(nxt,float('inf')):continue
                cost[nxt]=alternative;parent[nxt]=node
                h=math.hypot(destination[0]-y,destination[1]-x)
                heapq.heappush(queue,(alternative+h,alternative,nxt))
        if destination not in cost:return None,[]
        path=[destination]
        while path[-1]!=origin:path.append(parent[path[-1]])
        path=[(float(xs[x]),float(ys[y])) for y,x in reversed(path)]
        # A line-of-sight lookahead avoids grid zig-zags and checks the actual,
        # unquantized heading before supplying the local objective.
        options=[p for p in path if .35<=math.dist(start,p)<=1.25 and self.segment(start,p,yaw)]
        if not options:return None,path
        return max(options,key=lambda p:path.index(p)),path

    def _pose_guide(self, pose, target):
        # Heading is a search variable when a translation-only reference has
        # no connected corridor. MPPI still validates finite-rate execution.
        step=.12;count=24
        if self.pose_grid is None:
            xs=np.arange(.48,17.6,step);ys=np.arange(*self._y_bounds,step)
            xx,yy=np.meshgrid(xs,ys);pts=np.c_[xx.ravel(),yy.ravel()]
            valid=np.stack([self.free(pts,k*2*math.pi/count,.105).reshape(xx.shape) for k in range(count)])
            self.pose_grid=(xs,ys,valid)
        xs,ys,valid=self.pose_grid
        iy=int(np.argmin(abs(ys-pose[1])));ix=int(np.argmin(abs(xs-pose[0])))
        ih=round(pose[2]*count/(2*math.pi))%count
        starts=[]
        for y in range(max(0,iy-2),min(len(ys),iy+3)):
            for x in range(max(0,ix-2),min(len(xs),ix+3)):
                if valid[ih,y,x] and self.segment(pose[:2],(xs[x],ys[y]),pose[2]):
                    starts.append((math.hypot(xs[x]-pose[0],ys[y]-pose[1]),(ih,y,x)))
        if not starts:return None,[]
        origin=min(starts)[1];queue=[(0.,0.,origin)];cost={origin:0.};parent={};destination=None
        rotation_cache={}
        while queue:
            _,distance,node=heapq.heappop(queue)
            if distance>cost[node]+1e-9:continue
            h,y,x=node;angle=h*2*math.pi/count
            if math.hypot(xs[x]-target[0],ys[y]-target[1])<.16:
                target_yaw = getattr(self, 'target_yaw', None)
                yaw_error = (0. if target_yaw is None else
                    math.atan2(math.sin(target_yaw-angle),math.cos(target_yaw-angle)))
                heading_reached = target_yaw is None or abs(yaw_error) <= math.pi/count + 1e-9
                connected = self.segment((xs[x],ys[y]),target,angle)
                terminal_turn = (target_yaw is None or all(
                    bool(self.free([target],angle+fraction*yaw_error,.10)[0])
                    for fraction in (0.,.25,.5,.75,1.)))
                if heading_reached and connected and terminal_turn:
                    destination=node;break
            neighbors=[]
            for dy,dx in ((0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)):
                yy,xx=y+dy,x+dx
                if not(0<=yy<len(ys) and 0<=xx<len(xs)) or not valid[h,yy,xx]:continue
                if step*(dx*math.cos(angle)+dy*math.sin(angle))<-.02:continue
                if dx and dy and not(valid[h,y,xx] and valid[h,yy,x]):continue
                neighbors.append(((h,yy,xx),math.hypot(dx,dy)))
            for turn in (-1,1):
                hh=(h+turn)%count
                if not valid[hh,y,x]:continue
                key=(h,y,x,turn)
                if key not in rotation_cache:
                    rotation_cache[key]=bool(self.free([(xs[x],ys[y])],angle+turn*math.pi/count,.10)[0])
                if rotation_cache[key]:neighbors.append(((hh,y,x),.7))
            for nxt,edge in neighbors:
                alternative=distance+edge
                if alternative>=cost.get(nxt,float('inf')):continue
                cost[nxt]=alternative;parent[nxt]=node
                heuristic=math.hypot(xs[nxt[2]]-target[0],ys[nxt[1]]-target[1])/step
                heapq.heappush(queue,(alternative+heuristic,alternative,nxt))
        if destination is None:return None,[]
        path=[destination]
        while path[-1]!=origin:path.append(parent[path[-1]])
        self.pose_path=[(float(xs[x]),float(ys[y]),h*2*math.pi/count) for h,y,x in reversed(path)]
        selected=self.pose_path[0]
        # A discretized SE(2) route represents an in-place turn as adjacent
        # states at the same position. Stage at the pre-turn state first;
        # once physically there, advance the guide to the next heading state.
        # This prevents the local MPPI horizon from replacing a required
        # translate-then-turn sequence with an infeasible chord.
        turn_edge=None
        for index,(previous,point) in enumerate(zip(self.pose_path,self.pose_path[1:])):
            angular=abs(math.atan2(math.sin(point[2]-previous[2]),
                                   math.cos(point[2]-previous[2])))
            if math.dist(previous[:2],point[:2])<step*.25 and angular>math.pi/count*.5:
                turn_edge=index
                break
        if turn_edge is not None:
            before=self.pose_path[turn_edge]
            after=self.pose_path[turn_edge+1]
            if math.dist(pose[:2],before[:2])>.08:
                # A pre-turn pose may be metres away beyond translation
                # corners. Only expose a short, footprint-visible chord;
                # otherwise a local sampler aims through the intervening rack.
                length=0.
                for index,point in enumerate(self.pose_path[:turn_edge+1]):
                    if index:
                        length+=math.dist(self.pose_path[index-1][:2],point[:2])
                    if length>.72:
                        break
                    if self.segment(pose[:2],point[:2],pose[2]):
                        selected=point
            else:
                selected=after
        else:
            length=0.
            for previous,point in zip(self.pose_path,self.pose_path[1:]):
                length+=math.dist(previous[:2],point[:2])
                if length>0.72:break
                if self.segment(pose[:2],point[:2],pose[2]):
                    selected=point
        self.guide_yaw=selected[2]
        return selected[:2],[p[:2] for p in self.pose_path]
