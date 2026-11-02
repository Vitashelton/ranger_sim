"""Continuous, single-goal MPPI with task-specific footprint clearance.

Reuses the existing Ranger rollout, scan checks and feasible weighted update.
No registered route bank or intermediate waypoint is used.
"""
import math
from .observation_mppi import ObservationMPPI

class ContinuousSemanticMPPI(ObservationMPPI):
    def __init__(self,*args,semantic_target=None,semantic_radius=.18,
                 desired_gap=.70,semantic_weight=35.,semantic_constraint=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.semantic_target=semantic_target
        self.semantic_radius=semantic_radius
        self.desired_gap=desired_gap
        self.semantic_weight=semantic_weight
        self.semantic_constraint=semantic_constraint
        self.population=[];self.semantic_rejected=0
        self._profile_step=0;self._profile_bias=0.
    def _reference(self,behavior,pose,goal,hypotheses,goal_yaw):
        vx,vy,wz,projection=super()._reference(behavior,pose,goal,hypotheses,goal_yaw)
        # A correlated control perturbation is sampled continuously per rollout.
        # Both geometric and semantic planners use the same distribution.
        phase=self._profile_step/max(1,self.cfg.horizon-1);self._profile_step+=1
        envelope=max(0.,1.-phase)**.65
        if projection=='PARALLEL':
            angle=math.atan2(goal[1]-pose[1],goal[0]-pose[0])-pose[2]
            vx-=math.sin(angle)*self._profile_bias*envelope
            vy+=math.cos(angle)*self._profile_bias*envelope
        elif projection=='ACKERMANN':
            wz+=3.5*self._profile_bias*envelope
        return vx,vy,wz,projection
    def target_gap(self,pose):
        if self.semantic_target is None:return float('inf')
        x,y,yaw=pose;dx,dy=self.semantic_target[0]-x,self.semantic_target[1]-y
        bx=math.cos(yaw)*dx+math.sin(yaw)*dy;by=-math.sin(yaw)*dx+math.cos(yaw)*dy
        return max(0.,math.hypot(max(abs(bx)-self.cfg.body_length/2,0),
                               max(abs(by)-self.cfg.body_width/2,0))-self.semantic_radius)
    def _evaluate(self,start,goal,points,hypotheses,behavior,initial_mode,goal_yaw,
                  sequence=None,current_steering_angles=None,current_steering_transition=False,
                  reference_trajectory=None):
        self._profile_step=0
        if sequence is None:
            self._profile_bias=0. if self._nominal_sample or behavior=='STOP' else max(-.38,min(.38,self.rng.gauss(0.,.24)))
        result=super()._evaluate(start,goal,points,hypotheses,behavior,initial_mode,goal_yaw,
                                 sequence=sequence,current_steering_angles=current_steering_angles,
                                 current_steering_transition=current_steering_transition,
                                 reference_trajectory=reference_trajectory)
        if result is None:return None
        gaps=[self.target_gap(p) for p in result['trajectory'][1:]]
        min_gap=min(gaps) if gaps else self.target_gap(start)
        if self.semantic_target is not None:
            deficit=[max(self.desired_gap-g,0.) for g in gaps]
            exposure=self.cfg.dt*sum(d*d for d in deficit)
            result['cost']+=self.semantic_weight*exposure
            result['cost_terms']['semantic_exposure']=exposure
            result['cost_terms']['semantic_min_gap']=min_gap
            if self.semantic_constraint and min_gap<self.desired_gap:
                self.semantic_rejected+=1
                if sequence is None:self.population.append(dict(feasible=False,rejection='task_clearance',behavior=behavior,trajectory=result['trajectory'],min_semantic_gap=min_gap))
                return None
        if sequence is None:
            self.population.append(dict(feasible=True,behavior=behavior,trajectory=result['trajectory'],
                cost=result['cost'],min_clearance=result['min_clearance'],min_semantic_gap=None if not math.isfinite(min_gap) else min_gap))
        return result
    def plan(self,*args,**kwargs):
        self.population=[];self.semantic_rejected=0
        result=super().plan(*args,**kwargs)
        result.update(objective_type='task_clearance' if self.semantic_target is not None else 'geometry',
            semantic_rejected_candidates=self.semantic_rejected,
            semantic_target=self.semantic_target,semantic_margin=self.desired_gap if self.semantic_target is not None else None,
            route_bank_used=False)
        return result

