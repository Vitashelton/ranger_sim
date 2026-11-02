"""Bounded mode-sequence/timing search around the existing 4WS MPPI rollout.

This is grid-based hybrid trajectory optimization, not an implementation of
H-PI's hybrid stochastic change-of-measure theorem. Old primitives remain
candidates. Each fixed structure AND boundary tuple has its own MPPI update;
no controls are averaged across different switching times or structures.
"""
import itertools
import math
import time
from dataclasses import replace
from .interaction_mppi import InteractionMPPI
from .ranger_4ws_kinematics import canonical_motion_mode
from .command_semantics_mppi import ranger_command


class HybridSequenceMPPI(InteractionMPPI):
    # Enumerate all adjacent-distinct structures up to two transitions; no
    # hand-selected winning maneuver is encoded for the office doorway.
    STRUCTURES=tuple(s for length in (2,3)
                     for s in itertools.product(('A','P','R'),repeat=length)
                     if all(a!=b for a,b in zip(s,s[1:])))

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._schedule=None
        self._schedule_step=0
        self.minimum_phase_steps=4
        self.samples_per_schedule=2
        self.preparation_cost_per_s=.5
        self.maximum_switches=2
        self.fixed_switch_timing=False
        self.rollout_budget=101
        self._evaluations=0

    def _evaluate(self,*args,**kwargs):
        if self._evaluations>=self.rollout_budget:return None
        self._evaluations+=1
        value=super()._evaluate(*args,**kwargs)
        if value is not None:
            preparation=sum(value['alignment_trace'])*self.cfg.dt
            value['cost']+=self.preparation_cost_per_s*preparation
            value['cost_terms']['steering_preparation']=self.preparation_cost_per_s*preparation
        return value

    def _reference(self,behavior,pose,goal,hypotheses,goal_yaw):
        if self.main_yaw is not None and (
                (self._schedule is None and behavior=='SPIN_THEN_DRIVE') or
                (self._schedule is not None and self._schedule[0][sum(self._schedule_step>=j for j in self._schedule[1])]=='R')):
            error=math.atan2(math.sin(self.main_yaw-pose[2]),math.cos(self.main_yaw-pose[2]))
            if self._schedule is not None:self._schedule_step+=1
            return 0.,0.,2.*error if abs(error)>self.obs.arrival_heading_tolerance*.5 else 0.,'SPIN'
        path=getattr(self,'reference_path',None)
        if path and self.interaction_behavior=='CONTINUE':
            # Both the primitive and hybrid populations use the same route
            # preview, so a sequence advantage is not a hidden reference change.
            nearest=min(range(len(path)),key=lambda i:math.dist(pose[:2],path[i][:2]))
            length=0.;index=nearest
            while index+1<len(path) and length<.45:
                length+=math.dist(path[index][:2],path[index+1][:2]);index+=1
            goal=path[index][:2]
            goal_yaw=self.main_yaw if index+1>=len(path) else None
        if self._schedule is None:
            return super()._reference(behavior,pose,goal,hypotheses,goal_yaw)
        structure,boundaries=self._schedule
        k=self._schedule_step;self._schedule_step+=1
        phase=sum(k>=j for j in boundaries);family=structure[phase]
        if family=='R':
            heading=goal_yaw if goal_yaw is not None else math.atan2(goal[1]-pose[1],goal[0]-pose[0])
            error=math.atan2(math.sin(heading-pose[2]),math.cos(heading-pose[2]))
            return 0.,0.,2.*error,'SPIN'
        # Translation includes both lateral and oblique body velocity; no
        # external label prescribes a lateral direction or switching instant.
        return super()._reference('ACKERMANN' if family=='A' else 'PARALLEL',
                                 pose,goal,hypotheses,goal_yaw)

    def _scheduled_evaluation(self,schedule,args,kwargs,sequence=None,nominal=False,reference_goal=None):
        self._schedule=schedule;self._schedule_step=0;self._nominal_sample=nominal
        self._sample_reference_goal=reference_goal
        try:
            # Parent evaluation retains shared route, task, footprint and
            # dynamic-obstacle costs and the same finite-rate wheel model.
            result=self._evaluate(*args,sequence=sequence,**kwargs)
        finally:
            self._schedule=None;self._nominal_sample=False;self._sample_reference_goal=None
        if result is None:return None
        structure,boundaries=schedule
        ends=(0,)+boundaries+(self.cfg.horizon,)
        executed=result['executed_controls']
        expected={'A':'DUAL_ACKERMANN','P':'PARALLEL','R':'SPINNING'}
        realized=[]
        for family,begin,end in zip(structure,ends,ends[1:]):
            # Command arbitration must preserve the proposed mode structure.
            # A nominal P phase that actually requests straight Ackermann
            # before its boundary is not a valid P->A switching-time sample.
            for u in result['controls'][begin:end]:
                requested=ranger_command(*u,min_turn_radius=self.cfg.min_turn_radius,
                                        wheelbase=self.cfg.wheelbase)[3]
                if requested not in (expected[family],'STOPPED'):return None
            active=any((abs(u[2])>.001 if family=='R' else math.hypot(u[0],u[1])>.001)
                       for u in executed[begin:end])
            realized.append(active)
        # A phase entirely consumed by steering preparation is not credited
        # as an executed maneuver. Do not silently delete/relabel that phase.
        if not all(realized):return None
        result['schedule']={'structure':list(structure),'switch_steps':list(boundaries),
                            'switch_times_s':[j*self.cfg.dt for j in boundaries]}
        result['alignment_s']=sum(result['alignment_trace'])*self.cfg.dt
        result['sampling_reference_goal']=reference_goal
        return result

    def plan(self,state,goal,scan=None,current_mode=None,goal_yaw=None,
             hypotheses=(),obstacle_points=None,current_steering_angles=None,
             current_steering_transition=False,reference_trajectory=None):
        begin=time.perf_counter()
        self._evaluations=0
        from .native_kernels import prepare_points
        obstacle_points=prepare_points(obstacle_points if obstacle_points is not None
                                       else self._scan_points_world(state,scan))
        cfg=self.cfg
        if self.maximum_switches==0:
            self.cfg=replace(cfg,samples_per_behavior=max(1,(self.rollout_budget-5)//4))
        try:
            result=super().plan(state,goal,scan,current_mode,goal_yaw,hypotheses,
                                obstacle_points,current_steering_angles,
                                current_steering_transition,reference_trajectory)
        finally:self.cfg=cfg
        result['sampling_profile']='bounded_hybrid_sequences_v1'
        result['primitive_reference_score']=result['predicted_score']
        result['primitive_reference_trajectory']=result.get('selected_trajectory',[])
        if self.maximum_switches==0 or self.interaction_behavior=='WAIT' or getattr(self,'capture_spin_only',False):
            result.update(rollout_evaluations=self._evaluations,rollout_budget=self.rollout_budget,
                          planning_time_ms=(time.perf_counter()-begin)*1000)
            return result
        points=obstacle_points
        mode=canonical_motion_mode(current_mode or result.get('initial_mode'))
        args=(state,goal,points,hypotheses,'SPIN_THEN_DRIVE',mode,goal_yaw)
        kwargs=dict(current_steering_angles=current_steering_angles,
                    current_steering_transition=current_steering_transition,
                    reference_trajectory=reference_trajectory)
        grid=sorted({self.cfg.horizon//4,self.cfg.horizon//2,3*self.cfg.horizon//4})
        schedules=[]
        for structure in self.STRUCTURES:
            if len(structure)-1>self.maximum_switches:continue
            options=([tuple(j*self.cfg.horizon//len(structure) for j in range(1,len(structure)))]
                     if self.fixed_switch_timing else itertools.combinations(grid,len(structure)-1))
            for boundaries in options:
                durations=[b-a for a,b in zip((0,)+boundaries,boundaries+(self.cfg.horizon,))]
                if min(durations)>=self.minimum_phase_steps:schedules.append((structure,boundaries))
        records={};evaluations=[];attempted=0;rejected=0;nominals={};reference_goals={}
        def optimize(schedule,refine=False):
            nonlocal attempted,rejected
            if self._evaluations>=self.rollout_budget:return
            samples=[]
            if refine and schedule in nominals:samples.append(nominals[schedule])
            attempted+=1
            if schedule not in reference_goals:
                references=[None]+list(getattr(self,'obstacle_reference_goals',()))
                reference_goals[schedule]=references[(attempted-1)%len(references)]
            sample=self._scheduled_evaluation(schedule,args,kwargs,nominal=not refine,
                                              reference_goal=reference_goals[schedule])
            if sample is None:rejected+=1
            else:samples.append(sample)
            if not samples:
                records.setdefault(schedule,dict(structure=list(schedule[0]),switch_steps=list(schedule[1]),
                    feasible=False,rejection='evaluated rollout infeasible or unrealized phase'))
                return
            best=min(samples,key=lambda sample:sample['cost'])
            if not refine:nominals[schedule]=best
            checked=best;fallback=False
            if refine and len(samples)>1 and self._evaluations<self.rollout_budget:
                minimum=best['cost']
                weights=[math.exp(-min(60.,(sample['cost']-minimum)/self.cfg.temperature)) for sample in samples]
                z=sum(weights)
                controls=[tuple(sum(w*sample['controls'][k][i] for sample,w in zip(samples,weights))/z
                                for i in range(3)) for k in range(self.cfg.horizon)]
                attempted+=1
                weighted=self._scheduled_evaluation(schedule,args,kwargs,sequence=controls)
                fallback=weighted is None or weighted['cost']>best['cost']
                if weighted is None:rejected+=1
                checked=best if fallback else weighted
            checked['weighted_action_fallback']=fallback;evaluations.append(checked)
            item=dict(**checked['schedule'],feasible=True,cost=checked['cost'],
                trajectory=checked['trajectory'],controls=checked['controls'],
                predicted_executed_controls=checked['executed_controls'],
                predicted_modes=checked['modes'],steering_trace=checked['steering_trace'],
                alignment_trace=checked['alignment_trace'],alignment_s=checked['alignment_s'],
                predicted_min_clearance_m=checked['min_clearance'],cost_terms=checked['cost_terms'],
                weighted_action_fallback=fallback)
            item['sampling_reference_goal']=checked.get('sampling_reference_goal')
            old=records.get(schedule)
            if old is None or not old.get('feasible') or item['cost']<old['cost']:records[schedule]=item
        # Nominal structure screening precedes stochastic MPPI updates. The
        # hard rollout cap is shared with the primitive-only ablation.
        for schedule in schedules:optimize(schedule)
        elites=sorted(nominals,key=lambda key:nominals[key]['cost'])[:4]
        for schedule in elites:optimize(schedule,refine=True)
        if evaluations and not self.fixed_switch_timing:
            seed=min(evaluations,key=lambda sample:sample['cost'])['schedule']
            bounds=tuple(seed['switch_steps']);structure=tuple(seed['structure'])
            for index in range(len(bounds)):
                for direction in (-1,1):
                    refined=list(bounds);refined[index]+=direction;refined=tuple(refined)
                    durations=[b-a for a,b in zip((0,)+refined,refined+(self.cfg.horizon,))]
                    if min(durations)>=self.minimum_phase_steps:optimize((structure,refined))
        # Reinvest remaining evaluations in the current elite rather than
        # enlarge the budget for the timing-search variant.
        while evaluations and self._evaluations<self.rollout_budget:
            before=self._evaluations
            elite=min(evaluations,key=lambda sample:sample['cost'])['schedule']
            optimize((tuple(elite['structure']),tuple(elite['switch_steps'])),refine=True)
            if self._evaluations==before:break
        if evaluations:
            selected=min(evaluations,key=lambda s:s['cost'])
            if result['predicted_score'] is None or selected['cost']<result['predicted_score']:
                action=selected['controls'][0]
                result.update(action=action,selected_behavior='HYBRID_SEQUENCE',
                    command_mode=selected['modes'][0],predicted_score=selected['cost'],
                    cost_terms=selected['cost_terms'],selected_trajectory=selected['trajectory'],
                    predicted_executed_controls=selected['executed_controls'],
                    terminal_steering_angles=selected['terminal_steering_angles'],
                    terminal_steering_transition=selected['terminal_steering_transition'],
                    terminal_mode=selected['terminal_mode'],
                    predicted_min_clearance_m=selected['min_clearance'],
                    selected_schedule=selected['schedule'],
                    predicted_steering_trace=selected['steering_trace'],
                    predicted_alignment_trace=selected['alignment_trace'])
                result['terminal_visibility']=selected['terminal_visibility']
                self.last_command_mode=result['command_mode']
        result.update(hybrid_schedule_candidates=list(records.values()),
                      rollout_evaluations=self._evaluations,rollout_budget=self.rollout_budget,
                      maximum_switches=self.maximum_switches,fixed_switch_timing=self.fixed_switch_timing,
                      candidate_count=result['candidate_count']+attempted,
                      rejected_candidates=result['rejected_candidates']+rejected,
                      planning_time_ms=(time.perf_counter()-begin)*1000)
        return result

    def note_execution(self,*args):
        # No fixed commitment clock: replan from measured mode and wheel state.
        pass
