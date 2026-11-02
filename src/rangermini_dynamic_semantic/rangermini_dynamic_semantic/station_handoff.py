"""Opt-in receiving-station conditions; independent of legacy delivery trials.

The station is registered map geometry, not an alleged visual detection.
The selector chooses the service relation; geometry supplies its realization.
"""
import math
import xml.etree.ElementTree as ET
import numpy as np

from .delivery_execution_recovery import DeliveryExecutionHarness


def angle_error(a, b):
    return abs(math.atan2(math.sin(a-b), math.cos(a-b)))


def station_conditions(graph, world_file):
    world = ET.parse(world_file).getroot().find('world')
    desk = world.find("model[@name='office_desk_619']")
    if desk is None:
        raise ValueError('registered receiving desk missing')
    p = list(map(float, desk.findtext('pose').split()))
    door = graph.doors['D619']
    # Receiving pose lies beside the desk, east of the chair, rather than on
    # the doorway axis. The same registered pose is used by both controllers.
    target = (door['center_x']+.16, p[1])
    # Prepare before the receiving position along the doorway-to-receiver
    # direction. A staging reference must not force an overshoot and return.
    staging = (door['center_x']+.01, p[1]-.95)
    bearing = math.atan2(p[1]-target[1], p[0]-target[0])
    choices = {}
    for name, offset, relation in [
            ('HANDOFF_FRONT', 0., 'at the receiving pose, the robot front points toward the desk'),
            ('HANDOFF_LEFT', -math.pi/2, 'receiving desk is on the robot left')]:
        yaw = math.atan2(math.sin(bearing+offset), math.cos(bearing+offset))
        condition = dict(behavior_id=name, grounding_kind='registered_station',
            station_id='DESK619', relation=relation, target_pose=[*target, yaw],
            staging_pose=[*staging, yaw], heading_tolerance_rad=math.radians(10),
            position_tolerance_m=.10,
            approach_constraint='retain body heading from staging to receiving pose')
        if not all(graph.supports_pose((*q, yaw), margin=.10) for q in (staging, target)):
            raise ValueError('station condition lacks footprint support: '+name)
        choices[name] = condition
    graph.nodes['DESK619'] = dict(kind='receiving_station', position=p[:2],
        source='registered_world_geometry', model='office_desk_619')
    return choices


class StationHandoffHarness(DeliveryExecutionHarness):
    """Door capture followed by selector-conditioned, heading-retaining docking."""
    def __init__(self, graph, instruction, world_file, emit=None, terminal_pose_only=False, **kwargs):
        super().__init__(graph, instruction, emit, **kwargs)
        if self.portal != 'D619':
            raise ValueError('receiving-station profile currently registers Office 619')
        self.station_choices = station_conditions(graph, world_file)
        self.motion_condition = None
        self.terminal_pose_only = terminal_pose_only
        self.staging = tuple(next(iter(self.station_choices.values()))['staging_pose'][:2])
        self.staging_observed = False
        self.alignment_pose_supported = False
        self.last_executed_speed = float('inf')
        self.station_stop_since = None

    @property
    def ready_for_station_selection(self):
        return self.stage == 'HANDOFF_SELECT' and self.station_stop_since is not None

    def refresh_staging_from_scan(self, pose, points, stamp):
        """Support the complete alignment sweep using observed furniture returns.

        Mesh furniture is not fully represented by the registered box map.
        A circumscribed footprint plus the position capture tolerance supports
        every intermediate heading, without querying simulator actor truth.
        """
        if self.stage not in ('STATION_APPROACH','HANDOFF_ALIGN') or pose[1] < .65:
            return
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        if not len(pts):
            return
        radius = math.hypot(.751/2, .500/2)
        self.alignment_pose_supported = (
            float(np.linalg.norm(pts-np.asarray(pose[:2]),axis=1).min())-radius >= .035
            and all(self.graph.supports_pose((*pose[:2],k*math.pi/18),margin=.10)
                    for k in range(36)))
        if self.stage == 'HANDOFF_ALIGN':
            if self.alignment_pose_supported:
                return
            # Newly exposed mesh returns can invalidate an earlier alignment
            # region. Retain the receiving relation and repair its spatial
            # support instead of repeatedly sampling an impossible rotation.
            self.stage = 'STATION_APPROACH'
            self.staging_observed = False
            self.query_epoch += 1
            self._event('STATION_ALIGNMENT_SUPPORT_CHANGED',stamp,
                        measured_pose=list(pose),retained_condition=self.motion_condition)
        if self.staging_observed:
            return
        options = []
        for i in range(-6, 7):
            for j in range(-16, 7):
                q = (self.staging[0]+i*.025, self.staging[1]+j*.025)
                if q[1] > min(c['target_pose'][1] for c in self.station_choices.values())-.35:
                    continue
                gap = float(np.linalg.norm(pts-np.asarray(q), axis=1).min())-radius
                if gap < .035+.04:
                    continue
                if not all(self.graph.supports_pose((*q,k*math.pi/18),margin=.10) for k in range(36)):
                    continue
                if not all(self.graph.supports_pose((
                        q[0]+a*(c['target_pose'][0]-q[0]),
                        q[1]+a*(c['target_pose'][1]-q[1]),c['target_pose'][2]),margin=.10)
                        for c in self.station_choices.values() for a in np.linspace(0,1,21)):
                    continue
                options.append((math.dist(q,self.staging),-gap,q,gap))
        if not options:
            return
        _,_,q,gap = min(options)
        previous = self.staging
        self.staging = q
        for condition in self.station_choices.values():
            condition['staging_pose'][:2] = q
        self.staging_observed = True
        self._event('STATION_ALIGNMENT_SUPPORT_VERIFIED',stamp,
            original_staging=list(previous),observed_staging=list(q),
            observed_rotation_gap_m=gap,source='current laser returns and registered static map')

    @property
    def anchor_id(self):
        if self.stage.startswith(('STATION_', 'HANDOFF')):
            return 'DESK619'
        return super().anchor_id

    def subgoal(self):
        if self.stage in ('STATION_APPROACH', 'HANDOFF_SELECT'):
            return self.staging, math.pi/2
        if self.stage == 'HANDOFF_ALIGN':
            return self.staging, self.motion_condition['target_pose'][2]
        if self.stage == 'HANDOFF_APPROACH' or (self.stage == 'DONE' and self.motion_condition):
            q = self.motion_condition['target_pose']
            return tuple(q[:2]), q[2]
        return super().subgoal()

    def feedback(self, pose, stamp, tracks, executed_speed=0):
        self.last_executed_speed = executed_speed
        if self.stage == 'HANDOFF_SELECT' and executed_speed < .025:
            self.station_stop_since = stamp if self.station_stop_since is None else self.station_stop_since
        else:
            self.station_stop_since = None
        if not self.stage.startswith(('STATION_', 'HANDOFF')):
            super().feedback(pose, stamp, tracks, executed_speed)
            if self.stage == 'ARRIVE':
                self.stage = 'STATION_APPROACH'
                self._release_binding('RECEIVING_STATION_APPROACH', stamp)
            return
        if (self.stage == 'STATION_APPROACH' and self.alignment_pose_supported
                and math.dist(pose[:2], self.staging) <= .12):
            self.stage = 'HANDOFF_ALIGN' if self.motion_condition else 'HANDOFF_SELECT'
            self.query_epoch += 1
            self._event('STATION_OBSERVED', stamp, measured_pose=list(pose))
        elif self.stage == 'HANDOFF_ALIGN':
            c = self.motion_condition
            if (math.dist(pose[:2], self.staging) <= .12 and
                    angle_error(pose[2], c['target_pose'][2]) <= c['heading_tolerance_rad']):
                self.stage = 'HANDOFF_APPROACH'
                self._event('HANDOFF_HEADING_ESTABLISHED', stamp, measured_pose=list(pose),
                            motion_condition=c)
        elif self.stage == 'HANDOFF_APPROACH':
            c = self.motion_condition
            close = (math.dist(pose[:2], c['target_pose'][:2]) <= c['position_tolerance_m']
                     and angle_error(pose[2], c['target_pose'][2]) <= c['heading_tolerance_rad'])
            if close and executed_speed < .025:
                self.arrival_since = stamp if self.arrival_since is None else self.arrival_since
                if stamp-self.arrival_since >= .35:
                    self.stage = 'DONE'
                    self.stage_events.append(self._event('HANDOFF_POSE_VERIFIED', stamp,
                        measured_pose=list(pose), motion_condition=c))
            else:
                self.arrival_since = None
        self.previous_pose = tuple(pose)

    def bind(self, proposal, candidates, tracks, stamp, query_stage):
        if self.stage != 'HANDOFF_SELECT':
            return super().bind(proposal, candidates, tracks, stamp, query_stage)
        if (self.station_stop_since is None or stamp-self.station_stop_since < .35
                or self.last_executed_speed >= .025):
            self._event('PROPOSAL_REJECTED',stamp,proposal=proposal,
                        reasons=['station_transition_requires_verified_stop'])
            return False
        name = proposal.get('behavior_id')
        valid = (query_stage == self.stage and proposal.get('anchor_id') == 'DESK619'
                 and name in self.station_choices and not proposal.get('evidence_ids')
                 and any(c['behavior_id'] == name and c['feasible'] for c in candidates))
        if not valid:
            self._event('PROPOSAL_REJECTED', stamp, proposal=proposal,
                        reasons=['station_relation_or_current_rollout_invalid'])
            return False
        self.motion_condition = dict(self.station_choices[name])
        self.graph.nodes['H619']['pose'] = self.motion_condition['target_pose']
        self.stage = 'HANDOFF_APPROACH' if self.terminal_pose_only else 'HANDOFF_ALIGN'
        self.selected_behavior = self.interaction = 'CONTINUE'
        self.bound_behavior_goal = None
        self._event('STATION_CONDITION_BOUND', stamp, proposal=proposal,
                    motion_condition=self.motion_condition)
        return True

    def provisional(self, candidate, tracks, pose, stamp):
        if self.stage.startswith(('STATION_', 'HANDOFF')):
            # An infeasible sample is not permission to discard the service relation.
            return
        return super().provisional(candidate, tracks, pose, stamp)

    def context(self, pose, tracks):
        result = super().context(pose, tracks)
        result['station_choices'] = self.station_choices if self.stage == 'HANDOFF_SELECT' else {}
        result['active_motion_condition'] = self.motion_condition
        return result
