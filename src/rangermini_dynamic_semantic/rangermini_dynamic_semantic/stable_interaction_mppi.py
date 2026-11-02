"""Preserve a sampled motion family while finite-rate steering is executed.

All compared selectors use this same execution-side continuity mechanism.
It changes neither geometric nor semantic safety thresholds.
"""
import math
from .interaction_mppi import InteractionMPPI
from .ranger_4ws_kinematics import canonical_motion_mode


class StableInteractionMPPI(InteractionMPPI):
    def __init__(self, config, prediction=True, static_boxes=None):
        super().__init__(config, prediction, static_boxes)
        self.committed_group = None
        self.commit_until = -1
        self.motion_group_gate = None

    def _evaluate(self, *args, **kwargs):
        group = args[4] if len(args) > 4 else kwargs.get('behavior')
        if self.motion_group_gate and group not in (self.motion_group_gate, 'STOP'):
            return None
        return super()._evaluate(*args, **kwargs)

    def behavior_plan(self, state, main_goal, scan, tracks, stamp, behavior, **kwargs):
        self.committed_group = canonical_motion_mode(self.committed_group)
        self.motion_group_gate = self.committed_group if stamp < self.commit_until else None
        return super().behavior_plan(state, main_goal, scan, tracks, stamp, behavior, **kwargs)

    def bank(self, state, main_goal, scan, tracks, stamp, **kwargs):
        result = super().bank(state, main_goal, scan, tracks, stamp, **kwargs)
        if self.motion_group_gate and not any(c['feasible'] and c['plan']['selected_behavior'] != 'STOP' for c in result):
            # A commitment is never allowed to suppress every feasible escape.
            self.commit_until = -1
            self.motion_group_gate = None
            result = super().bank(state, main_goal, scan, tracks, stamp, **kwargs)
        return result

    def note_execution(self, plan, stamp, action):
        group = canonical_motion_mode(plan['selected_behavior'])
        if group != 'STOP' and (math.hypot(action[0], action[1]) > .001 or abs(action[2]) > .001):
            if group != self.committed_group:
                self.committed_group = group
                self.commit_until = stamp+1.2
