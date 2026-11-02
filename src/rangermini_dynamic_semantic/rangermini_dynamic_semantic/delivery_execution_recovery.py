"""Bounded decision validity and footprint-aware portal capture for delivery.

This execution revision is opt-in. Frozen v6/v7 results use the original
GroundedNavigationHarness and must retain their original outcome labels.
"""
import math

from .grounded_navigation_harness import GroundedNavigationHarness


class DeliveryExecutionHarness(GroundedNavigationHarness):
    decision_lease_s = 4.0

    def __init__(self, *args, binding_recovery=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.binding_recovery = binding_recovery
        self.decision_deadline = None
        self.last_bound_stamp = None
        self.alignment_phase = "POSITION"

    @property
    def anchor_id(self):
        return self.portal if self.stage == "ALIGN" else super().anchor_id

    def subgoal(self):
        if self.stage == "ALIGN":
            door = self.graph.doors[self.portal]
            yaw = door["direction"]*math.pi/2 if self.alignment_phase == 'HEADING' else None
            return ((door["center_x"]+.01,
                     door["plane_y"]-door["direction"]*.85), yaw)
        return super().subgoal()

    def _release_binding(self, kind, stamp, **fields):
        previous = self.selected_behavior
        ids = list(self.evidence_ids)
        self.interaction = self.selected_behavior = "CONTINUE"
        self.evidence_ids = []
        self.bound_behavior_goal = None
        self.decision_deadline = None
        self.interaction_started = None
        self.query_epoch += 1
        self._event(kind, stamp, previous_behavior=previous, evidence_ids=ids,
                    **fields)

    def feedback(self, pose, stamp, tracks, executed_speed=0):
        old_stage = self.stage
        if self.stage == "APPROACH":
            goal, _ = self.subgoal()
            if math.dist(pose[:2], goal) <= .34:
                self.stage = "ALIGN"
                self.alignment_phase = "POSITION"
                self.stage_events.append(self._event(
                    "APPROACH_VERIFIED", stamp, measured_pose=list(pose)))
                self._release_binding("PORTAL_CAPTURE_STARTED", stamp)
        elif self.stage == "ALIGN":
            goal, _ = self.subgoal()
            door = self.graph.doors[self.portal]
            yaw = door["direction"]*math.pi/2
            error = abs(math.atan2(math.sin(pose[2]-yaw), math.cos(pose[2]-yaw)))
            half_opening = (door["aperture"][1]-door["aperture"][0])/2
            lateral_tolerance = min(.10, max(.015, half_opening-.25-.10))
            if (self.alignment_phase == 'POSITION' and
                    abs(pose[0]-goal[0]) <= lateral_tolerance and
                    abs(pose[1]-goal[1]) <= .10):
                self.alignment_phase = 'HEADING'
                self.query_epoch += 1
                self._event('PORTAL_CENTERED', stamp, measured_pose=list(pose))
            aligned = (abs(pose[0]-goal[0]) <= lateral_tolerance and
                       abs(pose[1]-goal[1]) <= .12 and
                       error <= math.radians(10) and
                       self.graph.supports_pose(pose))
            if aligned:
                self.stage = "CROSS"
                self.stage_events.append(self._event(
                    "PORTAL_ALIGNED", stamp, measured_pose=list(pose),
                    lateral_tolerance_m=lateral_tolerance))
                self._release_binding("PORTAL_CAPTURE_VERIFIED", stamp)

        # During ALIGN the base evaluator performs interaction/arrival logic
        # but has no approach-to-cross transition. Other stages keep the same
        # measured directional-crossing and stopped-arrival definitions.
        super().feedback(pose, stamp, tracks, executed_speed)
        if self.stage != old_stage and self.stage not in ("ALIGN", "CROSS"):
            self._release_binding("TASK_STAGE_BINDING_RESET", stamp)

        if self.selected_behavior == "CONTINUE":
            self.decision_deadline = None
            return
        if not self.binding_recovery:
            return
        live = {t.evidence_id for t in tracks if stamp-t.stamp <= .65}
        if self.evidence_ids and not any(e in live for e in self.evidence_ids):
            # Losing an observation revokes the binding. It is NOT evidence
            # of successful passage; the scan and rollout gate still apply.
            self._release_binding("BINDING_EVIDENCE_EXPIRED", stamp)
        elif (self.selected_behavior == "WAIT" and
              self.decision_deadline is not None and stamp >= self.decision_deadline):
            self._release_binding("WAIT_REASSESSMENT_REQUIRED", stamp,
                                  lease_s=self.decision_lease_s)

    def bind(self, proposal, candidates, tracks, stamp, query_stage):
        if self.stage in ("ALIGN", "CROSS") and proposal.get("behavior_id") != "CONTINUE":
            self._event("PROPOSAL_REJECTED", stamp, proposal=proposal,
                        reasons=["portal_capture_requires_anchor_tracking"])
            return False
        accepted = super().bind(proposal, candidates, tracks, stamp, query_stage)
        if accepted:
            self.last_bound_stamp = stamp
            self.decision_deadline = (stamp+self.decision_lease_s
                                      if self.selected_behavior == "WAIT" else None)
        return accepted

    def provisional(self, candidate, tracks, pose, stamp):
        previous, replacement = self.selected_behavior, candidate["behavior_id"]
        if previous == replacement:
            return
        # A steering/portal failure has no object cause unless current
        # encounter geometry supports one. Never reattach an already-passed
        # trolley merely because it remains somewhere in the scan.
        relevant = [t.evidence_id for t in tracks
                    if math.dist(t.position, pose[:2]) < 2.5 and
                    t.position[0] >= pose[0]-.75 and
                    abs(t.position[1]-pose[1]) <= 1.25]
        self._event("EXECUTION_FEASIBILITY_CHANGED", stamp,
                    previous_behavior=previous, provisional_behavior=replacement,
                    reason="active behavior has no supported current rollout",
                    relevant_evidence_ids=relevant)
        self.selected_behavior = self.interaction = replacement
        self.bound_behavior_goal = tuple(candidate["goal"])
        self.evidence_ids = relevant if replacement != "CONTINUE" else []
        self.interaction_started = stamp
        self.decision_deadline = (stamp+self.decision_lease_s
                                  if replacement == "WAIT" else None)
        self.query_epoch += 1

    def context(self, pose, tracks):
        result = super().context(pose, tracks)
        result["execution_contract"] = {
            "wait_lease_s": self.decision_lease_s,
            "wait_is_temporary": True,
            "portal_capture": self.stage in ("ALIGN", "CROSS"),
            "alignment_phase": self.alignment_phase if self.stage == 'ALIGN' else None,
            "on_expiry": "reobserve and replan; expiry does not verify passage",
        }
        return result
