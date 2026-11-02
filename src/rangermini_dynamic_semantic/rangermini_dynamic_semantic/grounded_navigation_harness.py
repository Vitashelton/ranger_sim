"""Task/spatial/evidence binding for the furnished delivery simulator.

ROS-independent. Geometry is read from the registered static map; moving
object state is supplied by sensor tracking, never by the scene fixture.
"""
from collections import deque
from dataclasses import dataclass, asdict
import math
import re
import xml.etree.ElementTree as ET

import numpy as np


BEHAVIORS = ("CONTINUE", "PASS_LEFT", "PASS_RIGHT", "YIELD_LEFT", "YIELD_RIGHT", "WAIT")


@dataclass
class Track:
    evidence_id: str
    position: tuple
    velocity: tuple
    radius: float
    stamp: float
    first_seen: float
    observations: int = 1
    category: str = "obstacle"
    source: str = "map_filtered_lidar_cluster"
    registered_box: tuple = None

    def to_dict(self):
        return asdict(self)


class SpatialGraph:
    def __init__(self, world_file):
        world = ET.parse(world_file).getroot().find("world")
        self.nodes = {"corridor": {"kind": "region", "bounds": [0, 18, -3.81, -.39]}}
        self.edges = []
        self.static_boxes = []
        self.doors = {}
        for model in world.findall("model"):
            name = model.get("name", "")
            pose = list(map(float, model.findtext("pose", "0 0 0 0 0 0").split()))
            if re.fullmatch(r"door_\d+_lintel", name):
                room = name.split("_")[1]
                jambs = [world.find(f"model[@name='door_{room}_jamb_{s}']") for s in ("-0.46", "0.46")]
                if any(j is None for j in jambs):
                    continue
                xs = [float(j.findtext("pose").split()[0]) for j in jambs]
                center = sum(xs)/2
                # The open leaf is on the low-x side; center the footprint in
                # the remaining opening rather than assigning a language pose.
                portal = f"D{room}"
                direction = 1 if pose[1] > -2 else -1
                self.doors[portal] = {"center_x": center, "plane_y": pose[1],
                    "aperture": [min(xs)+.035, max(xs)-.035], "direction": direction}
                self.nodes[portal] = {"kind": "portal", **self.doors[portal], "source": "registered_world_geometry"}
                self.nodes[f"R{room}"] = {"kind": "room", "portal": portal}
                self.nodes[f"H{room}"] = {"kind": "arrival_region", "room": f"R{room}",
                    "pose": [center+.01, pose[1]+direction*1.45, direction*math.pi/2]}
                self.edges.extend([["corridor", portal], [portal, f"R{room}"], [f"R{room}", f"H{room}"]])
            if name == "rangermini_2_0" or name == "lab_student" or name.startswith(("service_trolley", "trolley_")):
                continue
            for link in model.findall("link"):
                lp = list(map(float, link.findtext("pose", "0 0 0 0 0 0").split()))
                for collision in link.findall("collision"):
                    size = collision.findtext("geometry/box/size")
                    if not size:
                        continue
                    cp = list(map(float, collision.findtext("pose", "0 0 0 0 0 0").split()))
                    sx, sy, sz = map(float, size.split())
                    z = pose[2]+lp[2]+cp[2]
                    # The chassis can hit low cabinet plinths that lie below
                    # the horizontal laser. Grounding uses its swept height,
                    # not only a single lidar-height slice of the map.
                    if z+sz/2 < .025 or z-sz/2 > .50:
                        continue
                    c, s = math.cos(pose[5]), math.sin(pose[5])
                    dx, dy = lp[0]+cp[0], lp[1]+cp[1]
                    self.static_boxes.append((pose[0]+c*dx-s*dy, pose[1]+s*dx+c*dy,
                        pose[5]+lp[5]+cp[5], sx/2, sy/2))
        self.revision = 0
        self.observations = {}

    def static_mask(self, points, margin=.13):
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        mask = np.zeros(len(pts), dtype=bool)
        for x, y, yaw, hx, hy in self.static_boxes:
            dx, dy = pts[:, 0]-x, pts[:, 1]-y
            c, s = math.cos(yaw), math.sin(yaw)
            mask |= (np.abs(c*dx+s*dy) <= hx+margin) & (np.abs(-s*dx+c*dy) <= hy+margin)
        return mask

    def update(self, tracks):
        current = {t.evidence_id: t.to_dict() for t in tracks}
        if set(current) != set(self.observations):
            self.revision += 1
        self.observations = current

    def supports_pose(self, pose, length=.751, width=.5, margin=.035):
        x, y, yaw = pose
        robot_axes = [(math.cos(yaw), math.sin(yaw)), (-math.sin(yaw), math.cos(yaw))]
        for cx, cy, angle, hx, hy in self.static_boxes:
            object_axes = [(math.cos(angle), math.sin(angle)), (-math.sin(angle), math.cos(angle))]
            separation = -float("inf")
            for ax, ay in robot_axes+object_axes:
                radius_a = length/2*abs(ax*robot_axes[0][0]+ay*robot_axes[0][1])+width/2*abs(ax*robot_axes[1][0]+ay*robot_axes[1][1])
                radius_b = hx*abs(ax*object_axes[0][0]+ay*object_axes[0][1])+hy*abs(ax*object_axes[1][0]+ay*object_axes[1][1])
                separation = max(separation, abs((cx-x)*ax+(cy-y)*ay)-radius_a-radius_b)
            if separation < margin:
                return False
        return True

    def remaining_connected(self, portal):
        room = "R"+portal[1:]
        return ["corridor", portal] in self.edges and [portal, room] in self.edges and [room, "H"+portal[1:]] in self.edges

    def snapshot(self):
        return {"nodes": self.nodes, "edges": self.edges, "revision": self.revision,
                "observed_objects": self.observations}


class LidarEvidenceTracker:
    def __init__(self, graph):
        self.graph = graph
        self.tracks = {}
        self.histories = {}
        self.next_id = 1

    def update(self, pose, scan, stamp):
        ranges, a0, da = scan
        points = []
        x, y, yaw = pose
        for i, value in enumerate(ranges):
            r = float(value)
            if math.isfinite(r) and .15 < r < 8:
                a = yaw+a0+i*da
                points.append((x+.30*math.cos(yaw)+r*math.cos(a),
                               y+.30*math.sin(yaw)+r*math.sin(a)))
        if not points:
            return self.active(stamp)
        pts = np.asarray(points)
        pts = pts[~self.graph.static_mask(pts)]
        # The interaction tracker covers the shared corridor. Returns from
        # furniture inside adjoining rooms remain ordinary scan obstacles,
        # rather than becoming alleged moving corridor agents.
        pts = pts[(pts[:, 0] > 1) & (pts[:, 0] < 17.5) &
                  (pts[:, 1] > -3.65) & (pts[:, 1] < -.55)]
        clusters = []
        if len(pts):
            # Preserve scan order; separate objects across a range discontinuity.
            splits = np.where(np.linalg.norm(np.diff(pts, axis=0), axis=1) > .24)[0]+1
            pieces = [p for p in np.split(pts, splits) if len(p) >= 2]
            for piece in pieces:
                center = piece.mean(axis=0)
                extent = float(np.linalg.norm(piece.max(axis=0)-piece.min(axis=0)))
                if extent > 1.35:
                    continue
                for index, (other, data) in enumerate(clusters):
                    if np.linalg.norm(center-other) < .70:
                        merged = np.vstack([piece, data])
                        clusters[index] = (merged.mean(axis=0), merged)
                        break
                else:
                    clusters.append((center, piece))
        used = set()
        for center, piece in clusters:
            options = []
            for tid, track in self.tracks.items():
                if tid in used or stamp-track.stamp > 1.5:
                    continue
                dt = stamp-track.stamp
                predicted = np.asarray(track.position)+dt*np.asarray(track.velocity)
                options.append((float(np.linalg.norm(center-predicted)), tid))
            best = min(options, default=(float("inf"), None))
            if best[0] < .75:
                tid = best[1]
                old = self.tracks[tid]
            else:
                tid = f"track_{self.next_id}"
                self.next_id += 1
                old = None
                self.histories[tid] = deque(maxlen=15)
            used.add(tid)
            history = self.histories[tid]
            if old is not None:
                elapsed = max(.001, stamp-old.stamp)
                predicted = np.asarray(old.position)+elapsed*np.asarray(old.velocity)
                alpha = elapsed/(.35+elapsed)
                center = predicted+alpha*(center-predicted)
            history.append((stamp, float(center[0]), float(center[1])))
            velocity = np.zeros(2)
            if len(history) >= 3 and history[-1][0]-history[0][0] >= .25:
                values = np.asarray(history)
                # Robust temporal slopes avoid interpreting a changing visible
                # face of the trolley as sudden lateral motion.
                slopes = [(values[j, 1:]-values[i, 1:])/(values[j, 0]-values[i, 0])
                    for i in range(len(values)) for j in range(i+1, len(values))
                    if values[j, 0]-values[i, 0] >= .25]
                if slopes:
                    velocity = np.median(slopes, axis=0)
                speed = np.linalg.norm(velocity)
                if speed > .7:
                    velocity *= .7/speed
                if speed < .04:
                    velocity[:] = 0
            extent = float(np.linalg.norm(piece.max(axis=0)-piece.min(axis=0)))
            # Sparse far-range returns must not shrink an unknown cart to a
            # single thin post. Once visually classified, person and trolley
            # use the registered category proxy sizes.
            category = old.category if old else "obstacle"
            radius = max(.20 if category == "person" else .42, min(.48, extent/2+.16))
            if category == "trolley":
                radius = max(radius, .46)
            self.tracks[tid] = Track(tid, tuple(map(float, center)), tuple(map(float, velocity)),
                radius, stamp, old.first_seen if old else stamp,
                (old.observations+1) if old else 1, category)
        for tid in list(self.tracks):
            if stamp-self.tracks[tid].stamp > 2:
                del self.tracks[tid]
                self.histories.pop(tid, None)
        tracks = self.active(stamp)
        self.graph.update(tracks)
        return tracks

    def active(self, stamp):
        return [t for t in self.tracks.values() if stamp-t.stamp <= .65 and t.observations >= 2]

    def label(self, labels):
        for tid, kind in labels.items():
            if tid in self.tracks and kind in ("person", "trolley", "obstacle"):
                self.tracks[tid].category = kind


class GroundedNavigationHarness:
    def __init__(self, graph, instruction, emit=None, destination_room=None):
        self.graph = graph
        aliases = {"619": ("office a", "619", "办公室a", "a办公室"),
                   "617": ("office c", "617", "办公室c", "c办公室")}
        matches = ([str(destination_room)] if destination_room is not None else
                   [room for room, words in aliases.items() if any(w in instruction.lower() for w in words)])
        if len(matches) != 1 or "D"+matches[0] not in graph.doors:
            raise ValueError("instruction must resolve to one registered destination")
        self.instruction = instruction
        self.portal = "D"+matches[0]
        self.destination = "R"+matches[0]
        self.stage = "APPROACH"
        self.interaction = "CONTINUE"
        self.memory = deque(maxlen=40)
        self.emit = emit or (lambda event: None)
        self.previous_pose = None
        self.selected_behavior = "CONTINUE"
        self.evidence_ids = []
        self.bound_behavior_goal = None
        self.interaction_started = None
        self.arrival_since = None
        self.query_epoch = 0
        self.stage_events = []
        self._event("TASK_GROUNDED", 0, destination=self.destination, portal=self.portal,
                    source="registered_alias_and_static_graph")

    def _event(self, kind, stamp, **fields):
        event = {"event": kind, "sim_time": stamp, "stage": self.stage,
                 "interaction": self.interaction, **fields}
        self.memory.append(event)
        self.emit(event)
        return event

    def subgoal(self):
        door = self.graph.doors[self.portal]
        x, y, d = door["center_x"]+.01, door["plane_y"], door["direction"]
        if self.stage == "APPROACH":
            return (x, y-d*.85), None
        if self.stage == "CROSS":
            return (x, y+d*1.00), d*math.pi/2
        arrival = self.graph.nodes["H"+self.portal[1:]]["pose"]
        return tuple(arrival[:2]), arrival[2]

    @property
    def anchor_id(self):
        return self.portal if self.stage in ("APPROACH", "CROSS") else "H"+self.portal[1:]

    def feedback(self, pose, stamp, tracks, executed_speed=0):
        goal, _ = self.subgoal()
        door = self.graph.doors[self.portal]
        d = door["direction"]
        if self.stage == "APPROACH" and math.dist(pose[:2], goal) <= .34:
            self.stage = "CROSS"
            self.stage_events.append(self._event("APPROACH_VERIFIED", stamp, measured_pose=list(pose)))
            self.query_epoch += 1
        elif self.stage == "CROSS" and self.previous_pose:
            before = d*(self.previous_pose[1]-door["plane_y"])
            after = d*(pose[1]-door["plane_y"])
            if before <= 0 < after:
                fraction = -before/max(1e-8, after-before)
                crossing_x = self.previous_pose[0]+fraction*(pose[0]-self.previous_pose[0])
                if door["aperture"][0] <= crossing_x <= door["aperture"][1]:
                    self.stage = "ARRIVE"
                    self.stage_events.append(self._event("PORTAL_CROSSED", stamp,
                        measured_pose=list(pose), crossing_x=crossing_x, anchor_id=self.portal))
                    self.query_epoch += 1
        if self.stage == "ARRIVE":
            goal, yaw = self.subgoal()
            error = abs(math.atan2(math.sin(pose[2]-yaw), math.cos(pose[2]-yaw)))
            if math.dist(pose[:2], goal) <= .32 and error <= math.radians(18):
                if executed_speed < .025:
                    self.arrival_since = stamp if self.arrival_since is None else self.arrival_since
                    if stamp-self.arrival_since >= .35:
                        self.stage = "DONE"
                        self.stage_events.append(self._event("ARRIVAL_VERIFIED", stamp, measured_pose=list(pose)))
                else:
                    self.arrival_since = None
            else:
                self.arrival_since = None
        self.previous_pose = tuple(pose)
        if self.interaction != "CONTINUE":
            by_id = {t.evidence_id: t for t in tracks}
            relevant = [by_id[eid] for eid in self.evidence_ids if eid in by_id]
            # Release on observed passage/divergence. Missing evidence alone
            # is not considered successful clearance of an encounter.
            observed_clear = self.observed_interaction_clear(pose, relevant)
            if observed_clear:
                self._event("INTERACTION_CLEARED", stamp, evidence_ids=self.evidence_ids)
                self.interaction = self.selected_behavior = "CONTINUE"
                self.evidence_ids = []
                self.bound_behavior_goal = None
                self.query_epoch += 1

    def observed_interaction_clear(self, pose, relevant):
        return bool(relevant) and all(
            t.position[0] < pose[0]-.75 or abs(t.position[1]-pose[1]) > 1.25
            for t in relevant)

    def bind(self, proposal, candidates, tracks, stamp, query_stage):
        valid_ids = {t.evidence_id for t in tracks if stamp-t.stamp <= .65}
        errors = []
        if query_stage != self.stage or proposal.get("anchor_id") != self.anchor_id:
            errors.append("task_stage_or_anchor_mismatch")
        behavior = proposal.get("behavior_id")
        available = {c["behavior_id"] for c in candidates if c.get("feasible")}
        if behavior not in available:
            errors.append("no_current_feasible_rollout_for_behavior")
        ids = proposal.get("evidence_ids")
        if not isinstance(ids, list) or any(e not in valid_ids for e in ids):
            errors.append("missing_or_stale_evidence")
        elif behavior != "CONTINUE" and not ids:
            errors.append("interaction_requires_object_evidence")
        if not self.graph.remaining_connected(self.portal):
            errors.append("remaining_task_disconnected")
        if errors:
            self._event("PROPOSAL_REJECTED", stamp, proposal=proposal, reasons=errors)
            return False
        self.selected_behavior = behavior
        self.evidence_ids = ids
        self.interaction = behavior
        self.interaction_started = stamp
        selected = next(c for c in candidates if c["behavior_id"] == behavior and c["feasible"])
        self.bound_behavior_goal = tuple(selected["goal"])
        self._event("PROPOSAL_BOUND", stamp, proposal=proposal, bound_goal=selected["goal"],
            motion_behavior=selected["plan"]["selected_behavior"],
            cost_terms=selected["plan"].get("cost_terms"), evidence_ids=ids)
        return True

    def context(self, pose, tracks):
        goal, yaw = self.subgoal()
        return {"instruction": self.instruction, "destination": self.destination,
                "task_stage": self.stage, "anchor_id": self.anchor_id,
                "grounded_goal": goal, "goal_yaw": yaw, "robot_pose": pose,
                "interaction": self.interaction, "space": self.graph.snapshot(),
                "objects": [t.to_dict() for t in tracks], "recent_events": list(self.memory)[-8:]}
