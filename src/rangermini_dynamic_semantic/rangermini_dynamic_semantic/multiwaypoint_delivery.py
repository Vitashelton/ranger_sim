"""Registered multi-stop task feeding the existing planner, never wheel commands.

Collection is a stopped service-pose acknowledgement in simulation. It is not
an observed physical loading operation or an autonomous language task parser.
"""
import math
from .delivery_execution_recovery import DeliveryExecutionHarness


class MultiwaypointDeliveryHarness(DeliveryExecutionHarness):
    def __init__(self, graph, instruction, emit=None, return_to_collection=False, **kwargs):
        super().__init__(graph, instruction, emit, **kwargs)
        self.stops = [
            dict(id='COLLECTION', pose=(4.4, -2.1, -math.pi/2), hold=.5),
            dict(id='DEPARTURE', pose=(4.4, -2.1, 0.), hold=.35),
            dict(id='CORRIDOR', pose=(6., -2.1, 0.), hold=0.),
        ]
        if return_to_collection:
            self.stops.insert(0, dict(id='TURNAROUND', pose=(6.,-2.1,math.pi), hold=.35))
        for stop in self.stops:
            if not graph.supports_pose(stop['pose'], margin=.10):
                raise ValueError('unsupported registered service stop: '+stop['id'])
        self.stop_index = 0
        self.stop_since = None
        self.stage = 'SERVICE_'+self.stops[0]['id']
        self.collection_acknowledged = False

    @property
    def service_active(self):
        return self.stop_index < len(self.stops)

    @property
    def anchor_id(self):
        return self.stops[self.stop_index]['id'] if self.service_active else super().anchor_id

    def subgoal(self):
        if self.service_active:
            pose = self.stops[self.stop_index]['pose']
            return pose[:2], pose[2]
        return super().subgoal()

    def at_service_pose(self, pose):
        if not self.service_active:
            return False
        if self.stops[self.stop_index]['id'] == 'CORRIDOR':
            return False
        target = self.stops[self.stop_index]['pose']
        yaw_error = abs(math.atan2(math.sin(pose[2]-target[2]), math.cos(pose[2]-target[2])))
        return math.dist(pose[:2], target[:2]) <= .20 and yaw_error <= math.radians(10)

    def feedback(self, pose, stamp, tracks, executed_speed=0):
        if not self.service_active:
            return super().feedback(pose, stamp, tracks, executed_speed)
        # Retain the existing interaction release and WAIT lease while a
        # service anchor is active; service progression is checked below.
        super().feedback(pose, stamp, tracks, executed_speed)
        stop = self.stops[self.stop_index]
        transit = stop['id'] == 'CORRIDOR'
        if transit:
            # A transit waypoint is a region to pass, not a service pose that
            # requires stopping and returning after a lateral avoidance.
            if pose[0] < stop['pose'][0]-.24 or abs(pose[1]-stop['pose'][1]) > 1.1:
                return
        else:
            if not self.at_service_pose(pose) or executed_speed >= .025:
                self.stop_since = None
                return
            self.stop_since = stamp if self.stop_since is None else self.stop_since
            if stamp-self.stop_since < stop['hold']:
                return
        self.collection_acknowledged |= stop['id'] == 'COLLECTION'
        event = self._event('TRANSIT_REGION_VERIFIED' if transit else 'SERVICE_STOP_VERIFIED',
            stamp, stop_id=stop['id'],
            measured_pose=list(pose), target_pose=list(stop['pose']),
            collection='stationary simulation acknowledgement; no loading actuator')
        self.stage_events.append(event)
        self.stop_index += 1
        self.stop_since = None
        self.stage = ('SERVICE_'+self.stops[self.stop_index]['id']
                      if self.service_active else 'APPROACH')
        self.previous_pose = tuple(pose)
        self._release_binding('SERVICE_TARGET_ADVANCED', stamp)

    def context(self, pose, tracks):
        result = super().context(pose, tracks)
        result['service_task'] = dict(collection_acknowledged=self.collection_acknowledged,
            current_stop=self.anchor_id, remaining_stops=[s['id'] for s in self.stops[self.stop_index:]],
            destination=self.destination, departure_preference='forward travel; no reverse retreat')
        return result
