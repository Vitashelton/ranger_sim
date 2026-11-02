"""Single-pose local-planner benchmark with measured arrival verification."""
import math
from .delivery_execution_recovery import DeliveryExecutionHarness

class LocalPoseNavigationHarness(DeliveryExecutionHarness):
    def __init__(self,graph,instruction,target,emit=None,**kwargs):
        super().__init__(graph,instruction,emit,**kwargs)
        self.target=tuple(target);self.stop_since=None
        if not graph.supports_pose(self.target,margin=.10):
            raise ValueError('Local benchmark target lacks footprint support')
        graph.nodes['H'+self.portal[1:]]['pose']=list(self.target)
        self._event('LOCAL_POSE_TASK_REGISTERED',0,target_pose=list(self.target))

    def subgoal(self):
        return self.target[:2],self.target[2]

    @property
    def anchor_id(self):
        return 'H'+self.portal[1:]

    def feedback(self,pose,stamp,tracks,executed_speed=0):
        error=abs(math.atan2(math.sin(pose[2]-self.target[2]),math.cos(pose[2]-self.target[2])))
        captured=math.dist(pose[:2],self.target[:2])<=.18 and error<=math.radians(10) and executed_speed<.025
        self.stop_since=(stamp if self.stop_since is None else self.stop_since) if captured else None
        if self.stop_since is not None and stamp-self.stop_since>=.35:
            self.stage='DONE'
            self.stage_events.append(self._event('LOCAL_POSE_ARRIVAL_VERIFIED',stamp,measured_pose=list(pose),target_pose=list(self.target)))
