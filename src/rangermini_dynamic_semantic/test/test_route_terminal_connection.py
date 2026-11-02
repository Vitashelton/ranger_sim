from pathlib import Path
import math
import unittest
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph
from rangermini_dynamic_semantic.footprint_route_guide import FootprintRouteGuide
ROOT=Path(__file__).resolve().parents[3]

class RouteTerminalConnectionTests(unittest.TestCase):
    def test_nearby_grid_point_must_connect_actual_pose_goal(self):
        world=ROOT/'artifacts/transition_target_pairs_20261027/task_pairs_v4/target2_hybrid_s37/world.sdf'
        if not world.exists():self.skipTest('diagnostic world unavailable')
        planner=FootprintRouteGuide(SpatialGraph(world));goal=(12.95,.55)
        _,path=planner.guide((12.80016,.38938,.32439),goal,target_yaw=math.pi/2)
        self.assertGreater(len(path),1)
        self.assertLess(math.dist(path[-1],goal),1e-6)
        self.assertLess(abs(math.atan2(math.sin(planner.pose_path[-1][2]-math.pi/2),math.cos(planner.pose_path[-1][2]-math.pi/2))),math.pi/24+1e-8)
        for first,last in zip(planner.pose_path,planner.pose_path[1:]):
            if math.dist(first[:2],last[:2])<1e-8:
                delta=math.atan2(math.sin(last[2]-first[2]),math.cos(last[2]-first[2]))
                self.assertTrue(all(planner.free([first[:2]],first[2]+delta*u)[0] for u in (0,.25,.5,.75,1)))

if __name__=='__main__':unittest.main()
