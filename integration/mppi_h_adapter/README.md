# MPPI-H → Ranger Gazebo

Upstream: https://github.com/MizuhoAOKI/mppi_swerve_drive_ros
Commit: `5b53b4b2cf06fe37a31095d6c12cf695075d7705` (MIT; see UPSTREAM_LICENSE).

`build.py` compiles the author's three solver `.cpp` files without edits. The
ROS-independent C ABI is `bridge.cpp`; `compat/grid_map_core/GridMap.hpp`
provides nearest/bilinear lookup on metric grid samples. This is a **Ranger
adaptation**, not a recreation of the original paper's vehicle or benchmark.

## Platform adaptation

- Geometry: front/rear half-wheelbase .247 m, left/right half-track .182 m,
  wheel radius .09 m. Translation/wheel-linear bounds .35 m/s; 3D yaw .9 rad/s.
- Free 4WIDS twists are projected to the nearest Ranger-native translation,
  curvature-bounded arc, or spin. Projection is applied inside both 3D/4D
  rollout and at execution; it is not merely a command clamp after prediction.
  The shared physical adapter still applies finite steering preparation.
- The upstream fixed seed 623 is exposed as the episode seed. Single-thread
  OpenMP avoids the upstream shared random-engine data race.
- Retained: Gaussian exploitation/exploration sampling, importance weighting,
  weighted update, Savitzky–Golay filter, path-error switching thresholds .5 m
  and .5 rad, bidirectional warm-start conversion, original cost structure.
- 3000 samples, 40 steps × .12 s, .30 m/s reference speed. Original linear noise
  magnitudes are scaled by .35/2; yaw noise .55 rad/s. 3D lambda 200, 4D 100;
  exploration .1, alpha .975. All numerical settings are in `bridge.cpp`.
- The ROS2 adapter supplies observed lidar occupancy (6 cm raster; inscribed
  inflation .25+.035 m, exponential decay 8/m), reference-path distance/yaw
  fields and the shared navigation anchor. No actor ground truth enters it.

The source copies of modified headers are generated in `build/include`;
`provenance.json` records upstream commit, modifications and binary hash.

Build from workspace root:

```bash
python3 integration/mppi_h_adapter/build.py
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 src/rangermini_dynamic_semantic/scripts/run_grounded_delivery.py \
  --output artifacts/public_mppi_benchmark_20261020/example \
  --planner mppi-h --policy predictive --proposal-selector cost \
  --scenario encounter_entry --verified-recovery --map-recovery --stable-modes
```

`mppi-3d` and `mppi-4d` select the author's fixed-space solvers. Raw solver
commands, native commands, solver dimension, objective, computation time,
actual chassis pose, wheel angles and physical mode are logged each cycle.
