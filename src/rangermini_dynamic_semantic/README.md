# 行为条件 MPPI 与四轮转向执行

当前入口是 `scripts/run_grounded_delivery.py`，论文与展示分别见工作区根目录的 `paper/office_delivery_research_revision.tex` 和 `post/index.html`。下方保留的 v0.20 与任务记忆扩展属于历史系统。

## 当前执行链与实物迁移接口

VLM 提议通过行为参考、代价和可行域进入 MPPI；规划器读取当前轮角与运动模式，输出本体速度请求。Gazebo 执行端用有限转向速率模型实现该请求。硬件执行端应使用底盘驱动与实测轮角，不能将 Gazebo 的模型输出再当作硬件反馈。

| 接口 | 仿真来源 | 实物对接要求 |
|---|---|---|
| 位姿、时间戳 | Gazebo 里程计适配器 | 定位/里程计，统一 map/odom/base 坐标与时钟 |
| 激光与 RGB-D | Gazebo 传感器消息 | 激光、标定后的相机内外参与深度尺度 |
| 四轮角度、模式、执行速度 | `ranger_motion_adapter` 模型状态 | 独立执行器反馈、模式消息、里程计；确认轮位顺序、弧度、零点和正负号 |
| 本体速度请求 | `/cmd_vel_safe` | 经独立安全与超时保护接底盘驱动输入 |
| 静态几何和服务目标 | 已登记实验室几何 | 测量/建图后登记门洞、目标区域和低矮障碍，沿用同一规划表示 |

当前实机驱动的逐通道反馈修复包在工作区 `integration/ranger_ros2/`：修复八路状态重复读取首项的问题，已在隔离副本编译。补丁未应用到实机工作区。模型转向速率需用上车阶跃记录标定；实机报告使用反馈轮角，仿真报告使用模型轮角。

只读反馈桥 `ranger_feedback_bridge` 消费 `/actuator_state`、`/motion_state`、`/odom`，保留来源时间戳并发布规划器状态。它不发布底盘命令。使用时需要已安装 `ranger_msgs` 的 Ranger ROS2 overlay；参数模板为工作区 `integration/ranger_ros2/feedback_bridge.yaml`。按实物确认四轮顺序、弧度比例和零点后设置 `calibration_ready`；未确认或输入过期时发布无效状态，规划器保持停车。

```bash
ros2 run rangermini_dynamic_semantic ranger_feedback_bridge --ros-args \
  --params-file integration/ranger_ros2/feedback_bridge.yaml
```

校验记录分别在 `artifacts/grounded_delivery_revision_20261016/execution_contract/`：`ros_watchdog.json`（暂停ROS时钟）、`scan_dropout_check.json`（Gazebo断流恢复）、`kinematic_equivalence.json`（数值/计算等价）、`feedback_bridge_check.json`（合成ROS反馈）。后者保持四通道并验证比例/零点、模式/速度和过期失效，不等于实测舵角跟踪验收。


## 实验复现

```bash
python3 src/rangermini_dynamic_semantic/scripts/run_research_motion_matrix.py \
  --output artifacts/<new_batch> --scenarios heading_passing,encounter_entry \
  --methods predictive_4ws,predictive_instant_steering --seeds 37 --execute
```

`predictive_instant_steering` 仅改变规划预测，执行端仍受有限转向速率约束。配置对照不标作独立的 MPPI-H 或 Nav2 算法。每批源码哈希、原始失败和独立几何审计均保留。

---

## 历史实现

## Optional grounded task/LLM extension

### Implementation note

The project now includes an **opt-in** evidence-grounded mission layer for the
existing dynamic indoor semantic benchmark. See
`docs/GROUNDED_DYNAMIC_LLM_NAV_GUIDE_CN.md` (deployment) and
`docs/TEGM_NAV_RESEARCH_PLAN_CN.md` (research/limitations).

It preserves the original SDF and topology assets, adds `/grounded_agent/*`
ROS interfaces, versioned goal IDs/cancel acknowledgements, a bounded optional
OpenAI-compatible language compiler, SQLite execution evidence, and a **graph-
level synthetic ObjectNav proxy**. These are not Gazebo/real-LLM results.

Quick offline checks:

```bash
PYTHONPATH=. python scripts/run_grounded_abc_replay.py --out /tmp/grounded_abc.json
PYTHONPATH=. python scripts/run_grounded_proxy_ablation.py --out /tmp/grounded_proxy --episodes 240
PYTHONPATH=. pytest -q test/test_grounded_dynamic_research.py
```

The legacy v0.20 original reset-participant tests are still known to fail; for
a manual single-run ROS demo disable trial-reset until that P0 integration is
fixed. This package must **not** be described as publication-ready.

---

## Original dynamic semantic-maintenance benchmark v0.20.0

## v0.20 research benchmark

The package now contains two intentionally separate tracks:

- `dynamic_semantic_benchmark.launch.py`: the primary research benchmark for
  streaming semantic maintenance in a dynamic multi-zone indoor facility;
- the original corridor/Office RPG launches: retained as regression tests and
  historical baselines.

The benchmark geometry and node scope are now frozen for the thesis ablation
phase.  Paper-facing definitions are kept in:

- `docs/PAPER_METHOD_SECTION_ZH.md`;
- `docs/EXPERIMENT_PROTOCOL_V020_ZH.md`;
- `docs/EXPERIMENT_INTERFACE_GAP_AUDIT.md`.

P0 gaps in the interface audit must be closed before collecting final results;
they are not described as completed capabilities.

The new world is not an office-door recognition demo.  It is a 30 x 20 m
facility with lobby, office, inspection, storage, loading, packing and
receiving regions.  A central service block creates upper and lower routes,
so temporary blockage and door-state changes require graph-level rerouting.

### What is real in the online loop

`rgbd_semantic_perception` consumes rendered RGB, aligned depth, CameraInfo and
odometry.  It does **not** subscribe to Gazebo world poses, entity names or the
benchmark ground-truth topics.  Its colour detector is the simulation backend
for a replaceable detector contract; on the D435i it can be replaced with a
TensorRT model without changing the scheduler, memory or topology nodes.

The online chain is:

```text
task / navigation failure
  -> streaming_perception_scheduler
  -> rendered RGB-D semantic observation
  -> temporal_semantic_memory
  -> dynamic_topology_maintenance
  -> graph-aware navigation / Nav2 adapter
```

The evaluation-only chain is:

```text
dynamic_scene_manager -> /benchmark/ground_truth/* -> metrics only
```

Online components must never subscribe to `/benchmark/ground_truth/*`.

### Run

```bash
ros2 launch rangermini_doorway_sim dynamic_semantic_benchmark.launch.py
```

Start a task-aware perception context and send a region goal:

```bash
ros2 topic pub --once /task_context std_msgs/msg/String \
  "{data: '{\"active\":true,\"task_type\":\"delivery\",\"target\":\"parcel_red\"}'}"

ros2 topic pub --once /benchmark/goal std_msgs/msg/String \
  "{data: '{\"target_region\":\"storage\"}'}"
```

Inspect the non-ground-truth products:

```bash
ros2 topic echo /semantic_observations
ros2 topic echo /semantic_memory_v2/snapshot
ros2 topic echo /dynamic_semantic_graph
ros2 topic echo /streaming/scheduler_stats
```

Per-trial JSONL and summaries are written under
`/tmp/rangermini_dynamic_benchmark`.  The included controller is an executable
holonomic smoke-test controller with lidar stopping and graph rerouting.  It is
not presented as Nav2; real-robot experiments should replace only that node
with a Nav2 adapter and retain the perception/memory/graph event contracts.

### Controlled disturbances

The default `mixed_shift` scenario injects:

1. a parcel migration from receiving to storage;
2. a mobile cart temporarily blocking the lower route;
3. recovery of that route after the cart moves away;
4. a door closing on the upper route and reopening later.

Timing, sensor errors and semantic lifecycle parameters are defined in
`config/dynamic_benchmark.yaml`, which makes trials reproducible and suitable
for fixed-rate versus task-driven ablations.

---

## Legacy corridor benchmark v0.13.1

## Current simulation contract

The Gazebo model is a **navigation-level planar abstraction**, not a
four-wheel-steering dynamics digital twin.

- footprint: Ranger Mini 2.0 overall size, `0.738 m x 0.500 m`;
- command: body-frame `geometry_msgs/Twist` (`vx`, `vy`, `wz`);
- motion: Gazebo Fortress `VelocityControl` on one rigid chassis;
- localization: Gazebo `PosePublisher` converted to `/odom` and
  `odom -> base_link`;
- wheels: visualization only;
- real execution: official `ranger_ros2` and the chassis controller handle
  Ackermann, crab, traverse and spin modes.

This separation is intentional: Gazebo validates task switching, semantic
memory, obstacle interaction and navigation success. It does not claim to
validate steering motor dynamics.

The collision floor covers the complete `x=[0,22] m` corridor. The previous
world only had physical ground up to `x=12 m`, even though the floor remained
visible beyond that point.

Runtime retasking to a room behind the chassis uses a stop-and-spin transition:
translation is stopped, the chassis rotates to the opposite corridor heading,
and route following then resumes. Long reverse traversal is intentionally not
generated.

## Current run

```bash
ros2 launch rangermini_doorway_sim corridor_semantic_benchmark.launch.py
```

The default state is stopped. Start a task only after the launch prints:

```text
Gazebo ground-truth odometry active
```

```bash
ros2 topic pub --once /task_goal std_msgs/msg/String "{data: '906'}"
ros2 topic pub --once /task_control std_msgs/msg/String "{data: 'START'}"
```

---

# Previous benchmark notes (v4.3)

v4.3 fixes the "dies halfway" issue and adds a RangerMini 2.0-style four steering module visualization.

## What was wrong in v4.2

The target door-front was too close to the upper corridor wall. After adding robot radius and safety margin, the final target could become infeasible, so the filter rejected many candidates and appeared to stop halfway.

## Main changes

- Reachable Room 906 door-front: y=4.15 instead of y=4.55.
- Route-aware local target sequence:
  - follow corridor
  - detour above S2
  - return to Room 906 door-front
- Smaller and less opaque risk inflation visualization.
- Selected trajectory should continue through the S2 detour instead of dying midway.
- Added `/debug/rangermini2_steer_modules`:
  - four steerable wheel modules
  - steering angles computed from chassis twist
  - closer to RangerMini 2.0 than a plain rectangle

## Clean build

```bash
cd ~/ranger_ws
rm -rf build/rangermini_doorway_sim install/rangermini_doorway_sim log
```

## Install

```bash
cd ~/ranger_ws/src
rm -rf rangermini_doorway_sim
unzip ~/Downloads/rangermini_doorway_sim_v4_3.zip
mv rangermini_doorway_sim_v4_3 rangermini_doorway_sim

cd ~/ranger_ws
colcon build --packages-select rangermini_doorway_sim --symlink-install
source install/setup.bash
```

## Run

```bash
ros2 launch rangermini_doorway_sim corridor_semantic_benchmark.launch.py \
  use_gazebo:=false \
  launch_rviz:=true \
  human_mode:=unsafe_centerline
```

## Check four-steering visualization

```bash
ros2 topic echo /debug/rangermini2_steer_modules
```

See:
- `docs/v4_3_route_and_rangermini2_notes.md`
- `docs/paper_point_v4_3.md`

---


# RangerMini Corridor Semantic Shared-Control Benchmark v4.2

v4.2 updates v4.1 in the direction you asked for: **make obstacle avoidance visually obvious and make the paper-writing story clear**.

## Main changes

- Adds a visible unsafe raw-human baseline:
  - yellow dashed `raw human prediction`
  - default `human_mode:=unsafe_centerline`
  - the raw prediction is designed to collide with S2.
- Makes obstacle avoidance explicit:
  - red inflated safety zones around S1/S2/S3
  - red rejected rollouts
  - blue selected safe rollout
  - green filtered safe output
  - blue via point above S2
- Keeps semantic navigation structure:
  - semantic detector stub
  - semantic memory
  - target selection: `Room 906 + glass + door_front`
  - semantic target pose
  - shared-control local filter
- Adds metrics:
  - success
  - collision
  - wrong-door
  - final target distance
  - minimum clearance
  - intervention count/intensity
  - stop count
  - arrival time
  - velocity jitter
- Adds writing guide:
  - `docs/thesis_writing_guide_semantic_shared_control.md`
  - `docs/v4_2_experiment_protocol.md`

## Clean build is strongly recommended

```bash
cd ~/ranger_ws
rm -rf build/rangermini_doorway_sim install/rangermini_doorway_sim log
```

## Install

```bash
cd ~/ranger_ws/src
rm -rf rangermini_doorway_sim
unzip ~/Downloads/rangermini_doorway_sim_v4_2.zip
mv rangermini_doorway_sim_v4_2 rangermini_doorway_sim

cd ~/ranger_ws
colcon build --packages-select rangermini_doorway_sim --symlink-install
source install/setup.bash
```

## Run the clear obstacle-avoidance demo

```bash
ros2 launch rangermini_doorway_sim corridor_semantic_benchmark.launch.py \
  use_gazebo:=false \
  launch_rviz:=true \
  human_mode:=unsafe_centerline
```

## Check semantic chain

```bash
ros2 topic echo /semantic_detections_json
ros2 topic echo /semantic_memory_debug
ros2 topic echo /semantic_target_pose
```

## Compute corridor metrics

```bash
ros2 run rangermini_doorway_sim corridor_trial_metrics --log_dir /tmp/rangermini_corridor_logs
```

## Plot latest log

```bash
ros2 run rangermini_doorway_sim plot_latest_log --log_dir /tmp/rangermini_corridor_logs
```

---


# RangerMini Corridor Semantic Benchmark v4.1

v4.1 fixes the `StopIteration` entry-point crash and moves the corridor benchmark closer to a real semantic-navigation stack.

## What was wrong

Your error:

```text
StopIteration
load_entry_point(..., 'console_scripts', 'corridor_semantic_filter')
```

means the generated console-script wrapper existed, but the installed Python package metadata did not contain that entry point. This usually happens after rapid setup.py changes with stale `build/` and `install/` artifacts.

v4.1 avoids this for the new corridor nodes by installing them as direct Python scripts instead of only relying on console-script metadata.

## You must clean the old build once

Run this exactly once before rebuilding v4.1:

```bash
cd ~/ranger_ws
rm -rf build/rangermini_doorway_sim install/rangermini_doorway_sim log
```

Then install:

```bash
cd ~/ranger_ws/src
rm -rf rangermini_doorway_sim
unzip ~/Downloads/rangermini_doorway_sim_v4_1.zip
mv rangermini_doorway_sim_v4_1 rangermini_doorway_sim

cd ~/ranger_ws
colcon build --packages-select rangermini_doorway_sim --symlink-install
source install/setup.bash
```

## Run

```bash
ros2 launch rangermini_doorway_sim corridor_semantic_benchmark.launch.py \
  use_gazebo:=false \
  launch_rviz:=true \
  human_mode:=wrong_top_drift
```

## Semantic navigation structure

I inspected your uploaded `src.zip`. The useful architecture is:

```text
YOLO detector
  -> 2D detection
  -> localizer / projection
  -> semantic memory
  -> target selection
  -> coordinator / goal adapter
  -> Nav2 or local controller
```

v4.1 mirrors that structure in a lightweight way:

```text
corridor_semantic_detector_stub
  -> /semantic_detections_json
corridor_semantic_memory_node
  -> /semantic_target_pose
corridor_semantic_filter
  -> /cmd_vel_safe
kinematic_corridor_sim
  -> /odom
```

The detector is currently a stub, not YOLO. But it is intentionally separated so that later you can replace it with your YOLO detector output.

## Important distinction

The room labels and door materials are no longer just decoration. They are used by the memory node to select:

```text
target_room = 906
target_material = glass
target_label = door_front
```

That target pose then drives the local shared-control filter.



--- Previous README below ---

# RangerMini Corridor + Laboratory Semantic Benchmark v4.0

v4.0 adds a **corridor + laboratory semantic environment** inspired by your protocol figure:

- corridor with rooms **902 / 904 / 906 / 908**
- semantic door materials: **wood** for 902/904, **glass** for 906/908
- three corridor obstacles:
  - **S1 small box**
  - **S2 map-change box**
  - **S3 pedestrian proxy**
- target task: reach the **door-front of Room 906**
- RViz semantic markers show room names, door materials, door-front stars, protocol path, and corridor outline
- Gazebo world approximates the corridor + lab geometry
- shared-control filter becomes **goal-aware local navigation**, not just narrow-door avoidance

## Recommended run

```bash
ros2 launch rangermini_doorway_sim corridor_semantic_benchmark.launch.py \
  use_gazebo:=false \
  launch_rviz:=true \
  human_mode:=wrong_top_drift
```

`human_mode` options:
- `follow_corridor`
- `wrong_top_drift`
- `wrong_bottom_drift`
- `manual_bias`

## What the experiment demonstrates

- the human input tends to drift along the corridor or toward the wrong side;
- the system keeps forward progress;
- the system avoids corridor obstacles;
- the system uses **target room / door-front geometry** as the local goal;
- the system can incorporate **door material semantics** into the target module (currently exposed as semantic markers and parameters).

## Current limitations

This is still a **lightweight ROS2 kinematic benchmark**, not a full photorealistic digital twin.
It is meant for:
1. group meeting demo,
2. method illustration,
3. repeatable CSV metric collection,
4. preparation before replacing the geometry with your real MID360S + D435i local semantic perception.

## Build

```bash
cd ~/ranger_ws/src
rm -rf rangermini_doorway_sim
unzip /path/to/rangermini_doorway_sim_v4_0.zip
mv rangermini_doorway_sim_v4_0 rangermini_doorway_sim

cd ~/ranger_ws
colcon build --packages-select rangermini_doorway_sim
source install/setup.bash
```

## Plot latest corridor log

```bash
ros2 run rangermini_doorway_sim plot_latest_log --log_dir /tmp/rangermini_corridor_logs
```
