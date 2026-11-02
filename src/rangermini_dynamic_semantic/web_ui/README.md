# Ranger Office RPG Web UI

The Streamlit process owns one cached `rclpy` bridge and one background
`MultiThreadedExecutor`. Streamlit reruns only read its thread-safe cache; they
do not create ROS nodes and never publish `/cmd_vel`.

```bash
streamlit run ~/ranger_ws/src/rangermini_doorway_sim/web_ui/app.py --server.port 8502
```

The stop and reset controls call `/office_rpg/stop` and `/office_rpg/reset`.
`DEEPSEEK_API_KEY` is checked only as a boolean and its value is never rendered.

Normal mode never subscribes to simulation ground truth. It renders unknown
people only as `UNKNOWN`, and discovered people from their public `last_seen`
region. The developer truth toggle dynamically subscribes to
`/office_rpg/sim/npc_ground_truth`, displays a red simulation-only warning,
and destroys that subscription again when disabled. The toggle is visualization
only and is not connected to MissionSpec or the deterministic search executor.

Phase 1.5 additionally displays the state-machine phase, ordered search queue,
candidate beliefs, search/patrol progress, structured patrol report and event
timeline. The bridge never publishes chassis velocity commands.
