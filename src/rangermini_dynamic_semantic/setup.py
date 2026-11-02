from glob import glob
from setuptools import setup
import os

package_name = "rangermini_dynamic_semantic"

data_files = [
    ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
    ("share/" + package_name, ["package.xml", "README.md"]),
]

for folder in ["launch", "worlds", "description", "config", "rviz", "docs", "scripts", "web_ui"]:
    for path in glob(os.path.join(folder, "*")):
        if os.path.isfile(path):
            data_files.append((os.path.join("share", package_name, folder), [path]))

setup(
    name=package_name,
    version="0.20.0",
    packages=[package_name],
    data_files=data_files,
    scripts=[
        "scripts/corridor_trial_metrics",
        "scripts/corridor_human_command_generator",
        "scripts/kinematic_corridor_sim",
        "scripts/corridor_semantic_filter",
        "scripts/corridor_markers",
        "scripts/corridor_semantic_detector_stub",
        "scripts/corridor_semantic_memory_node",
        "scripts/cmd_vel_watchdog",
        "scripts/start_office_rpg_demo.sh",
    ],
    install_requires=["setuptools", "cloudpickle"],
    zip_safe=True,
    maintainer="Harry",
    maintainer_email="harry_vancepoy@mail.com",
    description="Dynamic indoor streaming semantic-maintenance benchmark for RangerMini 2.0.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "human_command_generator = rangermini_dynamic_semantic.human_command_generator:main",
            "kinematic_rangermini_sim = rangermini_dynamic_semantic.kinematic_rangermini_sim:main",
            "risk_cmd_filter = rangermini_dynamic_semantic.risk_cmd_filter:main",
            "doorway_markers = rangermini_dynamic_semantic.doorway_markers:main",
            "csv_logger = rangermini_dynamic_semantic.csv_logger:main",
            "compute_trial_metrics = rangermini_dynamic_semantic.compute_trial_metrics:main",
            "plot_latest_log = rangermini_dynamic_semantic.plot_latest_log:main",
            "cmd_vel_watchdog = rangermini_dynamic_semantic.cmd_vel_watchdog:main",
            "task_landmark_planner = rangermini_dynamic_semantic.task_landmark_planner:main",
            "evaluate_task_landmarks = rangermini_dynamic_semantic.evaluate_task_landmarks:main",
            "gazebo_odom_adapter = rangermini_dynamic_semantic.gazebo_odom_adapter:main",
            "office_rpg_npc_schedule = rangermini_dynamic_semantic.npc_schedule_node:main",
            "office_rpg_sim_perception_adapter = rangermini_dynamic_semantic.sim_perception_adapter:main",
            "office_rpg_mission_manager = rangermini_dynamic_semantic.office_mission_manager:main",
            "office_rpg_search_executor = rangermini_dynamic_semantic.office_search_executor:main",
            "dynamic_scene_manager = rangermini_dynamic_semantic.dynamic_scene_manager:main",
            "streaming_perception_scheduler = rangermini_dynamic_semantic.streaming_perception_scheduler:main",
            "rgbd_semantic_perception = rangermini_dynamic_semantic.rgbd_semantic_perception:main",
            "temporal_semantic_memory = rangermini_dynamic_semantic.temporal_semantic_memory:main",
            "dynamic_topology_maintenance = rangermini_dynamic_semantic.dynamic_topology_maintenance:main",
            "benchmark_navigation_controller = rangermini_dynamic_semantic.benchmark_navigation_controller:main",
            "ranger_mppi_waypoint_controller = rangermini_dynamic_semantic.ranger_mppi_waypoint_controller:main",
            "ranger_motion_adapter = rangermini_dynamic_semantic.ranger_motion_adapter:main",
            "ranger_feedback_bridge = rangermini_dynamic_semantic.ranger_feedback_bridge:main",
            "dynamic_benchmark_metrics = rangermini_dynamic_semantic.dynamic_benchmark_metrics:main",
            "trial_reset_coordinator = rangermini_dynamic_semantic.trial_reset_coordinator:main",
            "atomic_reset_smoke = rangermini_dynamic_semantic.atomic_reset_smoke:main",
            "grounded_dynamic_mission_agent = rangermini_dynamic_semantic.grounded_mission_node:main",
        ],
    },
)
