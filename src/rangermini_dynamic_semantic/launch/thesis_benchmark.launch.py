#!/usr/bin/env python3
"""论文最小复现实验入口。

把完整 benchmark 的工程参数收敛为少数实验变量；需要调试全部参数时，
仍可直接使用 dynamic_semantic_benchmark.launch.py。
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    upstream = PathJoinSubstitution([
        FindPackageShare("rangermini_dynamic_semantic"),
        "launch",
        "dynamic_semantic_benchmark.launch.py",
    ])

    include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(upstream),
        launch_arguments={
            "launch_gazebo": LaunchConfiguration("launch_gazebo"),
            "enable_trial_reset": "false",
            "enable_demo_controller": "true",
            "enable_disturbances": LaunchConfiguration("enable_disturbances"),
            "scenario": LaunchConfiguration("scenario"),
            "trial_id": LaunchConfiguration("trial_id"),
            "scenario_id": "S6",
            "seed": LaunchConfiguration("seed"),
            "method_mode": LaunchConfiguration("method_mode"),
            "auto_reset_on_start": "true",
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument("launch_gazebo", default_value="true"),
        DeclareLaunchArgument("enable_disturbances", default_value="true"),
        DeclareLaunchArgument("scenario", default_value="mixed_shift"),
        DeclareLaunchArgument("trial_id", default_value="manual_trial"),
        DeclareLaunchArgument("seed", default_value="0"),
        DeclareLaunchArgument("method_mode", default_value="Ours"),
        include,
    ])
