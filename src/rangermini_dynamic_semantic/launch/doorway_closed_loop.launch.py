#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def gazebo_command(gz_executable, world_path):
    return PythonExpression([
        "'", gz_executable, " gazebo -r ' if '", gz_executable,
        "' == 'ign' else '", gz_executable, " sim -r '"
    ])

def generate_launch_description():
    pkg = FindPackageShare("rangermini_dynamic_semantic")
    world = PathJoinSubstitution([pkg, "worlds", "narrow_doorway.sdf"])
    rviz_cfg = PathJoinSubstitution([pkg, "rviz", "doorway_sim.rviz"])
    return LaunchDescription([
        DeclareLaunchArgument("use_gazebo", default_value="false"),
        DeclareLaunchArgument("launch_rviz", default_value="true"),
        DeclareLaunchArgument("gz_executable", default_value="ign"),
        DeclareLaunchArgument("human_mode", default_value="right_bias"),
        DeclareLaunchArgument("output_dir", default_value="/tmp/rangermini_doorway_logs"),

        ExecuteProcess(
            cmd=[gazebo_command(LaunchConfiguration("gz_executable"), world), world],
            shell=True, output="screen",
            condition=IfCondition(LaunchConfiguration("use_gazebo")),
        ),
        Node(
            package="rviz2", executable="rviz2", name="rviz2",
            arguments=["-d", rviz_cfg], output="screen",
            condition=IfCondition(LaunchConfiguration("launch_rviz")),
        ),
        Node(
            package="rangermini_dynamic_semantic", executable="human_command_generator",
            name="human_command_generator", output="screen",
            parameters=[{
                "mode": LaunchConfiguration("human_mode"),
                "forward_speed": 0.32,
                "lateral_bias": -0.22,
                "duration": 25.0,
                "goal_y": 2.75,
            }]
        ),
        Node(
            package="rangermini_dynamic_semantic", executable="kinematic_rangermini_sim",
            name="kinematic_rangermini_sim", output="screen",
            parameters=[{
                "start_x": 0.0,
                "start_y": -1.65,
                "start_yaw_deg": 90.0,
                "goal_y": 2.75,
            }]
        ),
        Node(
            package="rangermini_dynamic_semantic", executable="risk_cmd_filter",
            name="risk_cmd_filter", output="screen",
            parameters=[PathJoinSubstitution([pkg, "config", "doorway_params.yaml"])]
        ),
        Node(
            package="rangermini_dynamic_semantic", executable="doorway_markers",
            name="doorway_markers", output="screen",
            parameters=[PathJoinSubstitution([pkg, "config", "doorway_params.yaml"])]
        ),
        Node(
            package="rangermini_dynamic_semantic", executable="csv_logger",
            name="doorway_csv_logger", output="screen",
            parameters=[{
                "output_dir": LaunchConfiguration("output_dir"),
                "file_prefix": "doorway_trial",
            }]
        ),
    ])
