#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.substitutions import FindPackageShare


def gazebo_command(gz_executable, world_path):
    return PythonExpression([
        "'",
        gz_executable,
        " gazebo -r ' if '",
        gz_executable,
        "' == 'ign' else '",
        gz_executable,
        " sim -r '"
    ])


def generate_launch_description():
    pkg = FindPackageShare("rangermini_dynamic_semantic")
    world = PathJoinSubstitution([pkg, "worlds", "narrow_doorway.sdf"])
    gz_executable = LaunchConfiguration("gz_executable")

    return LaunchDescription([
        DeclareLaunchArgument("gz_executable", default_value="ign"),
        ExecuteProcess(
            cmd=[gazebo_command(gz_executable, world), world],
            shell=True,
            output="screen",
        ),
    ])
