#!/usr/bin/env python3
"""Real MID360S pipeline with the measured Ranger Mini installation TF."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = PathJoinSubstitution([
        FindPackageShare("rangermini_dynamic_semantic"), "config", "MID360s_config.json"
    ])
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("config_path", default_value=config),
        DeclareLaunchArgument("scan_min_height", default_value="0.05"),
        DeclareLaunchArgument("scan_max_height", default_value="0.55"),
        DeclareLaunchArgument("use_pointcloud_to_laserscan", default_value="false"),
        Node(
            package="tf2_ros", executable="static_transform_publisher",
            name="mid360s_static_tf",
            arguments=[
                "--x", "0.30", "--y", "0.0", "--z", "0.70",
                "--roll", "0.0", "--pitch", "0.523599", "--yaw", "0.0",
                "--frame-id", "base_link", "--child-frame-id", "livox_frame",
            ],
            parameters=[{"use_sim_time": ParameterValue(
                LaunchConfiguration("use_sim_time"), value_type=bool)}],
            output="screen",
        ),
        Node(
            package="livox_ros_driver2", executable="livox_ros_driver2_node",
            name="livox_lidar_publisher", output="screen",
            parameters=[{
                "use_sim_time": ParameterValue(
                    LaunchConfiguration("use_sim_time"), value_type=bool),
                "xfer_format": 0, "multi_topic": 0, "data_src": 0,
                "publish_freq": 10.0, "output_data_type": 0,
                "frame_id": "livox_frame",
                "lvx_file_path": "/tmp/unused.lvx",
                "user_config_path": LaunchConfiguration("config_path"),
                "cmdline_input_bd_code": "livox0000000001",
                "qos_overrides./livox/lidar.publisher.reliability": "best_effort",
                "qos_overrides./livox/lidar.publisher.history": "keep_last",
                "qos_overrides./livox/lidar.publisher.depth": 5,
            }],
        ),
        Node(
            package="pointcloud_to_laserscan",
            executable="pointcloud_to_laserscan_node",
            name="mid360s_pointcloud_to_laserscan", output="screen",
            parameters=[{
                "use_sim_time": ParameterValue(
                    LaunchConfiguration("use_sim_time"), value_type=bool),
                "target_frame": "base_link", "transform_tolerance": 0.10,
                "min_height": ParameterValue(
                    LaunchConfiguration("scan_min_height"), value_type=float),
                "max_height": ParameterValue(
                    LaunchConfiguration("scan_max_height"), value_type=float),
                "angle_min": -3.14159, "angle_max": 3.14159,
                "angle_increment": 0.0087, "scan_time": 0.1,
                "range_min": 0.20, "range_max": 30.0,
                "use_inf": True, "inf_epsilon": 1.0,
            }],
            remappings=[("cloud_in", "/livox/lidar"), ("scan", "/scan")],
            condition=IfCondition(LaunchConfiguration("use_pointcloud_to_laserscan")),
        ),
    ])
