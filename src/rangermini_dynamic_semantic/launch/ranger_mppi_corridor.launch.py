#!/usr/bin/env python3
"""Run paired Ranger command-semantics MPPI trials in the established corridor world."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare("rangermini_dynamic_semantic")
    world = PathJoinSubstitution([pkg, "worlds", "ranger_lab_corridor_firedoor_half_open.sdf"])
    rviz = PathJoinSubstitution([pkg, "rviz", "corridor_semantic.rviz"])
    sim_time = {"use_sim_time": True, "require_sim_time": True}
    trial = {
        "trial_id": LaunchConfiguration("trial_id"),
        "scenario_id": LaunchConfiguration("scenario_id"),
        "seed": ParameterValue(LaunchConfiguration("seed"), value_type=int),
        "method_mode": "Ours",
    }
    return LaunchDescription([
        DeclareLaunchArgument("world_file", default_value=world),
        DeclareLaunchArgument("launch_rviz", default_value="false"),
        DeclareLaunchArgument("mppi_method", default_value="command_aware",
                              choices=["posthoc_adapter", "command_aware"]),
        DeclareLaunchArgument("trial_id", default_value="corridor_mppi_trial"),
        DeclareLaunchArgument("scenario_id", default_value="corridor_waypoint"),
        DeclareLaunchArgument("seed", default_value="0"),
        DeclareLaunchArgument("goal_x", default_value="17.0"),
        DeclareLaunchArgument("goal_y", default_value="-2.10"),
        DeclareLaunchArgument("goal_tolerance", default_value="0.42"),
        DeclareLaunchArgument("max_speed", default_value="0.30"),
        DeclareLaunchArgument("samples_per_behavior", default_value="32"),
        DeclareLaunchArgument("horizon", default_value="16"),
        DeclareLaunchArgument("account_initial_mode_switch", default_value="true"),
        DeclareLaunchArgument("allow_reverse", default_value="true"),
        DeclareLaunchArgument("require_goal_heading", default_value="false"),
        DeclareLaunchArgument("goal_yaw", default_value="0.0"),
        DeclareLaunchArgument("bridge_camera", default_value="false"),
        DeclareLaunchArgument("output_dir", default_value="/tmp/rangermini_mppi_trials"),
        ExecuteProcess(cmd=["ign", "gazebo", "-r", LaunchConfiguration("world_file")], output="screen"),
        TimerAction(period=2.0, actions=[Node(
            package="ros_gz_bridge", executable="parameter_bridge",
            name="ranger_mppi_corridor_bridge", output="screen",
            arguments=[
                "/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock",
                "/model/rangermini_2_0/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist",
                "/model/rangermini_2_0/pose@geometry_msgs/msg/PoseStamped[ignition.msgs.Pose",
                "/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan",
                "/world/ranger_lab_corridor/set_pose@ros_gz_interfaces/srv/SetEntityPose",
            ],
            remappings=[("/model/rangermini_2_0/cmd_vel", "/cmd_vel_gazebo")],
        )]),
        Node(package="ros_gz_bridge", executable="parameter_bridge",
             name="service_camera_bridge", output="screen",
             arguments=["/camera/image@sensor_msgs/msg/Image[ignition.msgs.Image"],
             condition=IfCondition(LaunchConfiguration("bridge_camera"))),
        Node(package="rangermini_dynamic_semantic", executable="ranger_motion_adapter",
             name="ranger_motion_adapter", output="screen",
             parameters=[{"input_topic": "/cmd_vel_safe",
                          "output_topic": "/cmd_vel_gazebo",
                          "min_turn_radius": 0.4764,
                          "wheelbase": 0.494, "track": 0.364,
                          "audit_rate_hz": 10.0}]),
        Node(package="rangermini_dynamic_semantic", executable="gazebo_odom_adapter",
             name="gazebo_odom_adapter", output="screen",
             parameters=[{"input_topic": "/model/rangermini_2_0/pose"}]),
        Node(package="rangermini_dynamic_semantic",
             executable="ranger_mppi_waypoint_controller",
             name="ranger_mppi_waypoint_controller", output="screen",
             parameters=[sim_time, trial, {
                 "mppi_method": LaunchConfiguration("mppi_method"),
                 "goal_x": ParameterValue(LaunchConfiguration("goal_x"), value_type=float),
                 "goal_y": ParameterValue(LaunchConfiguration("goal_y"), value_type=float),
                 "goal_tolerance": ParameterValue(
                     LaunchConfiguration("goal_tolerance"), value_type=float),
                 "max_speed": ParameterValue(
                     LaunchConfiguration("max_speed"), value_type=float),
                 "samples_per_behavior": ParameterValue(
                     LaunchConfiguration("samples_per_behavior"), value_type=int),
                 "horizon": ParameterValue(LaunchConfiguration("horizon"), value_type=int),
                 "account_initial_mode_switch": ParameterValue(
                     LaunchConfiguration("account_initial_mode_switch"), value_type=bool),
                 "allow_reverse": ParameterValue(LaunchConfiguration("allow_reverse"), value_type=bool),
                 "require_goal_heading": ParameterValue(LaunchConfiguration("require_goal_heading"), value_type=bool),
                 "goal_yaw": ParameterValue(LaunchConfiguration("goal_yaw"), value_type=float),
             }]),
        Node(package="rangermini_dynamic_semantic", executable="dynamic_benchmark_metrics",
             name="dynamic_benchmark_metrics", output="screen",
             parameters=[sim_time, trial, {
                 "controller_backend": "mppi",
                 "mppi_method": LaunchConfiguration("mppi_method"),
                 "output_dir": LaunchConfiguration("output_dir"),
             }]),
        Node(package="rviz2", executable="rviz2", name="rviz2",
             arguments=["-d", rviz], output="screen",
             condition=IfCondition(LaunchConfiguration("launch_rviz"))),
    ])
