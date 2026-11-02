#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    pkg = FindPackageShare("rangermini_dynamic_semantic")
    world = PathJoinSubstitution([pkg, "worlds", "corridor_902_904_906_908.sdf"])
    rviz_cfg = PathJoinSubstitution([pkg, "rviz", "corridor_semantic.rviz"])
    cfg = PathJoinSubstitution([pkg, "config", "corridor_semantic_params.yaml"])

    return LaunchDescription([
        DeclareLaunchArgument("use_gazebo", default_value="true"),
        DeclareLaunchArgument("launch_rviz", default_value="true"),
        DeclareLaunchArgument("human_mode", default_value="unsafe_centerline"),
        DeclareLaunchArgument("output_dir", default_value="/tmp/rangermini_corridor_logs"),

        ExecuteProcess(
            cmd=["ign", "gazebo", "-r", world],
            shell=False, output="screen",
            condition=IfCondition(LaunchConfiguration("use_gazebo")),
        ),
        TimerAction(period=2.0, actions=[Node(
            package="ros_gz_bridge", executable="parameter_bridge",
            name="corridor_gazebo_bridge", output="screen",
            arguments=[
                "/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock",
                "/model/rangermini_2_0/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist",
                "/model/rangermini_2_0/pose@geometry_msgs/msg/PoseStamped[ignition.msgs.Pose",
                "/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan",
                "/camera/image@sensor_msgs/msg/Image[ignition.msgs.Image",
                "/camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
                "/camera/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image",
            ],
            remappings=[
                ("/model/rangermini_2_0/cmd_vel", "/cmd_vel_gazebo"),
            ],
            condition=IfCondition(LaunchConfiguration("use_gazebo")),
        )]),
        Node(
            package="rangermini_dynamic_semantic", executable="ranger_motion_adapter",
            name="ranger_motion_adapter", output="screen",
            parameters=[{"input_topic": "/cmd_vel_safe",
                         "output_topic": "/cmd_vel_gazebo",
                         "min_turn_radius": 0.4764,
                         "wheelbase": 0.494, "track": 0.364}],
            condition=IfCondition(LaunchConfiguration("use_gazebo")),
        ),
        Node(
            package="rangermini_dynamic_semantic", executable="gazebo_odom_adapter",
            name="gazebo_odom_adapter", output="screen",
            parameters=[{
                "input_topic": "/model/rangermini_2_0/pose",
            }],
            condition=IfCondition(LaunchConfiguration("use_gazebo")),
        ),

        Node(
            package="rviz2", executable="rviz2", name="rviz2",
            arguments=["-d", rviz_cfg], output="screen",
            condition=IfCondition(LaunchConfiguration("launch_rviz")),
        ),

        Node(
            package="rangermini_dynamic_semantic", executable="corridor_semantic_detector_stub",
            name="corridor_semantic_detector_stub", output="screen",
        ),
        Node(
            package="rangermini_dynamic_semantic", executable="corridor_semantic_memory_node",
            name="corridor_semantic_memory_node", output="screen",
            parameters=[cfg],
        ),
        Node(
            package="rangermini_dynamic_semantic", executable="task_landmark_planner",
            name="task_landmark_planner", output="screen",
            parameters=[{"target_room": "906", "budget": 4}],
        ),

        Node(
            package="rangermini_dynamic_semantic", executable="corridor_human_command_generator",
            name="corridor_human_command_generator", output="screen",
            parameters=[{
                "mode": LaunchConfiguration("human_mode"),
                "forward_speed": 0.30,
                "goal_x": 13.35,
            }]
        ),
        Node(
            package="rangermini_dynamic_semantic", executable="corridor_semantic_filter",
            name="corridor_semantic_filter", output="screen",
            parameters=[cfg],
            remappings=[("/cmd_vel_safe", "/cmd_vel_safe_raw")],
        ),
        Node(
            package="rangermini_dynamic_semantic", executable="cmd_vel_watchdog",
            name="cmd_vel_watchdog", output="screen",
            parameters=[{
                "input_topic": "/cmd_vel_safe_raw",
                "output_topic": "/cmd_vel_safe",
                "timeout": 0.35,
            }],
        ),
        Node(
            package="rangermini_dynamic_semantic", executable="corridor_markers",
            name="corridor_markers", output="screen",
            parameters=[cfg],
        ),
        Node(
            package="rangermini_dynamic_semantic", executable="csv_logger",
            name="corridor_csv_logger", output="screen",
            parameters=[{
                "output_dir": LaunchConfiguration("output_dir"),
                "file_prefix": "corridor_trial",
            }]
        ),
    ])
