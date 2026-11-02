#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare("rangermini_dynamic_semantic")
    world = PathJoinSubstitution([pkg, "worlds", "corridor_902_904_906_908.sdf"])
    rviz = PathJoinSubstitution([pkg, "rviz", "corridor_semantic.rviz"])
    config = PathJoinSubstitution([pkg, "config", "corridor_semantic_params.yaml"])
    web_app = PathJoinSubstitution([pkg, "web_ui", "app.py"])

    return LaunchDescription([
        DeclareLaunchArgument("launch_gazebo", default_value="true"),
        DeclareLaunchArgument("launch_rviz", default_value="true"),
        DeclareLaunchArgument("launch_web_ui", default_value="true"),
        DeclareLaunchArgument("enable_npc_schedule", default_value="true"),
        DeclareLaunchArgument("enable_sim_perception", default_value="true"),
        DeclareLaunchArgument("use_llm", default_value="false"),
        DeclareLaunchArgument("llm_provider", default_value="offline"),
        DeclareLaunchArgument("llm_model", default_value="deepseek-v4-flash"),
        DeclareLaunchArgument("web_ui_port", default_value="8502"),
        DeclareLaunchArgument("npc_initial_time", default_value="-1.0"),
        DeclareLaunchArgument("npc_random_seed", default_value="-1"),
        DeclareLaunchArgument("perception_forced_target_misses", default_value="0"),
        DeclareLaunchArgument("planner_test_response", default_value=""),
        # Compatibility guard for the legacy MID360 launch. Office RPG never
        # starts or consumes LaserScan; this must remain false.
        DeclareLaunchArgument("use_pointcloud_to_laserscan", default_value="false"),

        ExecuteProcess(cmd=["ign", "gazebo", "-r", world], output="screen",
                       condition=IfCondition(LaunchConfiguration("launch_gazebo"))),
        Node(
            package="ros_gz_bridge", executable="parameter_bridge",
            name="office_rpg_gazebo_bridge", output="screen",
            arguments=[
                "/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock",
                "/model/rangermini_2_0/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist",
                "/model/rangermini_2_0/pose@geometry_msgs/msg/PoseStamped[ignition.msgs.Pose",
                "/world/corridor_902_904_906_908/pose/info@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V",
                "/world/corridor_902_904_906_908/set_pose@ros_gz_interfaces/srv/SetEntityPose",
                "/camera/image@sensor_msgs/msg/Image[ignition.msgs.Image",
                "/camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
                "/camera/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image",
            ],
            remappings=[("/model/rangermini_2_0/cmd_vel", "/cmd_vel_gazebo")],
            condition=IfCondition(LaunchConfiguration("launch_gazebo")),
        ),
        Node(package="rangermini_dynamic_semantic", executable="ranger_motion_adapter",
             name="ranger_motion_adapter", output="screen",
             parameters=[{"input_topic": "/cmd_vel_safe",
                          "output_topic": "/cmd_vel_gazebo",
                          "min_turn_radius": 0.4764,
                          "wheelbase": 0.494, "track": 0.364}],
             condition=IfCondition(LaunchConfiguration("launch_gazebo"))),
        Node(package="rangermini_dynamic_semantic", executable="gazebo_odom_adapter",
             name="gazebo_odom_adapter", output="screen",
             parameters=[{"input_topic": "/model/rangermini_2_0/pose"}],
             condition=IfCondition(LaunchConfiguration("launch_gazebo"))),
        Node(package="rviz2", executable="rviz2", name="office_rpg_rviz",
             arguments=["-d", rviz], output="screen",
             condition=IfCondition(LaunchConfiguration("launch_rviz"))),

        # Existing semantic navigation and safety chain, retained intact.
        Node(package="rangermini_dynamic_semantic", executable="corridor_semantic_detector_stub",
             name="corridor_semantic_detector_stub", output="screen"),
        Node(package="rangermini_dynamic_semantic", executable="corridor_semantic_memory_node",
             name="corridor_semantic_memory_node", parameters=[config], output="screen"),
        Node(package="rangermini_dynamic_semantic", executable="corridor_human_command_generator",
             name="corridor_human_command_generator", parameters=[{
                 "mode": "unsafe_centerline", "forward_speed": 0.30, "goal_x": 13.35}],
             output="screen"),
        Node(package="rangermini_dynamic_semantic", executable="corridor_semantic_filter",
             name="corridor_semantic_filter", parameters=[config],
             remappings=[("/cmd_vel_safe", "/cmd_vel_safe_raw")], output="screen"),
        Node(package="rangermini_dynamic_semantic", executable="cmd_vel_watchdog",
             name="cmd_vel_watchdog", parameters=[{
                 "input_topic": "/cmd_vel_safe_raw", "output_topic": "/cmd_vel_safe",
                 "timeout": 0.35, "enable_depth_guard": True,
                 "depth_topic": "/camera/depth_image",
                 "obstacle_stop_distance": 0.80,
                 "obstacle_clear_distance": 0.95}], output="screen"),
        Node(package="rangermini_dynamic_semantic", executable="corridor_markers",
             name="corridor_markers", parameters=[config], output="screen"),

        Node(package="rangermini_dynamic_semantic", executable="office_rpg_npc_schedule",
             name="office_rpg_npc_schedule", output="screen",
             parameters=[{"initial_time": ParameterValue(
                              LaunchConfiguration("npc_initial_time"), value_type=float),
                          "random_seed": ParameterValue(
                              LaunchConfiguration("npc_random_seed"), value_type=int)}],
             condition=IfCondition(LaunchConfiguration("enable_npc_schedule"))),
        Node(package="rangermini_dynamic_semantic", executable="office_rpg_sim_perception_adapter",
             name="office_rpg_sim_perception_adapter", output="screen",
             parameters=[{"forced_target_misses": ParameterValue(
                 LaunchConfiguration("perception_forced_target_misses"), value_type=int)}],
             condition=IfCondition(LaunchConfiguration("enable_sim_perception"))),
        Node(package="rangermini_dynamic_semantic", executable="office_rpg_mission_manager",
             name="office_rpg_mission_manager", output="screen", parameters=[{
                 "llm_provider": LaunchConfiguration("llm_provider"),
                 "llm_model": LaunchConfiguration("llm_model")}]),
        Node(package="rangermini_dynamic_semantic", executable="office_rpg_search_executor",
             name="office_rpg_search_executor", output="screen",
             parameters=[{"initial_time": ParameterValue(
                 LaunchConfiguration("npc_initial_time"), value_type=float),
                          "planner_test_response": LaunchConfiguration(
                              "planner_test_response")}]),

        TimerAction(period=4.0, actions=[ExecuteProcess(
            cmd=["streamlit", "run", web_app, "--server.port",
                 LaunchConfiguration("web_ui_port"), "--server.headless", "true"],
            output="screen", condition=IfCondition(LaunchConfiguration("launch_web_ui")))])
    ])
