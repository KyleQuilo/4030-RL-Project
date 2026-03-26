#!/usr/bin/env python3
"""
Headless TurtleBot3 Gazebo launch for training.

This mirrors the upstream turtlebot3_world.launch.py from turtlebot3_gazebo,
but omits the separate Gazebo GUI client process. That keeps the simulator,
bridge, spawn, and robot state publisher running even when no window is open.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    turtlebot3_gazebo_dir = get_package_share_directory("turtlebot3_gazebo")
    launch_file_dir = os.path.join(turtlebot3_gazebo_dir, "launch")
    ros_gz_sim_dir = get_package_share_directory("ros_gz_sim")

    use_sim_time = LaunchConfiguration("use_sim_time")
    x_pose = LaunchConfiguration("x_pose")
    y_pose = LaunchConfiguration("y_pose")
    gz_version = LaunchConfiguration("gz_version")
    gz_args = LaunchConfiguration("gz_args")
    on_exit_shutdown = LaunchConfiguration("on_exit_shutdown")

    world = os.path.join(
        turtlebot3_gazebo_dir,
        "worlds",
        "turtlebot3_world.world",
    )

    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation (Gazebo) clock if true",
    )
    declare_x_pose = DeclareLaunchArgument(
        "x_pose",
        default_value="-2.0",
        description="Initial robot x position",
    )
    declare_y_pose = DeclareLaunchArgument(
        "y_pose",
        default_value="-0.5",
        description="Initial robot y position",
    )
    declare_gz_version = DeclareLaunchArgument(
        "gz_version",
        default_value="8",
        description="Gazebo Sim major version",
    )
    declare_gz_args = DeclareLaunchArgument(
        "gz_args",
        default_value="",
        description="Extra arguments passed to Gazebo Sim server",
    )
    declare_on_exit_shutdown = DeclareLaunchArgument(
        "on_exit_shutdown",
        default_value="false",
        description="Shutdown launch system when gz sim exits",
    )

    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_dir, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={
            "gz_args": ["-r -s -v2 ", world, " ", gz_args],
            "gz_version": gz_version,
            "on_exit_shutdown": on_exit_shutdown,
        }.items(),
    )

    spawn_turtlebot_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, "spawn_turtlebot3.launch.py")
        ),
        launch_arguments={
            "x_pose": x_pose,
            "y_pose": y_pose,
        }.items(),
    )

    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, "robot_state_publisher.launch.py")
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    set_env_vars_resources = AppendEnvironmentVariable(
        "GZ_SIM_RESOURCE_PATH",
        os.path.join(turtlebot3_gazebo_dir, "models"),
    )

    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_x_pose)
    ld.add_action(declare_y_pose)
    ld.add_action(declare_gz_version)
    ld.add_action(declare_gz_args)
    ld.add_action(declare_on_exit_shutdown)
    ld.add_action(set_env_vars_resources)
    ld.add_action(gzserver_cmd)
    ld.add_action(spawn_turtlebot_cmd)
    ld.add_action(robot_state_publisher_cmd)
    return ld
