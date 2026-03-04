# 4030-RL-Project
ROS 2 Navigation RL Environment (AISE 4030)
Project Summary

This project implements a ROS 2 plus Gazebo mobile robot navigation environment wrapped in a Gymnasium style API for reinforcement learning. The TurtleBot3 Burger robot is controlled using continuous velocity commands and observes the environment using simulated 2D LiDAR and odometry. Phase 2 focuses on environment API confirmation and a Stable Baselines3 PPO baseline wrapper.

System Requirements

Ubuntu 24.04
ROS 2 Jazzy
Gazebo Sim (gz sim)
Python 3.10 plus

Recommended hardware
CPU is enough for Phase 2 validation
GPU optional for later training

Dependencies

Python packages
gymnasium
stable baselines3
numpy
pyyaml
torch

Install with pip

pip install -r requirements.txt

ROS packages
TurtleBot3 packages for Jazzy and simulation packages must be installed via apt.

Setup
1 Source ROS 2
source /opt/ros/jazzy/setup.bash
2 Set TurtleBot3 model
export TURTLEBOT3_MODEL=burger

Optional: make it permanent

echo "export TURTLEBOT3_MODEL=burger" >> ~/.bashrc
source ~/.bashrc
Launch Simulation

In Terminal 1

ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

Leave this running.

Verify ROS Topics

In Terminal 2

ros2 topic list
ros2 topic info /scan
ros2 topic info /odom
ros2 topic info /cmd_vel

Expected topics
/scan publishes sensor_msgs msg LaserScan
/odom publishes nav_msgs msg Odometry
/cmd_vel uses geometry_msgs msg TwistStamped

Optional sensor sanity checks

ros2 topic echo /scan --once
ros2 topic echo /odom --once
Phase 2 Validation Script

With the simulator running, execute

python3 training_script.py

This script:
Instantiates the environment
Prints observation space and action space information
Performs reset
Executes one step
Prints reward and termination flags

The console output from this run is included in the Phase 2 progress report.

Configuration

Key runtime parameters are stored in config.yaml:
Topic names
/scan
/odom
/cmd_vel

Environment constants
lidar_beams set to 60
control_hz set to 10
dt set to 0.1
max_steps set to 600
v_max set to 0.22
w_max set to 2.0
r_safe set to 0.30

Code Structure

environment.py
ROS 2 plus Gazebo to Gymnasium style environment wrapper

sensor_processing.py
LiDAR downsampling and observation construction utilities

ppo_agent.py
Stable Baselines3 PPO wrapper used as the baseline agent structure

training_script.py
Phase 2 required API confirmation and one step execution script

utils.py
Helper functions

ppo_results
Output folder for logs and checkpoints
