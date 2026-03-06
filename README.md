# 4030 RL Project

## ROS 2 Navigation Reinforcement Learning Environment  
**AISE 4030 Phase 2**

## Project Summary

This project implements a ROS 2 and Gazebo mobile robot navigation environment wrapped in a Gymnasium style API for reinforcement learning. The TurtleBot3 Burger robot is controlled using continuous velocity commands and observes the environment using simulated 2D LiDAR and odometry.

Phase 2 focuses on confirming that the custom environment exposes the required reinforcement learning interface and that the robot simulation can successfully provide observations and accept actions. A baseline PPO structure is also included for later development.

## System Requirements

- Ubuntu 24.04
- ROS 2 Jazzy
- Gazebo Sim
- Python 3.10 or later

## Recommended Hardware

- CPU is sufficient for Phase 2 validation
- GPU is optional for later training experiments

## Dependencies

### Python Packages

- gymnasium
- stable baselines3
- numpy
- pyyaml
- torch

### ROS Packages

TurtleBot3 packages for ROS 2 Jazzy and the required simulation packages must be installed separately using `apt`.

## Setup

### 1. Source ROS 2
```bash
source /opt/ros/jazzy/setup.bash
```

### Optional: Make the TurtleBot3 model permanent
```bash
echo "export TURTLEBOT3_MODEL=burger" >> ~/.bashrc
source ~/.bashrc
```

### Launch Simulation
Open a terminal and launch the TurtleBot3 Gazebo world:
```bash
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```
Leave this terminal running while performing the remaining checks.

### Verify ROS Topics
In a second terminal, verify the required ROS topics:
```bash
ros2 topic list
ros2 topic info /scan
ros2 topic info /odom
ros2 topic info /cmd_vel
```

## Expected Topics
```bash
/scan publishes sensor_msgs/msg/LaserScan
/odom publishes nav_msgs/msg/Odometry
/cmd_vel uses geometry_msgs/msg/TwistStamped
```

## Optional Sensor Sanity Checks
```bash
ros2 topic echo /scan --once
ros2 topic echo /odom --once
```
## Phase 2 Validation Script

With the simulator running, execute:
```bash
python3 training_script.py
```
This script:
- Instantiates the environment
- Prints the observation space
- Prints the action space
- Prints the selected device
- Performs an environment reset
- Executes one environment step
- Prints the reward and termination flags
- Confirms that the environment API is functioning correctly
The console output from this run is included in the Phase 2 progress report as Task 1 evidence.

## Configuration
Key runtime parameters are stored in config.yaml.

### Topics
- /scan
- /odom
- /cmd_vel

### Environment Parameters
- lidar_beams: 60
- control_hz: 10
- dt: 0.1
- max_steps: 600
- v_max: 0.22
- w_max: 2.0
- r_safe: 0.30

## Code Structure
`environment.py`
ROS 2 and Gazebo to Gymnasium style environment wrapper. Handles reset, step, observation construction, and action publishing.
`sensor_processing.py`
Utilities for LiDAR downsampling and observation preprocessing.

`ppo_agent.py`
Baseline PPO agent structure for reinforcement learning development.

`policy_network.py`
Policy network definition for the PPO agent skeleton.

`rollout_buffer.py`
Experience storage structure for PPO rollouts.

`training_script.py`
Phase 2 validation script used to confirm the environment API and execute one successful environment step.

`utils.py`
Helper functions for configuration loading and shared utilities.

`config.yaml`
Central configuration file containing topics and environment parameters.

`ppo_results/`
Output directory for logs, checkpoints, and training results.

`launch/`
Launch related files for simulation support.

## Phase 2 Deliverable Purpose
This repository supports the Phase 2 requirements by demonstrating:
- ROS 2 and Gazebo topic connectivity
- A valid Gymnasium style environment wrapper
- Defined observation and action spaces
- Successful environment reset and one step execution
- A modular reinforcement learning project structure for future training

## Notes
- Goal related features in the observation are currently placeholders for Phase 2
- Reward logic is currently minimal and used only to confirm interface functionality
- Full training and navigation performance improvements will be developed in later phases
