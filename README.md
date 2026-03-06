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

## ROS Packages

TurtleBot3 packages for ROS 2 Jazzy and the required simulation packages must be installed separately using `apt`.

## Setup

### 1. Source ROS 2

```bash
source /opt/ros/jazzy/setup.bash
