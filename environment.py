import time
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped

import gymnasium as gym
from gymnasium import spaces

from sensor_processing import downsample_lidar
from utils import load_config


class Ros2NavEnv(gym.Env):
    """
    Minimal Gymnasium style environment wrapper for TurtleBot3 in ROS 2 and Gazebo.

    Phase 2 goals:
    1. Confirm topic connectivity for /scan, /odom, and /cmd_vel
    2. Expose Gymnasium compatible reset() and step() methods
    3. Provide valid observation_space and action_space definitions
    4. Execute at least one successful environment step

    Notes:
        Goal related features are placeholders in Phase 2.
        The /cmd_vel topic uses geometry_msgs/msg/TwistStamped in this setup.
    """

    metadata = {"render_modes": []}

    def __init__(self, config_path="config.yaml"):
        """
        Initialize the ROS 2 navigation environment.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        super().__init__()

        self.cfg = load_config(config_path)
        topics = self.cfg["topics"]
        env_cfg = self.cfg["env"]
        reward_cfg = self.cfg.get("reward", {})

        self.scan_topic = topics["scan"]
        self.odom_topic = topics["odom"]
        self.cmd_topic = topics["cmd_vel"]

        self.lidar_beams = int(env_cfg["lidar_beams"])
        self.max_range = float(env_cfg["lidar_max_range"])
        self.control_hz = float(env_cfg["control_hz"])
        self.dt = float(env_cfg["dt"])
        self.max_steps = int(env_cfg["max_steps"])
        self.v_max = float(env_cfg["v_max"])
        self.w_max = float(env_cfg["w_max"])
        self.r_safe = float(env_cfg["r_safe"])

        self.time_penalty = float(reward_cfg.get("time_penalty", -0.5))

        self.obs_dim = self.lidar_beams + 2 + 2

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([0.0, -self.w_max], dtype=np.float32),
            high=np.array([self.v_max, self.w_max], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

        self._latest_scan = None
        self._latest_odom = None
        self._step_count = 0

        if not rclpy.ok():
            rclpy.init(args=None)

        self.node = Node("ros2_nav_env")
        self.node.create_subscription(LaserScan, self.scan_topic, self._scan_cb, 10)
        self.node.create_subscription(Odometry, self.odom_topic, self._odom_cb, 10)
        self.cmd_pub = self.node.create_publisher(TwistStamped, self.cmd_topic, 10)

        self.node.get_logger().info(
            f"Ros2NavEnv ready scan={self.scan_topic} odom={self.odom_topic} cmd_vel={self.cmd_topic}"
        )

        self._wait_for_topics()

    def _scan_cb(self, msg: LaserScan):
        """
        Store the most recent LaserScan message.

        Args:
            msg (LaserScan): Incoming lidar scan message.
        """
        self._latest_scan = msg

    def _odom_cb(self, msg: Odometry):
        """
        Store the most recent Odometry message.

        Args:
            msg (Odometry): Incoming odometry message.
        """
        self._latest_odom = msg

    def _wait_for_topics(self, timeout_sec=10.0):
        """
        Wait until both scan and odometry messages have been received.

        Args:
            timeout_sec (float): Maximum time to wait in seconds.

        Raises:
            RuntimeError: If required topics are not received within the timeout.
        """
        start = time.time()
        while (self._latest_scan is None or self._latest_odom is None) and (time.time() - start < timeout_sec):
            rclpy.spin_once(self.node, timeout_sec=0.1)

        if self._latest_scan is None or self._latest_odom is None:
            raise RuntimeError("Timed out waiting for /scan and /odom messages")

    def _publish_action(self, v, w):
        """
        Publish a velocity command to the robot.

        Args:
            v (float): Linear velocity command.
            w (float): Angular velocity command.
        """
        msg = TwistStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.twist.linear.x = float(v)
        msg.twist.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def _get_obs(self):
        """
        Build the current observation vector.

        Returns:
            np.ndarray: Observation containing normalized lidar values,
            placeholder goal features, and robot velocities.
        """
        rclpy.spin_once(self.node, timeout_sec=0.01)

        lidar = downsample_lidar(
            self._latest_scan.ranges,
            target_beams=self.lidar_beams,
            max_range=self.max_range,
        )

        goal_dist = 0.0
        goal_heading_err = 0.0

        v = float(self._latest_odom.twist.twist.linear.x)
        w = float(self._latest_odom.twist.twist.angular.z)

        obs = np.concatenate(
            [lidar, np.array([goal_dist, goal_heading_err, v, w], dtype=np.float32)],
            axis=0,
        )
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        """
        Reset the environment state.

        Args:
            seed (int | None): Optional random seed.
            options (dict | None): Optional reset arguments.

        Returns:
            tuple[np.ndarray, dict]: Initial observation and reset info.
        """
        super().reset(seed=seed)
        self._step_count = 0

        self._publish_action(0.0, 0.0)
        self._wait_for_topics()

        obs = self._get_obs()
        info = {"status": "reset_success"}
        return obs, info

    def step(self, action):
        """
        Execute one environment step.

        Args:
            action (np.ndarray): Action containing [linear_velocity, angular_velocity].

        Returns:
            tuple: Observation, reward, terminated, truncated, and info dictionary.
        """
        self._step_count += 1

        v = float(np.clip(action[0], 0.0, self.v_max))
        w = float(np.clip(action[1], -self.w_max, self.w_max))

        self._publish_action(v, w)

        end_time = time.time() + self.dt
        while time.time() < end_time:
            rclpy.spin_once(self.node, timeout_sec=0.01)

        obs = self._get_obs()

        reward = self.time_penalty
        terminated = False
        truncated = self._step_count >= self.max_steps

        info = {
            "v_cmd": v,
            "w_cmd": w,
            "step": self._step_count,
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        """
        Stop the robot and shut down the ROS 2 node cleanly.
        """
        try:
            self._publish_action(0.0, 0.0)
        except Exception:
            pass

        try:
            self.node.destroy_node()
        except Exception:
            pass

        if rclpy.ok():
            rclpy.shutdown()
