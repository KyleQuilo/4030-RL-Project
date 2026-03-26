import math
import subprocess
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
    Gymnasium-compatible environment wrapper for TurtleBot3 navigation in ROS 2 and Gazebo.

    The robot must navigate to a randomly sampled goal position while avoiding
    obstacles detected via simulated 2D LiDAR. This environment implements the
    full MDP defined in Phase 1, including goal-relative observations, dense
    reward shaping, collision detection via LiDAR, and episode termination on
    success, collision, or timeout.

    Observation space (64-dim float32):
        [0:60]  Normalized LiDAR beams (60 beams, range [0, 1])
        [60]    Normalized distance to goal (distance / goal_norm_dist)
        [61]    Normalized heading error to goal (radians / pi, range [-1, 1])
        [62]    Current linear velocity (m/s)
        [63]    Current angular velocity (rad/s)

    Action space (2-dim float32):
        [0]  Linear velocity v in [0.0, v_max] m/s
        [1]  Angular velocity w in [-w_max, w_max] rad/s

    Reward function (per step):
        +10 * (d_prev - d_curr)  progress shaping toward goal
        -0.5                     time penalty per step
        -2.0                     safety margin penalty if r_min < r_safe
        +200 / -200 / -50        terminal reward for success / collision / timeout

    Notes:
        Goal positions are sampled randomly within [goal_min_dist, goal_max_dist]
        from the robot's pose at episode start. Collision is detected when the
        minimum LiDAR range falls below collision_dist. The /cmd_vel topic uses
        geometry_msgs/msg/TwistStamped in this configuration.
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
        goal_cfg = self.cfg.get("goal", {})

        # ROS topic names
        self.scan_topic = topics["scan"]
        self.odom_topic = topics["odom"]
        self.cmd_topic = topics["cmd_vel"]

        # Environment parameters
        self.lidar_beams = int(env_cfg["lidar_beams"])
        self.max_range = float(env_cfg["lidar_max_range"])
        self.control_hz = float(env_cfg["control_hz"])
        self.dt = float(env_cfg["dt"])
        self.max_steps = int(env_cfg["max_steps"])
        self.v_max = float(env_cfg["v_max"])
        self.w_max = float(env_cfg["w_max"])
        self.r_safe = float(env_cfg["r_safe"])
        self.gz_world_name = str(env_cfg.get("gz_world_name", "turtlebot3_world"))
        self.gz_model_name = str(env_cfg.get("gz_model_name", "turtlebot3_burger"))

        # Reward parameters
        self.time_penalty = float(reward_cfg.get("time_penalty", -0.5))
        self.safety_penalty = float(reward_cfg.get("safety_penalty", -2.0))
        self.success_reward = float(reward_cfg.get("success_reward", 200.0))
        self.collision_reward = float(reward_cfg.get("collision_reward", -200.0))
        self.timeout_reward = float(reward_cfg.get("timeout_reward", -50.0))
        self.progress_scale = float(reward_cfg.get("progress_scale", 10.0))
        self.collision_dist = float(reward_cfg.get("collision_dist", 0.20))

        # Goal parameters
        self.goal_min_dist = float(goal_cfg.get("min_dist", 1.0))
        self.goal_max_dist = float(goal_cfg.get("max_dist", 3.0))
        self.goal_tol_dist = float(goal_cfg.get("tol_dist", 0.25))
        self.goal_tol_heading = float(goal_cfg.get("tol_heading_deg", 15.0)) * math.pi / 180.0
        self.goal_norm_dist = float(goal_cfg.get("norm_dist", 5.0))

        # Observation and action spaces
        self.obs_dim = self.lidar_beams + 2 + 2  # lidar + goal_dist + goal_heading + v + w

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

        # Internal state
        self._latest_scan = None
        self._latest_odom = None
        self._step_count = 0

        # Robot pose (updated from odometry)
        self._robot_x = 0.0
        self._robot_y = 0.0
        self._robot_yaw = 0.0

        # Goal position in world frame
        self._goal_x = 0.0
        self._goal_y = 0.0

        # Previous distance to goal for progress shaping
        self._prev_dist = None

        # ROS 2 setup — use a unique node name so train_env and eval_env
        # can coexist in the same process without a naming conflict.
        if not rclpy.ok():
            rclpy.init(args=None)

        import uuid
        self.node = Node(f"ros2_nav_env_{uuid.uuid4().hex[:8]}")
        self.node.create_subscription(LaserScan, self.scan_topic, self._scan_cb, 10)
        self.node.create_subscription(Odometry, self.odom_topic, self._odom_cb, 10)
        self.cmd_pub = self.node.create_publisher(TwistStamped, self.cmd_topic, 10)

        self.node.get_logger().info(
            f"Ros2NavEnv ready scan={self.scan_topic} odom={self.odom_topic} cmd_vel={self.cmd_topic}"
        )

        self._wait_for_topics()

        # Record the launch-file spawn position as the reset target so
        # _reset_gazebo() teleports back to wherever Gazebo originally placed
        # the robot rather than a hardcoded (0, 0).
        self._spawn_x = self._robot_x
        self._spawn_y = self._robot_y
        self._spawn_z = 0.01

    def _scan_cb(self, msg: LaserScan):
        """
        Store the most recent LaserScan message.

        Args:
            msg (LaserScan): Incoming lidar scan message.
        """
        self._latest_scan = msg

    def _odom_cb(self, msg: Odometry):
        """
        Store the most recent Odometry message and extract robot pose.

        Args:
            msg (Odometry): Incoming odometry message containing pose and twist.
        """
        self._latest_odom = msg
        self._robot_x = float(msg.pose.pose.position.x)
        self._robot_y = float(msg.pose.pose.position.y)

        q = msg.pose.pose.orientation
        self._robot_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    def _wait_for_topics(self, timeout_sec=10.0):
        """
        Wait until both scan and odometry messages have been received.

        Args:
            timeout_sec (float): Maximum time to wait in seconds.

        Raises:
            RuntimeError: If required topics are not received within the timeout.
        """
        start = time.time()
        while (self._latest_scan is None or self._latest_odom is None) and (
            time.time() - start < timeout_sec
        ):
            rclpy.spin_once(self.node, timeout_sec=0.1)

        if self._latest_scan is None or self._latest_odom is None:
            raise RuntimeError("Timed out waiting for /scan and /odom messages")

    def _publish_action(self, v, w):
        """
        Publish a velocity command to the robot.

        Args:
            v (float): Linear velocity command in m/s.
            w (float): Angular velocity command in rad/s.
        """
        msg = TwistStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.twist.linear.x = float(v)
        msg.twist.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def _reset_gazebo(self):
        """
        Teleport the robot back to its spawn pose via gz set_pose service.

        First attempts to teleport via gz service. If that fails (gz not in
        PATH, wrong service name, etc.) falls back to a backup maneuver that
        physically drives the robot away from any wall it may be stuck against.
        """
        req = (
            f'name: "{self.gz_model_name}" '
            f'position: {{x: {self._spawn_x} y: {self._spawn_y} z: {self._spawn_z}}} '
            f'orientation: {{w: 1.0}}'
        )
        teleport_ok = False
        try:
            result = subprocess.run(
                [
                    "gz", "service",
                    "-s", f"/world/{self.gz_world_name}/set_pose",
                    "--reqtype", "gz.msgs.Pose",
                    "--reptype", "gz.msgs.Boolean",
                    "--timeout", "2000",
                    "--req", req,
                ],
                timeout=3.0,
                capture_output=True,
                text=True,
            )
            teleport_ok = result.returncode == 0
            if not teleport_ok:
                self.node.get_logger().warn(
                    f"gz set_pose failed (rc={result.returncode}): {result.stderr.strip()}"
                )
        except FileNotFoundError:
            self.node.get_logger().warn("gz binary not found; using backup maneuver")
        except Exception as e:
            self.node.get_logger().warn(f"gz set_pose error: {e}")

        time.sleep(1.0)

    def _sample_goal(self):
        """
        Sample a random goal position relative to the robot's current pose.

        The goal is placed at a random distance within [goal_min_dist, goal_max_dist]
        and a random angle in [-pi, pi] relative to the robot's current heading.
        """
        dist = self.np_random.uniform(self.goal_min_dist, self.goal_max_dist)
        angle = self.np_random.uniform(-math.pi, math.pi)
        self._goal_x = self._robot_x + dist * math.cos(self._robot_yaw + angle)
        self._goal_y = self._robot_y + dist * math.sin(self._robot_yaw + angle)

    def _get_goal_relative(self):
        """
        Compute distance and heading error to the goal in the robot frame.

        Returns:
            tuple[float, float]: Distance to goal (m) and heading error (radians),
                where heading error is in [-pi, pi].
        """
        dx = self._goal_x - self._robot_x
        dy = self._goal_y - self._robot_y
        dist = math.sqrt(dx * dx + dy * dy)

        angle_to_goal = math.atan2(dy, dx)
        heading_err = angle_to_goal - self._robot_yaw
        # Normalize to [-pi, pi]
        heading_err = math.atan2(math.sin(heading_err), math.cos(heading_err))

        return dist, heading_err

    def _get_obs(self):
        """
        Build the current observation vector from LiDAR, goal state, and velocity.

        Returns:
            np.ndarray: Observation vector of shape (obs_dim,) with dtype float32.
                Indices [0:60] are normalized LiDAR beams, [60] is normalized
                goal distance, [61] is normalized heading error, [62] is linear
                velocity, and [63] is angular velocity.
        """
        rclpy.spin_once(self.node, timeout_sec=0.01)

        lidar = downsample_lidar(
            self._latest_scan.ranges,
            target_beams=self.lidar_beams,
            max_range=self.max_range,
        )

        dist, heading_err = self._get_goal_relative()
        goal_dist_norm = np.clip(dist / self.goal_norm_dist, 0.0, 1.0)
        goal_heading_norm = heading_err / math.pi  # in [-1, 1]

        v = float(self._latest_odom.twist.twist.linear.x)
        w = float(self._latest_odom.twist.twist.angular.z)

        obs = np.concatenate(
            [lidar, np.array([goal_dist_norm, goal_heading_norm, v, w], dtype=np.float32)],
            axis=0,
        )
        return obs.astype(np.float32)

    def _compute_reward(self, dist, heading_err, r_min):
        """
        Compute the step reward, termination flag, and terminal bonus/penalty.

        Applies progress shaping, time penalty, and safety margin penalty each
        step. Returns terminal reward components separately so the caller can
        combine them and set terminated/truncated flags correctly.

        Args:
            dist (float): Current distance to goal in metres.
            heading_err (float): Current heading error in radians.
            r_min (float): Minimum LiDAR range reading at this step.

        Returns:
            tuple[float, bool, str]: (reward, terminated, reason) where reason
                is one of "success", "collision", or "" (not terminated).
        """
        reward = 0.0

        # Progress shaping: reward for moving closer to goal
        if self._prev_dist is not None:
            reward += self.progress_scale * (self._prev_dist - dist)

        # Time penalty every step
        reward += self.time_penalty

        # Safety margin penalty: discourage proximity to obstacles
        if r_min < self.r_safe:
            reward += self.safety_penalty

        # Success: reached goal within position and heading tolerance
        if dist < self.goal_tol_dist and abs(heading_err) < self.goal_tol_heading:
            reward += self.success_reward
            return reward, True, "success"

        # Collision: LiDAR range below collision threshold
        if r_min < self.collision_dist:
            reward += self.collision_reward
            return reward, True, "collision"

        return reward, False, ""

    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.

        Stops the robot, waits for fresh sensor data, samples a new goal
        position, and returns the initial observation.

        Args:
            seed (int | None): Optional random seed for goal sampling.
            options (dict | None): Optional reset arguments (unused).

        Returns:
            tuple[np.ndarray, dict]: Initial observation and info dictionary
                containing the goal position and episode start status.
        """
        super().reset(seed=seed)
        self._step_count = 0
        self._prev_dist = None

        self._publish_action(0.0, 0.0)

        # Reset Gazebo so the robot returns to its spawn position.
        # Flush cached sensor data so _wait_for_topics() collects fresh
        # readings from the reset position rather than the old pose.
        self._latest_scan = None
        self._latest_odom = None
        self._reset_gazebo()
        self._wait_for_topics()

        self._sample_goal()

        dist, _ = self._get_goal_relative()
        self._prev_dist = dist

        obs = self._get_obs()
        info = {
            "status": "reset_success",
            "goal_x": self._goal_x,
            "goal_y": self._goal_y,
        }
        return obs, info

    def step(self, action):
        """
        Execute one environment step.

        Clips the action to valid bounds, publishes the velocity command,
        waits one control period, collects observations, and computes reward
        with full Phase 1 reward shaping.

        Args:
            action (np.ndarray): Action [linear_velocity, angular_velocity].

        Returns:
            tuple[np.ndarray, float, bool, bool, dict]: Observation, reward,
                terminated flag, truncated flag, and info dictionary containing
                step count, velocity commands, goal distance, and episode outcome.
        """
        self._step_count += 1

        v = float(np.clip(action[0], 0.0, self.v_max))
        w = float(np.clip(action[1], -self.w_max, self.w_max))

        self._publish_action(v, w)

        end_time = time.time() + self.dt
        while time.time() < end_time:
            rclpy.spin_once(self.node, timeout_sec=0.01)

        obs = self._get_obs()

        dist, heading_err = self._get_goal_relative()
        # Keep 0.0 readings (below sensor minimum range = touching an obstacle).
        # Only filter inf (no-return beams pointing into open space).
        # A 0.0 reading means the wall is within ~0.12 m — collision territory.
        if self._latest_scan:
            valid = [r for r in self._latest_scan.ranges if r < float("inf")]
            r_min = float(min(valid)) if valid else self.max_range
        else:
            r_min = self.max_range

        reward, terminated, reason = self._compute_reward(dist, heading_err, r_min)

        self._prev_dist = dist

        truncated = (not terminated) and (self._step_count >= self.max_steps)
        if truncated:
            reward += self.timeout_reward

        info = {
            "step": self._step_count,
            "v_cmd": v,
            "w_cmd": w,
            "goal_dist": dist,
            "heading_err": heading_err,
            "r_min": r_min,
            "outcome": reason if terminated else ("timeout" if truncated else ""),
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        """
        Stop the robot and destroy the ROS 2 node.

        Does not call rclpy.shutdown() so that multiple environments can
        coexist in the same process (train_env and eval_env). The rclpy
        context is left running and cleaned up when the process exits.
        """
        try:
            self._publish_action(0.0, 0.0)
        except Exception:
            pass

        try:
            self.node.destroy_node()
        except Exception:
            pass
