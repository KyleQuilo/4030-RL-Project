import math
import os
import subprocess
import time
import xml.etree.ElementTree as ET
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
from rosgraph_msgs.msg import Clock

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
        self.robot_radius = float(env_cfg.get("robot_radius", 0.105))  # TurtleBot3 Burger footprint
        # Circular arena boundary — inscribed circle of the hexagonal room.
        # arena_radius is the safe goal-centre radius (already accounts for walls).
        self.arena_radius = float(env_cfg.get("arena_radius", 1.8))
        self.gz_world_name = str(env_cfg.get("gz_world_name", "default"))
        self.gz_model_name = str(env_cfg.get("gz_model_name", "burger"))

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

        # Debug logging: tracks which episodes to log in detail
        self._episode_count = 0
        self._debug_this_episode = True   # log every episode; change to (% N == 0) once working

        # Track how far the robot moved this episode for stuck-detection
        self._episode_start_x = 0.0
        self._episode_start_y = 0.0

        # Odometry world-frame offset.
        # set_pose moves the physical robot but the wheel-encoder /odom keeps
        # accumulating from wherever it was.  We compute the offset each reset
        # as  offset = spawn - odom_raw_at_teleport  so that:
        #   world_pos = odom_raw + offset
        self._odom_raw_x = 0.0
        self._odom_raw_y = 0.0
        self._odom_offset_x = 0.0
        self._odom_offset_y = 0.0

        # Simulation time tracking — updated from /clock so step() waits for
        # exactly dt seconds of SIM time rather than wall time.  This makes
        # training consistent regardless of Gazebo real-time factor.
        self._sim_time = 0.0

        # ROS 2 setup — use a unique node name so train_env and eval_env
        # can coexist in the same process without a naming conflict.
        if not rclpy.ok():
            rclpy.init(args=None)

        import uuid
        self.node = Node(f"ros2_nav_env_{uuid.uuid4().hex[:8]}")
        self.node.create_subscription(LaserScan, self.scan_topic, self._scan_cb, 10)
        self.node.create_subscription(Odometry, self.odom_topic, self._odom_cb, 10)
        self.node.create_subscription(Clock, "/clock", self._clock_cb, 10)
        self.cmd_pub = self.node.create_publisher(TwistStamped, self.cmd_topic, 10)

        self.node.get_logger().info(
            f"Ros2NavEnv ready scan={self.scan_topic} odom={self.odom_topic} cmd_vel={self.cmd_topic}"
        )

        self._wait_for_topics()

        # Log Gazebo real-time factor once at startup so slow-sim issues are
        # immediately visible in the training log.
        self._log_gz_rtf()

        # Use config reset position if provided, otherwise fall back to the
        # position captured from odometry at startup.
        cfg_reset_x = env_cfg.get("reset_x", None)
        cfg_reset_y = env_cfg.get("reset_y", None)
        if cfg_reset_x is not None and cfg_reset_y is not None:
            self._spawn_x = float(cfg_reset_x)
            self._spawn_y = float(cfg_reset_y)
        else:
            self._spawn_x = self._robot_x
            self._spawn_y = self._robot_y
        self._spawn_z = 0.01

        self.node.get_logger().info(
            f"[ENV] Reset position: x={self._spawn_x:.3f} y={self._spawn_y:.3f}  "
            f"(odom start was x={self._robot_x:.3f} y={self._robot_y:.3f})"
        )

        # Load obstacle positions from the turtlebot3_world SDF.
        # _obstacle_boxes is populated as a side-effect of the call.
        self._obstacle_boxes: list = []
        self._obstacle_cylinders = self._load_obstacle_positions()

    # ------------------------------------------------------------------
    # Obstacle map
    # ------------------------------------------------------------------

    def _log_gz_rtf(self):
        """
        Query ``gz stats`` once and log the current Gazebo real-time factor.

        Runs at environment startup so any sim-speed issues are immediately
        visible. Non-fatal: if gz is unavailable the warning is logged and
        training continues normally.
        """
        try:
            result = subprocess.run(
                ["gz", "stats", "-d", "1"],
                capture_output=True, text=True, timeout=5.0,
            )
            rtf = None
            for line in result.stdout.splitlines():
                if "Real time factor" in line or "RTF" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        rtf = parts[-1].strip()
                        break
            if rtf:
                self.node.get_logger().info(f"[ENV] Gazebo RTF: {rtf}")
            else:
                # Fall back: show raw output on first line
                first = result.stdout.strip().splitlines()[0] if result.stdout.strip() else "(no output)"
                self.node.get_logger().info(f"[ENV] gz stats: {first}")
        except Exception as e:
            self.node.get_logger().warn(f"[ENV] Could not read gz stats: {e}")

    @staticmethod
    def _parse_pose(pose_el) -> tuple:
        """
        Parse an SDF ``<pose>x y z r p yaw</pose>`` element.

        Returns:
            tuple: (x, y, z, roll, pitch, yaw) floats, all zero if element
                is None or has no text.
        """
        if pose_el is not None and pose_el.text:
            parts = pose_el.text.strip().split()
            vals = [float(v) for v in parts]
            while len(vals) < 6:
                vals.append(0.0)
            return tuple(vals)
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def _load_obstacle_positions(self):
        """
        Parse the turtlebot3_world model SDF to extract obstacle positions.

        SDF structure (confirmed from file inspection):
          - ONE model (ros_symbol) with ONE link (symbol)
          - Collision elements are direct children of that link, named
            one_one … three_three (9 cylinders) plus head/hands/feet/body
            which use mesh geometry and are outside the navigable arena
          - Walls are a scaled .dae mesh — no parseable box geometry, so
            arena bounds stay at their config.yaml values

        Also parses any <box> collision elements found (none in this world,
        but supported for extensibility). Results stored in:
            self._obstacle_cylinders  — [x, y, radius] list (return value)
            self._obstacle_boxes      — [cx, cy, hx, hy, yaw] list

        Returns:
            list: [world_x, world_y, radius] for each cylinder obstacle.
        """
        cylinders: list = []
        self._obstacle_boxes = []

        try:
            # --- locate SDF ---
            res = subprocess.run(
                ["ros2", "pkg", "prefix", "turtlebot3_gazebo"],
                capture_output=True, text=True, timeout=5.0,
            )
            if res.returncode != 0:
                self.node.get_logger().warn("[ENV] ros2 pkg prefix turtlebot3_gazebo failed")
                return cylinders
            prefix = res.stdout.strip()
            sdf_path = os.path.join(
                prefix, "share", "turtlebot3_gazebo",
                "models", "turtlebot3_world", "model.sdf",
            )
            if not os.path.exists(sdf_path):
                self.node.get_logger().warn(f"[ENV] SDF not found: {sdf_path}")
                return cylinders

            # --- get model world-frame offset via gz model ---
            offset_x, offset_y = 0.0, 0.0
            try:
                gz_res = subprocess.run(
                    ["gz", "model", "-m", "turtlebot3_world", "-p"],
                    capture_output=True, text=True, timeout=3.0,
                )
                if gz_res.returncode == 0:
                    for line in gz_res.stdout.splitlines():
                        line = line.strip()
                        if line.startswith("x:"):
                            offset_x = float(line.split(":")[1])
                        elif line.startswith("y:"):
                            offset_y = float(line.split(":")[1])
            except Exception:
                pass

            # --- parse SDF ---
            # All collision elements live as direct named children of the
            # single 'symbol' link.  Iterate every <collision> in the file
            # and dispatch by geometry type; mesh-based ones are skipped.
            tree = ET.parse(sdf_path)
            sdf_root = tree.getroot()

            seen: set = set()
            for collision in sdf_root.iter("collision"):
                px, py, _, _, _, yaw = self._parse_pose(collision.find("pose"))
                wx, wy = offset_x + px, offset_y + py

                # Cylinder obstacle
                cyl_el = collision.find(".//cylinder")
                if cyl_el is not None:
                    r_el = cyl_el.find("radius")
                    if r_el is not None and r_el.text:
                        key = (round(wx, 3), round(wy, 3))
                        if key not in seen:
                            seen.add(key)
                            cylinders.append([wx, wy, float(r_el.text)])
                    continue

                # Box obstacle (none in this world, kept for extensibility)
                box_el = collision.find(".//box")
                if box_el is not None:
                    size_el = box_el.find("size")
                    if size_el is not None and size_el.text:
                        sx, sy, _ = [float(v) for v in size_el.text.strip().split()]
                        key = (round(wx, 3), round(wy, 3))
                        if key not in seen:
                            seen.add(key)
                            self._obstacle_boxes.append(
                                [wx, wy, sx / 2.0, sy / 2.0, yaw]
                            )
                # mesh geometry — skipped (walls/decorations outside arena)

            self.node.get_logger().info(
                f"[ENV] Obstacle map: {len(cylinders)} cylinders, "
                f"{len(self._obstacle_boxes)} boxes  "
                f"(model offset x={offset_x:.3f} y={offset_y:.3f})  |  "
                f"Arena radius={self.arena_radius:.2f}m"
            )
            self.node.get_logger().info(
                "[ENV] Cylinders: "
                + ", ".join(f"({c[0]:.2f},{c[1]:.2f}) r={c[2]:.2f}" for c in cylinders)
            )

        except Exception as e:
            self.node.get_logger().warn(f"[ENV] Could not load obstacle map: {e}")

        return cylinders

    def _goal_in_obstacle(self, gx: float, gy: float, margin: float = 0.10) -> bool:
        """
        Return True if placing the robot centre at (gx, gy) would overlap an
        obstacle cylinder, accounting for both the cylinder radius and the
        robot's own physical footprint.

        The minimum safe distance is:
            cylinder_radius + robot_radius + margin

        Args:
            gx, gy (float): Candidate position in world frame.
            margin (float): Extra safety buffer beyond physical contact (m).
        """
        min_dist = self.robot_radius + margin
        for cx, cy, cr in self._obstacle_cylinders:
            if math.hypot(gx - cx, gy - cy) < cr + min_dist:
                return True
        for cx, cy, hx, hy, yaw in self._obstacle_boxes:
            # Transform point into the box's local frame, then do AABB check.
            dx, dy = gx - cx, gy - cy
            local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
            local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)
            if abs(local_x) < hx + min_dist and abs(local_y) < hy + min_dist:
                return True
        return False

    def _goal_out_of_bounds(self, gx: float, gy: float) -> bool:
        """
        Return True if (gx, gy) is outside the navigable arena.

        Uses a conservative circular approximation of the arena's navigable
        floor. The effective limit is the inner wall face with a safety
        margin, so the robot centre must simply stay within arena_radius of
        the world origin.

        Args:
            gx, gy (float): Candidate goal position in world frame.
        """
        return math.hypot(gx, gy) > self.arena_radius

    def _clock_cb(self, msg: Clock):
        """
        Track Gazebo simulation time from /clock.

        Storing sim time lets step() wait for exactly dt seconds of simulation
        time rather than wall time, keeping each step consistent regardless of
        the Gazebo real-time factor.

        Args:
            msg (Clock): Incoming clock message from Gazebo.
        """
        self._sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9

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

        /odom integrates wheel encoders from (0,0) after each world reset.
        _odom_offset converts that to the true world-frame position so that
        all other code always works in consistent world coordinates.

        Args:
            msg (Odometry): Incoming odometry message containing pose and twist.
        """
        self._latest_odom = msg
        self._odom_raw_x = float(msg.pose.pose.position.x)
        self._odom_raw_y = float(msg.pose.pose.position.y)
        self._robot_x = self._odom_raw_x + self._odom_offset_x
        self._robot_y = self._odom_raw_y + self._odom_offset_y

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

        self.node.get_logger().info(
            f"[ENV] Topics ready: x={self._robot_x:.3f} y={self._robot_y:.3f} yaw={self._robot_yaw:.3f}"
        )

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
        Teleport the robot to its spawn position for a new episode.

        set_pose moves the physical model in Gazebo but the wheel-encoder /odom
        keeps accumulating from wherever it was.  We capture odom_raw just
        before the teleport and set:
            offset = spawn - odom_raw_at_teleport
        so that throughout the episode:
            world_pos = odom_raw + offset = spawn + (movement since reset)
        """
        # Capture raw odom position just before teleporting
        self._odom_offset_x = self._spawn_x - self._odom_raw_x
        self._odom_offset_y = self._spawn_y - self._odom_raw_y

        req = (
            f'name: "{self.gz_model_name}" '
            f'position: {{x: {self._spawn_x} y: {self._spawn_y} z: {self._spawn_z}}} '
            f'orientation: {{w: 1.0}}'
        )
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
            if result.returncode == 0:
                self.node.get_logger().info(
                    f"[EP {self._episode_count}] Reset OK -> "
                    f"world=({self._spawn_x:.3f},{self._spawn_y:.3f})"
                )
            else:
                self.node.get_logger().warn(
                    f"set_pose failed (rc={result.returncode}): {result.stderr.strip()}"
                )
        except Exception as e:
            self.node.get_logger().warn(f"set_pose error: {e}")

        # Brief pause for Gazebo to process set_pose before we read observations
        time.sleep(0.5)

    def _wait_for_pose(self, target_x, target_y, tolerance=0.15, timeout_sec=3.0):
        """
        Spin until odometry reports a position within *tolerance* metres of
        (target_x, target_y), or until timeout_sec elapses.

        Args:
            target_x, target_y (float): Expected position after teleport.
            tolerance (float): Acceptable distance from target (m).
            timeout_sec (float): Give up and continue after this many seconds.
        """
        start = time.time()
        while time.time() - start < timeout_sec:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            dx = self._robot_x - target_x
            dy = self._robot_y - target_y
            if math.sqrt(dx * dx + dy * dy) < tolerance:
                return
        self.node.get_logger().warn(
            f"[EP {self._episode_count}] _wait_for_pose timed out: "
            f"robot=({self._robot_x:.3f},{self._robot_y:.3f}) "
            f"target=({target_x:.3f},{target_y:.3f})"
        )

    def _sample_goal(self):
        """
        Sample a random goal position relative to the robot's current pose.

        The goal is placed at a random distance within [goal_min_dist, goal_max_dist]
        and a random angle in [-pi, pi] relative to the robot's current heading.
        Retries up to 100 times to avoid placing the goal inside a known
        obstacle cylinder. Falls back to the last sample if no clear position
        is found and logs a warning.
        """
        max_tries = 100
        gx, gy = self._robot_x, self._robot_y
        for attempt in range(max_tries):
            dist = self.np_random.uniform(self.goal_min_dist, self.goal_max_dist)
            angle = self.np_random.uniform(-math.pi, math.pi)
            gx = self._robot_x + dist * math.cos(self._robot_yaw + angle)
            gy = self._robot_y + dist * math.sin(self._robot_yaw + angle)
            if not self._goal_in_obstacle(gx, gy) and not self._goal_out_of_bounds(gx, gy):
                break
            if attempt == max_tries - 1:
                self.node.get_logger().warn(
                    f"[EP {self._episode_count}] _sample_goal: no clear position "
                    f"after {max_tries} tries — using last sample ({gx:.2f},{gy:.2f})"
                )
        self._goal_x = gx
        self._goal_y = gy

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
                is one of "success", "collision", "out_of_bounds", or "".
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

        # Out of bounds: robot escaped the arena
        if self._goal_out_of_bounds(self._robot_x, self._robot_y):
            reward += self.collision_reward
            return reward, True, "out_of_bounds"

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
        self._episode_count += 1
        self._debug_this_episode = True   # log every episode; change to (% N == 0) once stable

        self._publish_action(0.0, 0.0)

        # Reset Gazebo so the robot returns to its spawn position.
        # Flush cached sensor data so _wait_for_topics() collects fresh
        # readings from the reset position rather than the old pose.
        self._latest_scan = None
        self._latest_odom = None
        self._reset_gazebo()
        self._wait_for_topics()

        self._sample_goal()
        self._episode_start_x = self._robot_x
        self._episode_start_y = self._robot_y

        dist, _ = self._get_goal_relative()
        self._prev_dist = dist

        if self._debug_this_episode:
            self.node.get_logger().info(
                f"[EP {self._episode_count}] START  "
                f"robot=({self._robot_x:.3f},{self._robot_y:.3f}) yaw={self._robot_yaw:.3f}  "
                f"goal=({self._goal_x:.3f},{self._goal_y:.3f}) dist={dist:.3f}"
            )

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

        # Wait for dt seconds of SIMULATION time (not wall time).
        # This keeps each step consistent regardless of Gazebo RTF.
        # Falls back to wall time if /clock hasn't published yet.
        if self._sim_time > 0.0:
            sim_end = self._sim_time + self.dt
            while self._sim_time < sim_end:
                rclpy.spin_once(self.node, timeout_sec=0.01)
        else:
            end_time = time.time() + self.dt
            while time.time() < end_time:
                rclpy.spin_once(self.node, timeout_sec=0.01)

        obs = self._get_obs()

        dist, heading_err = self._get_goal_relative()
        # Compute minimum LiDAR range for collision detection.
        # Use numpy to uniformly handle nan, +inf, and -inf:
        #   nan  → treated as max_range (no return, open space)
        #   +inf → treated as max_range (beam beyond sensor limit)
        #   -inf → treated as max_range (invalid reading, ignore)
        #   0.0  → kept as-is (below minimum range = obstacle within ~0.12 m)
        if self._latest_scan:
            arr = np.array(self._latest_scan.ranges, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=self.max_range, posinf=self.max_range, neginf=self.max_range)
            r_min = float(np.min(arr)) if len(arr) > 0 else self.max_range
        else:
            r_min = self.max_range

        reward, terminated, reason = self._compute_reward(dist, heading_err, r_min)

        self._prev_dist = dist

        truncated = (not terminated) and (self._step_count >= self.max_steps)
        if truncated:
            reward += self.timeout_reward

        if self._debug_this_episode and self._step_count % 10 == 0:
            self.node.get_logger().info(
                f"[EP {self._episode_count}] step={self._step_count:4d}  "
                f"pos=({self._robot_x:.3f},{self._robot_y:.3f}) yaw={self._robot_yaw:.3f}  "
                f"v={v:.3f} w={w:.3f}  "
                f"goal_dist={dist:.3f} heading_err={heading_err:.3f}  "
                f"r_min={r_min:.3f}  reward={reward:.3f}"
            )

        if terminated or truncated:
            outcome = reason if terminated else "timeout"
            dist_moved = math.sqrt(
                (self._robot_x - self._episode_start_x) ** 2
                + (self._robot_y - self._episode_start_y) ** 2
            )
            if dist_moved < 0.05:
                self.node.get_logger().warn(
                    f"[EP {self._episode_count}] Robot moved only {dist_moved:.4f} m over "
                    f"{self._step_count} steps. Check: (1) /cmd_vel message type "
                    f"(Twist vs TwistStamped), (2) gz_model_name in config.yaml matches "
                    f"Gazebo model name, (3) Gazebo real-time factor is not near zero."
                )
            self.node.get_logger().info(
                f"[EP {self._episode_count}] END outcome={outcome}  "
                f"steps={self._step_count}  dist_moved={dist_moved:.3f}m  "
                f"pos=({self._robot_x:.3f},{self._robot_y:.3f})  "
                f"goal_dist={dist:.3f}  r_min={r_min:.3f}  reward={reward:.3f}  "
                f"last_cmd=v={v:.3f} w={w:.3f}"
            )

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
