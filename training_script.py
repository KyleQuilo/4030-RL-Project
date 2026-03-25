"""
training_script.py

Phase 3 training entry point for PPO (base) and SAC (advanced) agents.

Usage:
    Train PPO only:   python3 training_script.py --algo ppo
    Train SAC only:   python3 training_script.py --algo sac
    Train both:       python3 training_script.py --algo both
    Evaluate only:    python3 training_script.py --algo ppo --eval-only

Both agents are trained on the same Ros2NavEnv environment with identical
hyperparameters controlled via config.yaml. Per-episode metrics (reward,
length, loss, entropy parameter) are saved to CSV files in the respective
results directories for offline plotting with plot_results.py.
"""

import argparse
import csv
import os
import random

import numpy as np
import torch

from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

from environment import Ros2NavEnv
from ppo_agent import PPOAgent
from sac_agent import SACAgent
from utils import load_config


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device():
    """
    Select the best available PyTorch compute device.

    Returns:
        str: "cuda", "mps", or "cpu".
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Metrics callback
# ---------------------------------------------------------------------------

class MetricsCallback(BaseCallback):
    """
    SB3 callback that writes per-episode training metrics to a CSV file.

    Captures episode reward, episode length, training loss, and the current
    exploration parameter (entropy loss for PPO, entropy coefficient for SAC)
    from the SB3 internal logger after each episode completes. Requires the
    environment to be wrapped in stable_baselines3.common.monitor.Monitor so
    that episode info dicts contain the "episode" key.

    Attributes:
        csv_path (str): Destination path for the metrics CSV file.
        algorithm (str): "ppo" or "sac", selects which logger keys to read.
        _episode (int): Running episode counter.
        _file: Open CSV file handle.
        _writer: csv.writer instance for the open file.
    """

    def __init__(self, csv_path, algorithm="ppo", verbose=0):
        """
        Initialize the metrics callback.

        Args:
            csv_path (str): Path to write the CSV metrics file.
            algorithm (str): "ppo" or "sac". Controls which SB3 logger keys
                are used for loss and entropy parameter columns.
            verbose (int): Verbosity level passed to BaseCallback.
        """
        super().__init__(verbose)
        self.csv_path = csv_path
        self.algorithm = algorithm.lower()
        self._episode = 0
        self._file = None
        self._writer = None

    def _on_training_start(self):
        """Open the CSV file and write the header row."""
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        self._file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["episode", "reward", "length", "loss", "entropy_param"])

    def _on_step(self):
        """
        Check for completed episodes and log their metrics.

        Returns:
            bool: Always True to continue training.
        """
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue

            self._episode += 1
            ep_reward = float(info["episode"]["r"])
            ep_length = int(info["episode"]["l"])

            logger_vals = self.model.logger.name_to_value

            if self.algorithm == "ppo":
                loss = logger_vals.get(
                    "train/policy_gradient_loss",
                    logger_vals.get("train/loss", float("nan")),
                )
                entropy_param = logger_vals.get("train/entropy_loss", float("nan"))
            else:  # sac
                loss = logger_vals.get(
                    "train/actor_loss",
                    logger_vals.get("train/loss", float("nan")),
                )
                entropy_param = logger_vals.get("train/ent_coef", float("nan"))

            self._writer.writerow([self._episode, ep_reward, ep_length, loss, entropy_param])
            self._file.flush()

        return True

    def _on_training_end(self):
        """Close the CSV file when training finishes."""
        if self._file is not None:
            self._file.close()


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(config_path="config.yaml", monitor_path=None):
    """
    Create and optionally Monitor-wrap a Ros2NavEnv instance.

    Args:
        config_path (str): Path to the YAML configuration file.
        monitor_path (str | None): Directory to write Monitor episode logs.
            If None the env is returned unwrapped.

    Returns:
        Ros2NavEnv or Monitor: Configured environment instance.
    """
    env = Ros2NavEnv(config_path)
    if monitor_path is not None:
        os.makedirs(monitor_path, exist_ok=True)
        env = Monitor(env, monitor_path)
    return env


# ---------------------------------------------------------------------------
# Seed utilities
# ---------------------------------------------------------------------------

def set_global_seed(seed):
    """
    Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed (int): Seed value. Must be identical across runs to reproduce
            training results. Documented in config.yaml under training.seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_agent(agent, env, n_episodes=50):
    """
    Evaluate a trained agent in pure exploitation mode.

    Runs the agent deterministically for n_episodes episodes and reports the
    mean and standard deviation of episodic returns. Exploration is disabled
    (deterministic=True) to reflect deployment performance.

    Args:
        agent (PPOAgent | SACAgent): Trained agent wrapper.
        env (Ros2NavEnv): Evaluation environment (not Monitor-wrapped).
        n_episodes (int): Number of evaluation episodes to run.

    Returns:
        tuple[float, float]: Mean and standard deviation of episodic rewards.
    """
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.evaluate(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        print(f"  Eval episode {ep + 1}/{n_episodes}: reward={ep_reward:.2f}")

    mean_r = float(np.mean(rewards))
    std_r = float(np.std(rewards))
    print(f"\nEvaluation complete: mean={mean_r:.2f}  std={std_r:.2f}  n={n_episodes}")
    return mean_r, std_r


# ---------------------------------------------------------------------------
# PPO training
# ---------------------------------------------------------------------------

def train_ppo(config, config_path="config.yaml"):
    """
    Train the PPO baseline agent and save metrics, checkpoints, and the
    final model to ppo_results/.

    Training parameters are read from config.yaml under the 'ppo' and
    'training' sections. Per-episode metrics are saved to
    ppo_results/metrics.csv for offline analysis with plot_results.py.

    Args:
        config (dict): Full configuration dictionary from config.yaml.
        config_path (str): Path to config file passed to make_env.
    """
    training_cfg = config.get("training", {})
    paths_cfg = config.get("paths", {})

    timesteps = int(training_cfg.get("ppo_timesteps", 500_000))
    eval_freq = int(training_cfg.get("eval_freq", 10_000))
    n_eval_episodes = int(training_cfg.get("n_eval_episodes", 20))
    checkpoint_freq = int(training_cfg.get("checkpoint_freq", 50_000))
    seed = int(training_cfg.get("seed", 42))

    log_dir = paths_cfg.get("ppo_log_dir", "ppo_results/")
    model_path = paths_cfg.get("ppo_model_path", "ppo_results/ppo_model.zip")
    metrics_csv = paths_cfg.get("ppo_metrics_csv", "ppo_results/metrics.csv")

    os.makedirs(log_dir, exist_ok=True)
    set_global_seed(seed)

    device = training_cfg.get("device", "cpu")
    print("\n=== Training PPO (base algorithm) ===")
    print(f"Timesteps: {timesteps}  |  Seed: {seed}  |  Device: {device}")

    train_env = make_env(config_path, monitor_path=log_dir)
    eval_env = make_env(config_path)

    ppo_cfg = {**config.get("ppo", {}), "device": device}
    agent = PPOAgent(train_env, config=ppo_cfg)
    agent.model.set_random_seed(seed)

    callbacks = [
        MetricsCallback(csv_path=metrics_csv, algorithm="ppo"),
        CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=log_dir,
            name_prefix="ppo_checkpoint",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=log_dir,
            log_path=log_dir,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
        ),
    ]

    agent.train(timesteps=timesteps, callback=callbacks)
    agent.save(model_path)
    print(f"PPO model saved to {model_path}")

    train_env.close()
    eval_env.close()
    return agent


# ---------------------------------------------------------------------------
# SAC training
# ---------------------------------------------------------------------------

def train_sac(config, config_path="config.yaml"):
    """
    Train the SAC advanced agent and save metrics, checkpoints, and the
    final model to sac_results/.

    Training parameters are read from config.yaml under the 'sac' and
    'training' sections. Per-episode metrics are saved to
    sac_results/metrics.csv for offline analysis with plot_results.py.

    Args:
        config (dict): Full configuration dictionary from config.yaml.
        config_path (str): Path to config file passed to make_env.
    """
    training_cfg = config.get("training", {})
    paths_cfg = config.get("paths", {})

    timesteps = int(training_cfg.get("sac_timesteps", 500_000))
    eval_freq = int(training_cfg.get("eval_freq", 10_000))
    n_eval_episodes = int(training_cfg.get("n_eval_episodes", 20))
    checkpoint_freq = int(training_cfg.get("checkpoint_freq", 50_000))
    seed = int(training_cfg.get("seed", 42))

    log_dir = paths_cfg.get("sac_log_dir", "sac_results/")
    model_path = paths_cfg.get("sac_model_path", "sac_results/sac_model.zip")
    metrics_csv = paths_cfg.get("sac_metrics_csv", "sac_results/metrics.csv")

    os.makedirs(log_dir, exist_ok=True)
    set_global_seed(seed)

    device = training_cfg.get("device", "cpu")
    print("\n=== Training SAC (advanced algorithm) ===")
    print(f"Timesteps: {timesteps}  |  Seed: {seed}  |  Device: {device}")

    train_env = make_env(config_path, monitor_path=log_dir)
    eval_env = make_env(config_path)

    sac_cfg = {**config.get("sac", {}), "device": device}
    agent = SACAgent(train_env, config=sac_cfg)
    agent.model.set_random_seed(seed)

    callbacks = [
        MetricsCallback(csv_path=metrics_csv, algorithm="sac"),
        CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=log_dir,
            name_prefix="sac_checkpoint",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=log_dir,
            log_path=log_dir,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
        ),
    ]

    agent.train(timesteps=timesteps, callback=callbacks)
    agent.save(model_path)
    print(f"SAC model saved to {model_path}")

    train_env.close()
    eval_env.close()
    return agent


# ---------------------------------------------------------------------------
# Phase 2 environment verification (preserved for reference)
# ---------------------------------------------------------------------------

def run_phase2_verification(env):
    """
    Run the Phase 2 environment API sanity check.

    Prints observation space, action space, device, one reset, and one step.
    Retained from Phase 2 for reference; not required for Phase 3 training.

    Args:
        env (Ros2NavEnv): The ROS 2 navigation environment.
    """
    device = get_device()
    print("\n=== Environment Verification ===")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}")
    print(f"Device:            {device}")

    obs, info = env.reset()
    print(f"Reset OK  obs shape={obs.shape}  range=[{obs.min():.3f}, {obs.max():.3f}]")
    print(f"Reset info: {info}")

    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info = env.step(action)
    print(f"Step OK   reward={reward:.3f}  terminated={terminated}  truncated={truncated}")
    print(f"Step info: {info}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    """
    Parse command-line arguments for algorithm selection and mode.

    Returns:
        argparse.Namespace: Parsed arguments with fields algo and eval_only.
    """
    parser = argparse.ArgumentParser(description="Phase 3 training script: PPO vs SAC")
    parser.add_argument(
        "--algo",
        choices=["ppo", "sac", "both"],
        default="both",
        help="Which algorithm(s) to train (default: both)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and run evaluation on saved models",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes (default: 50)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    return parser.parse_args()


def main():
    """
    Main entry point for Phase 3 training and evaluation.

    Reads arguments from the CLI, loads config.yaml, and runs the requested
    combination of PPO and SAC training or evaluation. Both algorithms write
    metrics to their respective results directories.
    """
    args = parse_args()
    config = load_config(args.config)

    if args.eval_only:
        # Evaluation-only mode: load saved models and report performance
        eval_env = make_env(args.config)
        paths_cfg = config.get("paths", {})
        device = config.get("training", {}).get("device", "cpu")

        if args.algo in ("ppo", "both"):
            ppo_cfg = {**config.get("ppo", {}), "device": device}
            agent = PPOAgent(eval_env, config=ppo_cfg)
            model_path = paths_cfg.get("ppo_model_path", "ppo_results/ppo_model.zip")
            agent.load(model_path)
            print(f"\nLoaded PPO model from {model_path}")
            evaluate_agent(agent, eval_env, n_episodes=args.n_eval_episodes)

        if args.algo in ("sac", "both"):
            sac_cfg = {**config.get("sac", {}), "device": device}
            agent = SACAgent(eval_env, config=sac_cfg)
            model_path = paths_cfg.get("sac_model_path", "sac_results/sac_model.zip")
            agent.load(model_path)
            print(f"\nLoaded SAC model from {model_path}")
            evaluate_agent(agent, eval_env, n_episodes=args.n_eval_episodes)

        eval_env.close()
        return

    # Training mode
    if args.algo in ("ppo", "both"):
        train_ppo(config, config_path=args.config)

    if args.algo in ("sac", "both"):
        train_sac(config, config_path=args.config)

    print("\nTraining complete. Run plot_results.py to generate comparison plots.")


if __name__ == "__main__":
    main()
