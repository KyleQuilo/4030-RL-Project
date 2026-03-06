import numpy as np
import torch
from environment import Ros2NavEnv
from ppo_agent import PPOAgent
from utils import load_config


def get_device():
    """
    Select the available PyTorch device.

    Returns:
        str: "cuda", "mps", or "cpu"
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_task1_verification(env):
    """
    Run the Phase 2 Task 1 environment verification.

    This function prints the observation space, action space, device,
    reset confirmation, and one successful environment step.

    Args:
        env (Ros2NavEnv): The ROS 2 navigation environment.

    Returns:
        None
    """
    device = get_device()

    print("\n=== Phase 2 Task 1 Verification ===")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Device: {device}")

    obs, info = env.reset()
    print("Reset OK")
    print(
        f"Initial observation shape: {obs.shape}, "
        f"dtype: {obs.dtype}, "
        f"range: [{np.min(obs):.3f}, {np.max(obs):.3f}]"
    )
    print(f"Reset info: {info}")

    action = env.action_space.sample()
    print(f"Sampled action: {action}")

    obs2, reward, terminated, truncated, info = env.step(action)

    print("One environment step executed successfully.")
    print(
        f"Next observation shape: {obs2.shape}, "
        f"dtype: {obs2.dtype}, "
        f"range: [{np.min(obs2):.3f}, {np.max(obs2):.3f}]"
    )
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    print(f"Step info: {info}")


def build_agent(env, config):
    """
    Create the PPO agent wrapper for the current environment.

    Args:
        env (Ros2NavEnv): Gymnasium compatible environment.
        config (dict): Full project configuration dictionary.

    Returns:
        PPOAgent: Configured PPO agent instance.
    """
    ppo_cfg = config.get("ppo", {})
    agent = PPOAgent(env, config=ppo_cfg)
    return agent


def train_agent(agent, timesteps):
    """
    Placeholder training entry point for the PPO agent.

    Args:
        agent (PPOAgent): PPO agent wrapper.
        timesteps (int): Number of training timesteps.

    Returns:
        None
    """
    agent.train(timesteps)


def save_agent(agent, path):
    """
    Save the trained PPO agent.

    Args:
        agent (PPOAgent): PPO agent wrapper.
        path (str): Output model path.

    Returns:
        None
    """
    agent.save(path)


def evaluate_agent(agent, env):
    """
    Run one deterministic evaluation action from the current policy.

    Args:
        agent (PPOAgent): PPO agent wrapper.
        env (Ros2NavEnv): Evaluation environment.

    Returns:
        None
    """
    obs, _ = env.reset()
    action = agent.evaluate(obs)
    print(f"Evaluation action: {action}")


def main():
    """
    Main entry point for Phase 2.

    This script:
    1. Loads configuration
    2. Creates the environment
    3. Runs Task 1 verification output
    4. Instantiates the PPO agent wrapper
    5. Shows where training, saving, and evaluation occur
    """
    env = None

    try:
        config = load_config("config.yaml")
        env = Ros2NavEnv("config.yaml")

        run_task1_verification(env)

        print("\n=== Phase 2 Task 2 Skeleton Check ===")
        agent = build_agent(env, config)
        print("PPO agent instantiated successfully.")

        model_path = config.get("paths", {}).get("model_path", "ppo_results/ppo_model.zip")

        # Phase 2 note:
        # Full training is not required for Task 1 verification.
        # These lines show the expected training script structure for later phases.

        # train_agent(agent, timesteps=10000)
        # save_agent(agent, model_path)
        # evaluate_agent(agent, env)

        print("Training entry point defined.")
        print("Model save entry point defined.")
        print("Evaluation entry point defined.")
        print(f"Configured model path: {model_path}")

    except Exception as e:
        print(f"Phase 2 script failed: {e}")

    finally:
        if env is not None:
            env.close()
            print("Environment closed.")


if __name__ == "__main__":
    main()
