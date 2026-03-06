import numpy as np
import torch
from environment import Ros2NavEnv


def get_device():
    """
    Select the available PyTorch device.

    Returns:
        str: "cuda", "mps", or "cpu"
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    """
    Run Phase 2 Task 1 verification for the ROS 2 navigation environment.

    This script:
    1. Creates the environment
    2. Prints observation space
    3. Prints action space
    4. Prints device
    5. Resets the environment
    6. Samples one action
    7. Executes one environment step
    8. Prints the returned results
    """
    env = None

    try:
        env = Ros2NavEnv("config.yaml")

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

    except Exception as e:
        print(f"Task 1 verification failed: {e}")

    finally:
        if env is not None:
            env.close()
            print("Environment closed.")


if __name__ == "__main__":
    main()
