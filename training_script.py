import numpy as np
import torch
from environment import Ros2NavEnv

def main():
    env = Ros2NavEnv("config.yaml")

    print("=== Task 1 Verification ===")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    obs, info = env.reset()
    print("Reset OK")
    print(
        f"Initial observation shape: {obs.shape} "
        f"dtype {obs.dtype} "
        f"range [{np.min(obs):.3f}, {np.max(obs):.3f}]"
    )

    action = env.action_space.sample()
    print(f"Sampled action: {action}")

    obs2, reward, terminated, truncated, info = env.step(action)

    print("One environment step executed successfully.")
    print(f"Next observation shape: {obs2.shape}")
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    print(f"Info: {info}")

    env.close()

if __name__ == "__main__":
    main()
