import numpy as np
import torch
from environment import Ros2NavEnv

def main():
    env = Ros2NavEnv("config.yaml")

    obs, info = env.reset()
    print(f"Observation shape: {obs.shape} dtype {obs.dtype} range [{np.min(obs):.3f}, {np.max(obs):.3f}]")
    print(f"Action space: {env.action_space}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("Reset OK")

    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info = env.step(action)

    print(f"Step OK reward {reward} terminated {terminated} truncated {truncated}")
    print(f"Info: {info}")

    env.close()

if __name__ == "__main__":
    main()
