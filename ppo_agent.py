from stable_baselines3 import PPO

class PPOAgent:
    """
    Stable Baselines3 PPO wrapper.
    Phase 2 goal: provide a clean modular agent interface with docstrings.
    """

    def __init__(self, env, config=None):
        cfg = config or {}
        self.model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            gamma=cfg.get("gamma", 0.99),
            n_steps=cfg.get("n_steps", 2048),
            batch_size=cfg.get("batch_size", 64),
            learning_rate=cfg.get("learning_rate", 3e-4),
        )

    def train(self, timesteps):
        """Train the PPO agent."""
        self.model.learn(total_timesteps=timesteps)

    def save(self, path="ppo_model.zip"):
        """Save model checkpoint."""
        self.model.save(path)

    def load(self, path, env):
        """Load model checkpoint."""
        self.model = PPO.load(path, env=env)
