from stable_baselines3 import PPO


class PPOAgent:
    """
    Stable Baselines3 PPO wrapper for the ROS 2 navigation project.

    This class provides a clean modular interface for Phase 2 while using an
    external PPO implementation. It centralizes PPO configuration, training,
    action selection, saving, loading, and evaluation hooks so the overall
    project structure matches the required skeleton.

    Attributes:
        env: Gymnasium compatible environment used for training or evaluation.
        config (dict): PPO hyperparameters loaded from config.yaml.
        model (PPO): Stable Baselines3 PPO model instance.
    """

    def __init__(self, env, config=None):
        """
        Initialize the PPO agent and create the Stable Baselines3 model.

        Args:
            env: Gymnasium compatible environment.
            config (dict | None): Dictionary of PPO hyperparameters.
        """
        self.env = env
        self.config = config or {}

        self.policy = self.config.get("policy", "MlpPolicy")
        self.gamma = self.config.get("gamma", 0.99)
        self.n_steps = self.config.get("n_steps", 2048)
        self.batch_size = self.config.get("batch_size", 64)
        self.learning_rate = self.config.get("learning_rate", 3e-4)
        self.ent_coef = self.config.get("ent_coef", 0.0)
        self.clip_range = self.config.get("clip_range", 0.2)
        self.verbose = self.config.get("verbose", 1)

        self.model = PPO(
            self.policy,
            self.env,
            verbose=self.verbose,
            gamma=self.gamma,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            ent_coef=self.ent_coef,
            clip_range=self.clip_range,
        )

    def select_action(self, observation, deterministic=False):
        """
        Select an action from the current policy.

        Args:
            observation: Current environment observation.
            deterministic (bool): If True, use exploitation mode. If False,
                allow the policy's stochastic behavior for training.

        Returns:
            action: Action selected by the PPO policy.
        """
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def train(self, timesteps):
        """
        Train the PPO agent.

        Args:
            timesteps (int): Total number of training timesteps.
        """
        self.model.learn(total_timesteps=timesteps)

    def update(self):
        """
        Placeholder update method for skeleton completeness.

        Stable Baselines3 manages PPO updates internally inside learn().
        This method is included to document the expected learning
        responsibility in the modular project structure.
        """
        pass

    def evaluate(self, observation):
        """
        Run the policy in deterministic evaluation mode.

        Args:
            observation: Current environment observation.

        Returns:
            action: Deterministic action for deployment or evaluation.
        """
        return self.select_action(observation, deterministic=True)

    def save(self, path="ppo_model.zip"):
        """
        Save the PPO model to disk.

        Args:
            path (str): Output path for the saved model.
        """
        self.model.save(path)

    def load(self, path, env=None):
        """
        Load a PPO model from disk.

        Args:
            path (str): Path to the saved model.
            env: Optional environment to bind to the loaded model.
        """
        self.model = PPO.load(path, env=env if env is not None else self.env)
        if env is not None:
            self.env = env
