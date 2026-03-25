from stable_baselines3 import SAC


class SACAgent:
    """
    Stable Baselines3 SAC wrapper for the ROS 2 navigation project.

    Soft Actor-Critic (SAC) is the advanced comparison algorithm against the
    PPO baseline. SAC is an off-policy, maximum-entropy actor-critic algorithm
    that learns a stochastic policy by maximizing both expected return and
    policy entropy. Key differences from PPO:

    - Off-policy: SAC reuses past experience via a replay buffer, improving
      sample efficiency compared to PPO's on-policy rollout collection.
    - Entropy regularization: The automatic entropy coefficient (alpha) drives
      exploration throughout training, avoiding the need for explicit epsilon
      schedules.
    - Dual critics: SAC maintains two Q-networks and uses the minimum estimate
      to reduce overestimation bias (similar in spirit to Double DQN).

    This wrapper centralizes SAC configuration, training, evaluation, saving,
    and loading so the interface mirrors PPOAgent and both agents can be driven
    by the same training script.

    Attributes:
        env: Gymnasium-compatible environment used for training or evaluation.
        config (dict): SAC hyperparameters loaded from config.yaml.
        model (SAC): Stable Baselines3 SAC model instance.
    """

    def __init__(self, env, config=None):
        """
        Initialize the SAC agent and create the Stable Baselines3 model.

        All hyperparameters are read from config with sensible defaults so the
        agent can be reproduced by providing the same config.yaml.

        Args:
            env: Gymnasium-compatible environment.
            config (dict | None): Dictionary of SAC hyperparameters. Expected
                keys: policy, gamma, learning_rate, buffer_size,
                learning_starts, batch_size, tau, ent_coef, train_freq,
                gradient_steps, verbose.
        """
        self.env = env
        self.config = config or {}

        self.policy = self.config.get("policy", "MlpPolicy")
        self.gamma = self.config.get("gamma", 0.99)
        self.learning_rate = self.config.get("learning_rate", 3e-4)
        self.buffer_size = self.config.get("buffer_size", 100_000)
        self.learning_starts = self.config.get("learning_starts", 1_000)
        self.batch_size = self.config.get("batch_size", 256)
        self.tau = self.config.get("tau", 0.005)
        self.ent_coef = self.config.get("ent_coef", "auto")
        self.train_freq = self.config.get("train_freq", 1)
        self.gradient_steps = self.config.get("gradient_steps", 1)
        self.verbose = self.config.get("verbose", 1)
        self.device = self.config.get("device", "cpu")

        self.model = SAC(
            self.policy,
            self.env,
            verbose=self.verbose,
            gamma=self.gamma,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            learning_starts=self.learning_starts,
            batch_size=self.batch_size,
            tau=self.tau,
            ent_coef=self.ent_coef,
            train_freq=self.train_freq,
            gradient_steps=self.gradient_steps,
            device=self.device,
        )

    def select_action(self, observation, deterministic=False):
        """
        Select an action from the current SAC policy.

        Args:
            observation: Current environment observation.
            deterministic (bool): If True, return the mean action (exploitation).
                If False, sample from the policy distribution (exploration).

        Returns:
            np.ndarray: Action selected by the SAC policy.
        """
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def train(self, timesteps, callback=None):
        """
        Train the SAC agent for the specified number of environment timesteps.

        Args:
            timesteps (int): Total number of environment interactions to train for.
            callback: Optional SB3 callback or callback list for logging and
                checkpointing during training.
        """
        self.model.learn(total_timesteps=timesteps, callback=callback)

    def evaluate(self, observation):
        """
        Run the policy in deterministic evaluation mode (no exploration).

        Args:
            observation: Current environment observation.

        Returns:
            np.ndarray: Deterministic action for deployment or evaluation.
        """
        return self.select_action(observation, deterministic=True)

    def save(self, path="sac_results/sac_model.zip"):
        """
        Save the SAC model weights and configuration to disk.

        Args:
            path (str): Output path for the saved model zip file.
        """
        self.model.save(path)

    def load(self, path, env=None):
        """
        Load a SAC model from disk.

        Args:
            path (str): Path to the saved model zip file.
            env: Optional environment to bind to the loaded model. If None,
                the agent's current environment is used.
        """
        self.model = SAC.load(path, env=env if env is not None else self.env)
        if env is not None:
            self.env = env
