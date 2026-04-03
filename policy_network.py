"""
policy_network.py

Policy network notes for the archived Phase 3 run.

Both PPO and SAC use the Stable Baselines3 `MlpPolicy` interface. In the archived
run, the project did not explicitly override `policy_kwargs`, so the exact hidden
layer widths were inherited from the installed Stable Baselines3 defaults rather
than fixed directly by project code.

This means the comparison is still valid under the same archived software setup,
but future work should explicitly set matched custom architectures for both agents
to strengthen reproducibility and isolate algorithmic differences more cleanly.
"""

class PolicyNetwork:
    """
    Interface documentation for a custom PPO or SAC policy network.

    This class describes the responsibilities of a custom neural network policy
    for the ROS 2 navigation task. Both PPO and SAC currently delegate to
    Stable Baselines3 MlpPolicy. A custom network would be plugged in here and
    registered with SB3 via the policy_kwargs mechanism.

    Network input: 64-dim float32 observation vector
        [0:60]  Normalized LiDAR beams
        [60]    Normalized distance to goal
        [61]    Normalized heading error
        [62]    Linear velocity
        [63]    Angular velocity

    Network output (PPO actor): mean of 2-dim Gaussian over [v, w] action space
    Network output (SAC actor): mean and log-std of 2-dim squashed Gaussian
    """

    def __init__(self, observation_dim=64, action_dim=2, hidden_sizes=None):
        """
        Initialize policy network parameters.

        Args:
            observation_dim (int): Size of the observation vector. Matches the
                64-dim obs space defined in environment.py.
            action_dim (int): Size of the action vector (2: linear and angular
                velocity).
            hidden_sizes (list[int] | None): Hidden layer sizes for the MLP.
                Defaults to [64, 64] to match SB3 MlpPolicy defaults.
        """
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes or [64, 64]

    def forward(self, observation):
        """
        Compute policy logits or distribution parameters from an observation.

        Args:
            observation (torch.Tensor): Batch of observations, shape
                (batch_size, observation_dim).

        Returns:
            torch.Tensor | None: Action distribution parameters. None in this
                placeholder — delegated to SB3 MlpPolicy.
        """
        return None

    def get_action(self, observation, deterministic=False):
        """
        Sample or select an action from the policy.

        Args:
            observation (torch.Tensor): Input observation vector.
            deterministic (bool): If True, return the mean action (greedy).
                If False, sample from the distribution (exploration).

        Returns:
            torch.Tensor | None: Selected action. None in this placeholder —
                delegated to SB3 via PPOAgent.select_action or
                SACAgent.select_action.
        """
        return None

    def evaluate_actions(self, observations, actions):
        """
        Evaluate log-probabilities and entropy for a batch of actions.

        Used during PPO's surrogate objective computation to get log π(a|s)
        for the importance sampling ratio. For SAC, log π(a|s) is used in the
        entropy-augmented Q-value targets.

        Args:
            observations (torch.Tensor): Batch of observations, shape
                (batch_size, observation_dim).
            actions (torch.Tensor): Batch of actions, shape
                (batch_size, action_dim).

        Returns:
            tuple | None: (log_prob, entropy) tensors. None in this placeholder
                — delegated to SB3 MlpPolicy internally.
        """
        return None
