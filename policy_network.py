"""
policy_network.py

Policy and value function network definitions.

Phase 2 note:
Stable Baselines3 provides default policy networks for PPO. This file is
included to satisfy the required modular project structure and to document
where a custom network would be implemented in a later phase.
"""


class PolicyNetwork:
    """
    Placeholder class for a future custom PPO policy network.

    This class documents the expected responsibilities of a custom policy
    architecture for the ROS 2 navigation task, such as forward inference,
    action sampling, and value estimation.
    """

    def __init__(self, observation_dim=None, action_dim=None, hidden_sizes=None):
        """
        Initialize policy network parameters.

        Args:
            observation_dim (int | None): Size of the observation vector.
            action_dim (int | None): Size of the action vector.
            hidden_sizes (list[int] | None): Hidden layer sizes for a future MLP.
        """
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes or [128, 128]

    def forward(self, observation):
        """
        Placeholder forward pass.

        Args:
            observation: Input observation vector.

        Returns:
            None: Placeholder until a custom policy network is implemented.
        """
        return None

    def get_action(self, observation, deterministic=False):
        """
        Placeholder method for action selection.

        Args:
            observation: Input observation vector.
            deterministic (bool): Whether to use deterministic inference.

        Returns:
            None: Placeholder until a custom policy network is implemented.
        """
        return None

    def evaluate_actions(self, observations, actions):
        """
        Placeholder method for PPO policy evaluation.

        Args:
            observations: Batch of observations.
            actions: Batch of actions.

        Returns:
            None: Placeholder until a custom policy network is implemented.
        """
        return None
