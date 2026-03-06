"""
rollout_buffer.py

Rollout buffer for on policy algorithms such as PPO.

Phase 2 note:
Stable Baselines3 manages rollouts internally. This file is included to
document where a custom rollout buffer would be implemented if PPO were
extended or rewritten in a later phase.
"""

class RolloutBuffer:
    """
    Placeholder rollout buffer class for PPO style trajectory storage.

    Attributes:
        storage (list): In memory list of trajectory entries.
    """

    def __init__(self):
        """
        Initialize empty rollout storage.
        """
        self.storage = []

    def add(self, transition):
        """
        Add one transition to the buffer.

        Args:
            transition: A tuple or dictionary containing observation, action,
                reward, next observation, and terminal information.
        """
        self.storage.append(transition)

    def clear(self):
        """
        Remove all stored rollout data.
        """
        self.storage.clear()

    def compute_returns_advantages(self):
        """
        Placeholder method for computing PPO returns and advantages.

        Returns:
            None: Placeholder until custom PPO logic is implemented.
        """
        return None

    def __len__(self):
        """
        Return the number of stored transitions.

        Returns:
            int: Current rollout length.
        """
        return len(self.storage)
