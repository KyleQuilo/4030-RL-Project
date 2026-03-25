"""
rollout_buffer.py

Experience storage for on-policy (PPO) and off-policy (SAC) algorithms.

Phase 3 note:
PPO uses an on-policy rollout buffer: experience is collected for n_steps
steps, used once to compute advantages and update the policy, then discarded.
SB3 manages this internally via its RolloutBuffer class (capacity = n_steps).

SAC uses an off-policy replay buffer: all transitions are stored and randomly
sampled for gradient updates, enabling much higher sample efficiency than PPO.
SB3 manages this internally via its ReplayBuffer class (capacity = buffer_size,
configurable in config.yaml under sac.buffer_size).

The RolloutBuffer class below documents the interface a custom buffer would
implement. The key algorithmic difference between the two buffer types is what
drives PPO vs SAC's sample efficiency tradeoff — a central part of the
comparative analysis in Task 3.

Buffer comparison:
    PPO RolloutBuffer:  On-policy, fixed size (n_steps), discarded after update.
                        Advantage: simpler, stable; Disadvantage: sample-inefficient.
    SAC ReplayBuffer:   Off-policy, large circular buffer (100k+ transitions).
                        Advantage: reuses past data; Disadvantage: requires careful
                        hyperparameter tuning (tau, learning_starts, batch_size).
"""


class RolloutBuffer:
    """
    Interface documentation for a custom PPO rollout buffer.

    This class describes the storage responsibilities for on-policy trajectory
    data. In Phase 3 this is fulfilled by Stable Baselines3's internal
    RolloutBuffer (for PPO) and ReplayBuffer (for SAC). A custom implementation
    would be registered here and passed to the agent constructor.

    Attributes:
        storage (list): In-memory list of trajectory transition entries.
    """

    def __init__(self):
        """
        Initialize empty rollout storage.
        """
        self.storage = []

    def add(self, transition):
        """
        Add one transition to the buffer.

        For PPO, a transition is (obs, action, reward, done, value, log_prob).
        For SAC, a transition is (obs, action, reward, next_obs, done).

        Args:
            transition: A tuple or dictionary containing the transition data.
                The exact format depends on the algorithm.
        """
        self.storage.append(transition)

    def clear(self):
        """
        Remove all stored rollout data.

        Called by PPO after each policy update. SAC's replay buffer uses a
        circular overwrite strategy instead and does not call clear().
        """
        self.storage.clear()

    def compute_returns_advantages(self, gamma=0.99, gae_lambda=0.95):
        """
        Compute discounted returns and Generalized Advantage Estimates (GAE).

        PPO uses GAE(lambda) to reduce variance in advantage estimates while
        trading off bias. The advantage A(s,a) = Q(s,a) - V(s) is estimated
        as a weighted sum of TD residuals:
            delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            A_t = sum_{k=0}^{T-t} (gamma * lambda)^k * delta_{t+k}

        SAC does not use this method — it computes soft Q-value targets directly.

        Args:
            gamma (float): Discount factor. Set to 0.99 in config.yaml.
            gae_lambda (float): GAE smoothing parameter. SB3 PPO default is 0.95.

        Returns:
            None: Delegated to SB3 RolloutBuffer.compute_returns_and_advantage()
                in the current implementation.
        """
        return None

    def __len__(self):
        """
        Return the number of stored transitions.

        Returns:
            int: Current buffer length.
        """
        return len(self.storage)
