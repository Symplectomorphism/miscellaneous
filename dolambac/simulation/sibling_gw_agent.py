from collections import defaultdict

import numpy as np
from utils import *

class SiblingGWAgent(object):
    def __init__(
        self,
        gamma=1.0,
        init_alpha=0.5,
        min_alpha=-0.01,
        alpha_decay_ratio=0.5,
        init_epsilon=1.0,
        min_epsilon=0.1,
        epsilon_decay_ratio=0.9,
        n_episodes=3000,
    ):
        """
        Initialize an RL agent with an empty dictionary of state-action values
        (q_values).

        Args:
            the usual suspects.
        """

        self.Q = defaultdict(lambda: np.zeros(4))
        self.episode = 0

        self.gamma = gamma
        self.alphas = decay_schedule(
            init_alpha, min_alpha, alpha_decay_ratio, n_episodes
        )
        self.epsilons = decay_schedule(
            init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes
        )

        self.training_error = []

    def select_action(self, state,  epsilon):
        """
        Returns the best action with probability (1  - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        if np.random.random() > epsilon:
            return np.argmax(self.Q[state])
        else:
            return np.random.randint(len(self.Q[state]))

    def update(
        self,
        obs,
        action,
        reward,
        terminated,
        next_obs
    ):
        """Updates the Q-value of an action."""
        td_target = reward + self.gamma * np.max(self.Q[next_obs]) * (not terminated)
        td_error = td_target - self.Q[obs][action]
        self.Q[obs][action] = self.Q[obs][action] + self.alphas[self.episode] * td_error

        self.training_error.append(td_error)