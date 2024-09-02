from collections import defaultdict

import numpy as np
from utils import *

class SiblingGWAgent(object):
    def __init__(
        self,
        env,
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

        nS, nA = np.prod(env.observation_space.nvec), np.prod(env.action_space.nvec)
        self.pi_track = []
        self.Q = np.zeros((nS, nA), dtype=np.float64)
        self.Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
        self.episode = 0

        self.gamma = gamma
        self.alphas = decay_schedule(
            init_alpha, min_alpha, alpha_decay_ratio, n_episodes
        )
        self.epsilons = decay_schedule(
            init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes
        )

        self.training_error = []

    def select_action(self, state):
        """
        Returns the best action with probability (1  - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        state_idx = np.ravel_multi_index(state, self.env.observation_space.nvec, order='F')
        
        if np.random.random() > self.epsilons[self.episode]:
            return np.argmax(self.Q[state_idx])
        else:
            return np.random.randint(len(self.Q[state_idx]))

    def update(
        self,
        obs,
        action,
        reward,
        terminated,
        next_obs
    ):
        obs_idx = np.ravel_multi_index(obs, self.env.observation_space.nvec, order='F')
        next_obs_idx = np.ravel_multi_index(next_obs, self.env.observation_space.nvec, order='F')

        """Updates the Q-value of an action."""
        td_target = reward + self.gamma * np.max(self.Q[next_obs_idx]) * (not terminated)
        td_error = td_target - self.Q[obs_idx][action]
        self.Q[obs_idx][action] = self.Q[obs_idx][action] + self.alphas[self.episode] * td_error

        self.training_error.append(td_error)