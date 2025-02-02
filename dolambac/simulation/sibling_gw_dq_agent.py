from collections import defaultdict

import numpy as np
from utils import *

class SiblingGWAgent(object):
    def __init__(
        self,
        env,
        gamma=1.0,
        init_alpha=0.5,
        min_alpha=0.01,
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
        self.env = env

        nS, nA = np.prod(env.observation_space.nvec), np.prod(env.action_space.nvec)
        self.pi_track = []
        self.Q1 = np.zeros((nS, nA), dtype=np.float64)
        self.Q2 = np.zeros((nS, nA), dtype=np.float64)
        self.Q1_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
        self.Q2_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
        self.episode = 0

        self.gamma = gamma
        self.alphas = decay_schedule(
            init_alpha, min_alpha, alpha_decay_ratio, n_episodes
        )
        self.epsilons = decay_schedule(
            init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes
        )

        self.training_error = []

    def act_lin_to_multi(self, action):
        return np.unravel_index(action, self.env.action_space.nvec, order='F')

    def act_multi_to_lin(self, action):
        return np.ravel_multi_index(action, self.env.action_space.nvec, order='F')
    
    def state_lin_to_multi(self, state):
        return np.unravel_index(state, self.env.observation_space.nvec, order='F')
    
    def state_multi_to_lin(self, state):
        return np.ravel_multi_index(state, self.env.observation_space.nvec, order='F')


    def custom_action(self, state):
        state_idx = self.state_multi_to_lin(state)

        action = np.argmax(self.Q[state_idx])
        action = self.act_lin_to_multi(action)

        # P = self.env._update_P(state[-1])
        P = self.env._update_P(0)
        Q_gw, V_gw, pi_gw = value_iteration(P)
        loc_idx = np.ravel_multi_index(
            state[:-1], self.env.observation_space.nvec[:-1], order='F')

        return self.act_multi_to_lin(np.array([pi_gw(loc_idx), action[-1]]))


    def select_action(self, state):
        """
        Returns the best action with probability (1  - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        Q = 1/2 * (self.Q1 + self.Q2)
        state_idx = self.state_multi_to_lin(state)
        
        if np.random.random() > self.epsilons[self.episode]:
            return np.argmax(Q[state_idx])
            # return self.custom_action(state)
        else:
            return np.random.randint(len(self.Q1[state_idx]))

    def update(
        self,
        obs,
        action,
        reward,
        terminated,
        next_obs
    ):
        obs_idx = self.state_multi_to_lin(obs)
        next_obs_idx = self.state_multi_to_lin(next_obs)
        action_idx = self.act_multi_to_lin(action)

        if np.random.randint(2):
            argmax_Q1 = np.argmax(self.Q1[next_obs_idx])
            td_target = reward + self.gamma * self.Q2[next_obs_idx][argmax_Q1] * (not terminated)
            td_error = td_target - self.Q1[obs_idx][action_idx]
            self.Q1[obs_idx][action_idx] = self.Q1[obs_idx][action_idx] + \
                self.alphas[self.episode] * td_error
        else:
            argmax_Q2 = np.argmax(self.Q2[next_obs_idx])
            td_target = reward + self.gamma * self.Q1[next_obs_idx][argmax_Q2] * (not terminated)
            td_error = td_target - self.Q2[obs_idx][action_idx]
            self.Q2[obs_idx][action_idx] = self.Q2[obs_idx][action_idx] + \
                self.alphas[self.episode] * td_error

        self.training_error.append(td_error)