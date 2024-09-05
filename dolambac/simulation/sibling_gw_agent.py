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

        nS = np.prod(env.observation_space.nvec)
        nA_gw, nA_bandit = env.action_space.nvec
        self.pi_track = []
        self.Q_gw = np.ones((nS, nA_gw), dtype=np.float32)*-100
        self.Q_bandit = np.ones(nA_bandit, dtype=np.float32)*0
        self.N_bandit = np.zeros(nA_bandit, dtype=np.int32)
        self.episode = 0

        self.gamma = gamma

        if n_episodes > 1:
            self.alphas = decay_schedule(
                init_alpha, min_alpha, alpha_decay_ratio, n_episodes
            )
            self.epsilons = decay_schedule(
                init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes
            )
        else:
            self.alphas = [init_alpha]
            self.epsilons = [init_epsilon]

        self.training_error = []
    
    def state_lin_to_multi(self, state):
        return np.unravel_index(state, self.env.observation_space.nvec, order='F')
    
    def state_multi_to_lin(self, state):
        return np.ravel_multi_index(state, self.env.observation_space.nvec, order='F')


    def custom_action(self, state):
        state_idx = self.state_multi_to_lin(state)

        # max_value = np.max(self.Q_bandit)
        # max_indices = np.where(self.Q_bandit == max_value)[0]
        # act_bandit = np.random.choice(max_indices)
        act_bandit = np.argmax(self.Q_bandit)

        P = self.env._update_P(act_bandit)
        Q_gw, V_gw, pi_gw = value_iteration(P)

        return V_gw, np.array([pi_gw(state_idx), act_bandit])

    def greedy_action(self, state):
        state_idx = self.state_multi_to_lin(state)

        act_gw = np.argmax(self.Q_gw[state_idx])
        act_bandit = np.argmax(self.Q_bandit)
        return np.array([act_gw, act_bandit])

    def random_action(self, state):
        state_idx = self.state_multi_to_lin(state)

        act_gw = np.random.randint(len(self.Q_gw[state_idx]))
        act_bandit = np.random.randint(len(self.Q_bandit))
        return np.array([act_gw, act_bandit])


    def select_action(self, state):
        """
        Returns the best action with probability (1  - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        state_idx = self.state_multi_to_lin(state)
        
        if np.random.random() > self.epsilons[self.episode]:
            # return self.greedy_action(state)
            return self.custom_action(state)[1]
        else:
            return self.random_action(state)

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

        """Updates the Q-value of an action."""
        # V_gw, _ =  self.custom_action(obs)
        # td_target = V_gw[obs_idx]
        td_target = reward[0] + self.gamma * np.max(self.Q_gw[next_obs_idx]) * (not terminated)

        td_error = td_target - self.Q_gw[obs_idx][action[0]]
        self.Q_gw[obs_idx][action[0]] += self.alphas[self.episode] * td_error

        # Find all the worlds with the same expected direction and update the bandits
        expected_direction = self.env.worlds[action[1]][action[0]]
        similar_worlds = []
        for i in range(24):
            if np.array_equal(self.env.worlds[i][action[0]], expected_direction):
                similar_worlds.append(i)

        # Update the bandits
        for w in similar_worlds:
            self.N_bandit[w] += 1
            self.Q_bandit[w] += (reward[1] - self.Q_bandit[w]) / self.N_bandit[w]
        # self.N_bandit[action[1]] += 1
        # self.Q_bandit[action[1]] += (reward[1] - self.Q_bandit[action[1]]) / self.N_bandit[action[1]]

        self.training_error.append(td_error)