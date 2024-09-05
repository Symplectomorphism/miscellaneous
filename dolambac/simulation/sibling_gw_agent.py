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

        P = self.env._update_P(0)
        self.pi_gw = []
        for i in range(nA_bandit):
            P = self.env._update_P(i)
            self.pi_gw.append(value_iteration(P)[2])

        self.all_actions = np.zeros((nA_bandit, nS-1), dtype=np.int32)
        for i in range(nA_bandit):
            self.all_actions[i] = np.array([self.pi_gw[i](j) for j in range(nS-1)])

        self.blacklist = set()
        self.whitelist = set(range(24))

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

    def reset(self):
        nS = np.prod(self.env.observation_space.nvec)
        nA_gw, nA_bandit = self.env.action_space.nvec
        self.Q_gw = np.ones((nS, nA_gw), dtype=np.float32)*-100
        self.Q_bandit = np.ones(nA_bandit, dtype=np.float32)*0
        self.N_bandit = np.zeros(nA_bandit, dtype=np.int32)
        self.episode = 0
        self.blacklist.clear()
        self.whitelist = set(range(24))
    
    def state_lin_to_multi(self, state):
        return np.unravel_index(state, self.env.observation_space.nvec, order='F')
    
    def state_multi_to_lin(self, state):
        return np.ravel_multi_index(state, self.env.observation_space.nvec, order='F')


    def custom_action(self, state):
        state_idx = self.state_multi_to_lin(state)

        # if self.Q_bandit[self.env._world_belief] > 0:
        #     act_bandit = self.env._world_belief[0]
        # else:
        #     # max_value = np.max(self.Q_bandit)
        #     # max_indices = np.where(self.Q_bandit == max_value)[0]
        #     # act_bandit = np.random.choice(max_indices)
        #     act_bandit = np.argmax(self.Q_bandit)
        act_bandit = np.argmax(self.Q_bandit)

        return np.array([self.pi_gw[act_bandit](state_idx), act_bandit])

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
            return self.custom_action(state)
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
        # td_target = self.env.V_gw[action[1]][obs_idx]
        td_target = reward[0] + self.gamma * np.max(self.Q_gw[next_obs_idx]) * (not terminated)

        td_error = td_target - self.Q_gw[obs_idx][action[0]]
        self.Q_gw[obs_idx][action[0]] += self.alphas[self.episode] * td_error

        # Find all the worlds with the same expected direction and update the bandits and then some...
        true_direction = self.env.action_to_direction[action[0]]
        expected_direction = self.env.worlds[action[1]][action[0]]
        good_worlds = set()
        bad_worlds = set()
        for i in range(self.env.action_space.nvec[1]):
            if reward[1] == -3:
                if (self.all_actions[i] == action[0]).any():
                    bad_worlds.add(i)
            elif reward[1] == -1:
                test_direction = self.env.worlds[i][action[0]] 
                if np.array_equal(test_direction, expected_direction):
                   bad_worlds.add(i)
            elif reward[1] > 0:
                if (self.all_actions[i] == action[0]).any():
                    good_worlds.add(i)
        
        self.blacklist.update(bad_worlds) 
        good_worlds = good_worlds.difference(self.blacklist)
        if len(good_worlds) > 0:
            self.blacklist.update(self.whitelist.difference(good_worlds))
        if len(self.blacklist) > 0:
            self.whitelist = self.whitelist.difference(self.blacklist)


        # Update the bandits
        for w in self.whitelist:
            self.N_bandit[w] += 1
            # self.Q_bandit[w] += (reward[1] - self.Q_bandit[w]) / self.N_bandit[w]
            self.Q_bandit[w] += (1 - self.Q_bandit[w]) / self.N_bandit[w]
        for w in self.blacklist:
            self.N_bandit[w] += 1
            # self.Q_bandit[w] += (reward[1] - self.Q_bandit[w]) / self.N_bandit[w]
            self.Q_bandit[w] += (-1 - self.Q_bandit[w]) / self.N_bandit[w]

        # self.N_bandit[action[1]] += 1
        # self.Q_bandit[action[1]] += (reward[1] - self.Q_bandit[action[1]]) / self.N_bandit[action[1]]

        self.training_error.append(td_error)