import gymnasium as gym
import numpy as np

class SiblingGW(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(2, 2), dtype=np.float32)
        self.state = np.zeros((2, 2))
        self.state[0, 0] = 1
        self.state[1, 1] = 1

    def step(self, action):
        if action == 0:
            self.state[0, 0] = 0
            self.state[0, 1] = 1
        elif action == 1:
            self.state[0, 1] = 0
            self.state[0, 0] = 1
        elif action == 2:
            self.state[1, 0] = 0
            self.state[1, 1] = 1
        elif action == 3:
            self.state[1, 1] = 0
            self.state[1, 0] = 1
        return self.state, 0, False, {}

    def reset(self):
        self.state = np.zeros((2, 2))
        self.state[0, 0] = 1
        self.state[1, 1] = 1
        return self.state

    def render(self, mode='human'):
        print(self.state)

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def configure(self, *args, **kwargs):
        pass

    def __del__(self):
        pass

    def __str__(self):
        return "SiblingGW"

    def __repr__(self):
        return "SiblingGW"