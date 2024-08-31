import gymnasium as gym

class InvertedReward(gym.RewardWrapper):
    def __init__(self, env, render_mode=None):
        super().__init__(env)
        self.r_min = -1
        self.r_max = 0

    def reward(self, reward):
        if reward == 1:
            return self.r_max
        else:
            return self.r_min