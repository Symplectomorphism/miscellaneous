from itertools import permutations
import numpy as np
import copy
import pygame
from utils import value_iteration

import gymnasium as gym
from gymnasium import spaces

class SiblingGridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, P, render_mode=None, size=5):
        self.size = size        # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.gw_P = P           # Vanilla GridWorld transition matrix
        self.cur_P = copy.deepcopy(self.gw_P)
        self.V_gw = value_iteration(self.gw_P)[1]

        self.observation_space = spaces.MultiDiscrete([size, size])
        self.action_space = spaces.MultiDiscrete([4, 24])
        
        tmp = np.array(list(permutations(
            np.array([[1, 0], [0, -1], [-1, 0], [0, 1]])
        )))
        self.worlds = dict(zip(range(24), tmp))
        self._true_world_idx = 0
        
        """
        The following dictionary is the true map of abstract actions from
        `self.action_space` to the direction we will walk in if that action is
        taken. 
        """
        self.action_to_direction = dict(
            zip(range(4), self.worlds[self._true_world_idx])
        )

        self.num_moves = 0
        self.bad_moves = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        """
        If human-rendering is used, `self.window` will be a reference to the 
        window that we draw to. `self.clock` will be a clock that is used to 
        ensure that the environment is rendered at the correct framerate in 
        human-mode. They will remain `None` until human-mode is used for the 
        first time.
        """
        self.window = None
        self.clock = None

    """
    Translate the environment's state into an observation.
    Useful function to be used in the `reset` and `step` methods.
    """
    def _get_obs(self):
        return {"agent": (self._agent_location, self._world_belief), 
                "target": self._target_location
        }

    """
    Utility function to report auxiliary information returned by `reset` and `step`.
    """
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            ), 
        }

    def _update_P(self, belief):
        sigma = list(permutations(range(4)))[belief]
        P = copy.deepcopy(self.gw_P)
        for s in range(len(self.gw_P)):
            for a in range(len(self.gw_P[s])):
                P[s][a] = self.gw_P[s][sigma[a]]
        return P

    def reset(self, true_world_id=None, seed=None, options={'randomize_world': False}):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.num_moves = 0
        self.bad_moves = 0

        if options['randomize_world']:
            self._true_world_idx = np.random.randint(24)
        if true_world_id is not None:
            self._true_world_idx = true_world_id
        self.action_to_direction = dict(
            zip(range(4), self.worlds[self._true_world_idx])
        )

        # self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._agent_location = np.array([2, 2])
        self._target_location = np.array([self.size-1, self.size-1], dtype=int)
        # while np.array_equal(self._target_location, self._agent_location):
        #     self._agent_location = self.np_random.integers(
        #         0, self.size, size=2, dtype=int
        #     )
        self._world_belief = self.np_random.integers(0, 24, size=1, dtype=int)
        self.cur_P = self._update_P(self._world_belief[0])

        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation["agent"][0], info

    def step(self, action):
        # Actual direction in which the agent is going to move.
        true_direction = self.action_to_direction[action[0]]
        expected_direction = self.worlds[action[1]][action[0]]

        state_idx = np.ravel_multi_index(self._agent_location, self.observation_space.nvec, order='F')
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + true_direction, 0, self.size - 1
        )
        self._world_belief = np.array([action[1]])
        self.cur_P = self._update_P(self._world_belief[0])

        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        # terminated = terminated and np.array_equal(self._true_world, self.worlds[self._world_belief[0]])
        reward_gw = -1 if not terminated else 0

        # Reward (penalty) for getting incorrect directions
        next_state_idx = np.ravel_multi_index(self._agent_location, self.observation_space.nvec, order='F')
        reward_bandit = self.V_gw[next_state_idx] - self.V_gw[state_idx]
        if reward_bandit <= 0:
            self.bad_moves += 1
        if not np.array_equal(true_direction, expected_direction):
            reward_bandit -= 2
        reward = np.array([reward_gw, reward_bandit])

        self.num_moves += 1
        truncated = self.num_moves > 50

        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()

        return observation["agent"][0], reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size // self.size
        )   # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            )
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0, 
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window.
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            
            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else: # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()