from gymnasium.envs.registration import register

register(
    id="gridworld/SiblingGridWorld-v0",
    entry_point="gridworld.envs:SiblingGridWorldEnv",
    max_episode_steps=300,
)