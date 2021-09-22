from gym.envs.registration import register

register(
    id='CustomWalker2d-v0',
    entry_point='walker.walker2d_v3:Walker2dEnv',
    max_episode_steps=1000,
)
register(
    id='CustomWalker2dFeasibility-v0',
    entry_point='walker.walker2d_v3:Walker2dFeasibilityEnv',
    max_episode_steps=1000,
)

