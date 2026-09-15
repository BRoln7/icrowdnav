from gymnasium.envs.registration import register

register(
    id='social-bev-v0',
    entry_point='socbev_gym.envs:SocBevEnv'
)
