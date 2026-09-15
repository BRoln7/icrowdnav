from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from bev_perception.models.bev_backbone import BevEncoder, SocialBevEncoder


class BevExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        self.bev_encoder = BevEncoder(features_dim)

    def forward(self, obs):
        states = obs["bev_features"].squeeze(1)
        goal_r = obs["goal_position_r"].squeeze(0)
        return self.bev_encoder(states, goal_r)


class SocialBevExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        self.socialbev_encoder = SocialBevEncoder(features_dim)

    def forward(self, obs):
        bev_states = obs["bev_features"].squeeze(1)
        goal_r = obs["goal_position_r"].squeeze(0)
        human_pose = obs["human_poses"]
        return self.socialbev_encoder(bev_states, goal_r, human_pose)
