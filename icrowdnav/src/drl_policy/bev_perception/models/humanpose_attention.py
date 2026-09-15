import torch
import torch.nn as nn


def get_valid_person(x: torch.tensor):
    B, N, _, _ = x.shape
    x = x.reshape(B, N, -1)
    mask = (x.abs().sum(dim=-1) == 0)
    return ~mask


class PoseEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        B, N, J, C = x.shape
        x = x.view(B * N * J, C)
        out = self.mlp(x)
        return out.view(B, N, J, -1)


class RobotStateEncoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.mlp(x).unsqueeze(1)


class SpatialSelfAttention(nn.Module):
    def __init__(self, hidden_size=128, num_heads=8, ff_dim=256):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_size),
        )
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.ln1(x + attn_out)
        return self.ln2(x + self.ffn(x))


class SpatialCrossAttention(nn.Module):
    def __init__(self, hidden_size=128, num_heads=8, ff_dim=256):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_size),
        )
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, query_vec, context_vecs, key_padding_mask=None):
        attn_out, _ = self.attn(query_vec, context_vecs, context_vecs, key_padding_mask=key_padding_mask)
        x = self.ln1(query_vec + attn_out)
        return self.ln2(x + self.ffn(x))


class SpatialPoseTransformer(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.pose_encoder = PoseEncoder()
        self.spatial_attn = SpatialSelfAttention()
        self.robot_encoder = RobotStateEncoder()
        self.cross_attn = SpatialCrossAttention()

    def forward(self, pose, robot_state, person_mask=None):
        B, N, J, _ = pose.shape
        robot_state = robot_state.reshape(-1, 5)
        pose_feat = self.pose_encoder(pose)
        pose_feat = pose_feat.view(B * N, J, -1)
        pose_feat = self.spatial_attn(pose_feat)
        person_feat = pose_feat.view(B, N, J, -1).mean(dim=2)
        robot_feat = self.robot_encoder(robot_state)
        key_padding_mask = ~person_mask if person_mask is not None else None
        key_padding_mask[..., 0] = False
        fused_feat = self.cross_attn(robot_feat, person_feat, key_padding_mask=key_padding_mask)
        return fused_feat.squeeze(1)
