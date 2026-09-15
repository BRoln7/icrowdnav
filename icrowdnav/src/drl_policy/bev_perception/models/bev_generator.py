import torch
import torch.nn as nn
import torch.nn.functional as F

from bev_perception.config import load_policy_config
from bev_perception.models.resnet_encoder import Encoder
from bev_perception.models.temporal_model import TemporalModel
from bev_perception.utils.geometry import (
    calculate_birds_eye_view_parameters,
    cumulative_warp_features,
    pack_sequence_dim,
    unpack_sequence_dim,
)


class BevGenerator(nn.Module):
    def __init__(self, intrinsics, extrinsics, cfg=None):
        super().__init__()
        cfg = cfg or load_policy_config()
        camera_cfg = cfg["camera"]
        bev_cfg = cfg["bev"]
        device = intrinsics.device

        self.receptive_field = bev_cfg["receptive_field"]
        self.encoder_downsample = bev_cfg["encoder_downsample"]
        self.encoder_out_channels = bev_cfg["encoder_out_channels"]
        self.image_width = camera_cfg["width"]
        self.image_height = camera_cfg["height"]
        self.depth_min = bev_cfg["depth_min"]
        self.depth_max = bev_cfg["depth_max"]
        self.depth_input_max = bev_cfg["depth_input_max"]
        self.depth_nums = bev_cfg["depth_nums"]
        self.depth_input_nums = bev_cfg["depth_input_nums"]
        self.ground_z_min = bev_cfg["ground_z_min"]
        self.blur_sigma = bev_cfg["blur_sigma"]
        self.register_buffer(
            "depth_bins",
            torch.linspace(self.depth_min, self.depth_input_max, self.depth_input_nums, device=device),
        )
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

        self.x_bound = bev_cfg["x_bound"]
        self.y_bound = bev_cfg["y_bound"]
        self.z_bound = bev_cfg["z_bound"]
        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            x_bounds=self.x_bound,
            y_bounds=self.y_bound,
            z_bounds=self.z_bound,
        )
        self.bev_resolution = nn.Parameter(bev_resolution, requires_grad=False)
        self.bev_start_position = nn.Parameter(bev_start_position, requires_grad=False)
        self.bev_dimension = nn.Parameter(bev_dimension, requires_grad=False)

        self.bev_size = (self.bev_dimension[0].item(), self.bev_dimension[1].item())
        self.spatial_extent = (self.x_bound[1], self.y_bound[1])

        self.frustum = self.create_frustum(device)
        packed_intrinsics = pack_sequence_dim(self.intrinsics.reshape(1, 1, 1, 3, 3).repeat(1, 3, 2, 1, 1))
        packed_extrinsics = pack_sequence_dim(self.extrinsics.reshape(1, 1, 2, 4, 4).repeat(1, 3, 1, 1, 1))
        self.register_buffer("points", self.get_geometry(packed_intrinsics, packed_extrinsics))

        self.encoder = Encoder()
        temporal_in_channels = self.encoder_out_channels + 6
        self.temporal_model = TemporalModel(
            temporal_in_channels,
            self.receptive_field,
            input_shape=self.bev_size,
            start_out_channels=64,
            extra_in_channels=0,
            n_spatial_layers_between_temporal_layers=0,
            use_pyramid_pooling=False,
        )

    def create_frustum(self, device):
        h, w = self.image_height, self.image_width
        downsampled_h, downsampled_w = h // self.encoder_downsample, w // self.encoder_downsample

        depth_grid = torch.linspace(self.depth_min, self.depth_max, self.depth_nums, device=device)
        depth_grid = depth_grid.view(-1, 1, 1).expand(-1, downsampled_h, downsampled_w)
        n_depth_slices = depth_grid.shape[0]

        x_grid = torch.linspace(0, w - 1, downsampled_w, dtype=torch.float, device=device)
        x_grid = x_grid.view(1, 1, downsampled_w).expand(n_depth_slices, downsampled_h, downsampled_w)
        y_grid = torch.linspace(0, h - 1, downsampled_h, dtype=torch.float, device=device)
        y_grid = y_grid.view(1, downsampled_h, 1).expand(n_depth_slices, downsampled_h, downsampled_w)

        frustum = torch.stack((x_grid, y_grid, depth_grid), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, intrinsics, extrinsics):
        """Calculate the (x, y, z) 3D position of the features."""
        rotation, translation = extrinsics[..., :3, :3], extrinsics[..., :3, 3]
        B, N, _ = translation.shape
        points = self.frustum.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:3]), 5)
        combined_transformation = torch.inverse(rotation).matmul(torch.inverse(intrinsics))
        points = combined_transformation.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += translation.view(B, N, 1, 1, 1, 3)
        return points

    def forward(self, obs_all):
        obs = obs_all["image"]
        future_egomotion = obs_all["pose"]

        obs[..., -1] = torch.clamp(obs[..., -1], max=self.depth_input_max)

        image = obs[: self.receptive_field, ..., :3].permute(0, 1, 2, 5, 3, 4).contiguous()
        depth = obs[: self.receptive_field, ..., -1].contiguous()
        future_egomotion = future_egomotion[:, :self.receptive_field].contiguous()
        egomotion_0 = future_egomotion[:, -1] - future_egomotion[:, -1]
        egomotion_1 = future_egomotion[:, -1] - future_egomotion[:, -2]
        egomotion_2 = future_egomotion[:, -2] - future_egomotion[:, -3]
        future_egomotion[:, -1] = egomotion_0
        future_egomotion[:, -2] = egomotion_1
        future_egomotion[:, -3] = egomotion_2

        x = self.calculate_birds_eye_view_features(image, depth)
        x = self.blur_bev_features(x, sigma=self.blur_sigma)

        x = cumulative_warp_features(
            x.clone(), future_egomotion,
            mode="bilinear", spatial_extent=self.spatial_extent,
        )
        b, s, c = future_egomotion.shape
        h, w = x.shape[-2:]
        future_egomotions_spatial = future_egomotion.view(b, s, c, 1, 1).expand(b, s, c, h, w)
        future_egomotions_spatial = torch.cat(
            [torch.zeros_like(future_egomotions_spatial[:, :1]),
             future_egomotions_spatial[:, :(self.receptive_field - 1)]],
            dim=1,
        )
        x = torch.cat([x, future_egomotions_spatial], dim=-3)
        states = self.temporal_model(x)
        return states[0]

    def calculate_birds_eye_view_features(self, x, d):
        b, s, n, c, h, w = x.shape
        x = pack_sequence_dim(x)
        x, d = self.encoder_forward(x, d)
        x = self.projection_to_birds_eye_view(x, d, self.points, self.depth_bins[:self.depth_nums])
        x = unpack_sequence_dim(x, b, s)
        return x

    def encoder_forward(self, x, d):
        b, n, c, h, w = x.shape
        x = x.view(b * n, c, h, w)
        x, d = self.encoder(x, d)
        x = x.view(b, n, *x.shape[1:])
        x = x.permute(0, 1, 3, 4, 2)
        return x, d

    def projection_to_birds_eye_view(self, x, depth_map, geometry, depth_bins):
        """Lift image features into a BEV grid, dropping points below the ground."""
        B, N, H, W, C = x.shape
        D = depth_bins.shape[0]
        device = x.device

        depth_map_exp = depth_map.unsqueeze(2)
        depth_bins_exp = depth_bins.view(1, 1, D, 1, 1).to(device)
        depth_indices = torch.argmin(torch.abs(depth_map_exp - depth_bins_exp), dim=2)

        valid_depth_mask = (depth_map >= depth_bins.min()) & (depth_map <= depth_bins.max())
        h_idx = torch.arange(H, device=device).view(1, 1, H, 1).expand(B, N, H, W)
        w_idx = torch.arange(W, device=device).view(1, 1, 1, W).expand(B, N, H, W)
        b_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, N, H, W)
        n_idx = torch.arange(N, device=device).view(1, N, 1, 1).expand(B, N, H, W)

        geometry_selected = geometry[b_idx, n_idx, depth_indices, h_idx, w_idx]
        coords = geometry_selected.reshape(-1, 3)
        feats = x.reshape(-1, C)
        b_ids = b_idx.reshape(-1)
        valid_mask = valid_depth_mask.reshape(-1) & (coords[:, 2] >= self.ground_z_min)

        geometry_b = ((coords - (self.bev_start_position - self.bev_resolution / 2.0)) / self.bev_resolution)
        geometry_b = geometry_b.long()
        i, j, k = geometry_b[:, 0], geometry_b[:, 1], geometry_b[:, 2]
        mask = (
            valid_mask
            & (i >= 0) & (i < self.bev_dimension[0])
            & (j >= 0) & (j < self.bev_dimension[1])
            & (k >= 0) & (k < self.bev_dimension[2])
        )
        i, j, k = i[mask], j[mask], k[mask]
        feats = feats[mask]
        b_ids = b_ids[mask]

        bev = torch.zeros(
            (B, self.bev_dimension[2], self.bev_dimension[0], self.bev_dimension[1], C),
            device=device,
        )
        bev.index_put_((b_ids, k, i, j), feats, accumulate=True)
        return bev.permute(0, 1, 4, 2, 3)[:, 0]

    def gaussian_kernel(self, kernel_size=3, sigma=0.5, device="cpu"):
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=device)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, kernel_size, kernel_size)

    def blur_bev_features(self, bev_feat, kernel_size=3, sigma=0.5):
        """Gaussian blur on BEV features. Larger sigma yields a smoother occupancy field."""
        B, N, C, H, W = bev_feat.shape
        bev_reshape = bev_feat.reshape(B * N * C, 1, H, W)
        kernel = self.gaussian_kernel(kernel_size, sigma, device=bev_feat.device)
        bev_blur = F.conv2d(bev_reshape, kernel, padding=kernel_size // 2, groups=1)
        return bev_blur.view(B, N, C, H, W)
