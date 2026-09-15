import torch


def pack_sequence_dim(x):
    b, s = x.shape[:2]
    return x.view(b * s, *x.shape[2:])


def unpack_sequence_dim(x, b, s):
    return x.view(b, s, *x.shape[1:])


def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
    bev_resolution = torch.tensor([row[2] for row in [x_bounds, y_bounds, z_bounds]])
    bev_start_position = torch.tensor([row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
    bev_dimension = torch.tensor(
        [(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]],
        dtype=torch.long,
    )
    return bev_resolution, bev_start_position, bev_dimension


def mat2pose_vec(matrix: torch.Tensor):
    """Convert a 4x4 pose matrix into a 6-DoF pose vector (tx, ty, tz, rx, ry, rz)."""
    rotx = torch.atan2(-matrix[..., 1, 2], matrix[..., 2, 2])
    cosy = torch.sqrt(matrix[..., 1, 2] ** 2 + matrix[..., 2, 2] ** 2)
    roty = torch.atan2(matrix[..., 0, 2], cosy)
    rotz = torch.atan2(-matrix[..., 0, 1], matrix[..., 0, 0])
    rotation = torch.stack((rotx, roty, rotz), dim=-1)
    translation = matrix[..., :3, 3]
    return torch.cat((translation, rotation), dim=-1)


def euler2mat(angle: torch.Tensor):
    """Convert euler angles to a rotation matrix. Angle is [..., 3] in radians."""
    shape = angle.shape
    angle = angle.view(-1, 3)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz, sinz = torch.cos(z), torch.sin(z)
    zeros, ones = torch.zeros_like(z), torch.ones_like(z)
    zmat = torch.stack([cosz, -sinz, zeros, sinz, cosz, zeros, zeros, zeros, ones], dim=1).view(-1, 3, 3)

    cosy, siny = torch.cos(y), torch.sin(y)
    ymat = torch.stack([cosy, zeros, siny, zeros, ones, zeros, -siny, zeros, cosy], dim=1).view(-1, 3, 3)

    cosx, sinx = torch.cos(x), torch.sin(x)
    xmat = torch.stack([ones, zeros, zeros, zeros, cosx, -sinx, zeros, sinx, cosx], dim=1).view(-1, 3, 3)

    rot_mat = xmat.bmm(ymat).bmm(zmat)
    return rot_mat.view(*shape[:-1], 3, 3)


def pose_vec2mat(vec: torch.Tensor):
    """Convert 6-DoF parameters (tx, ty, tz, rx, ry, rz) to a 4x4 transform."""
    translation = vec[..., :3].unsqueeze(-1)
    rot_mat = euler2mat(vec[..., 3:].contiguous())
    transform_mat = torch.cat([rot_mat, translation], dim=-1)
    transform_mat = torch.nn.functional.pad(transform_mat, [0, 0, 0, 1], value=0)
    transform_mat[..., 3, 3] = 1.0
    return transform_mat


def warp_features(x, flow, mode="nearest", spatial_extent=None):
    """Apply a rotation and translation to feature map x using the xy portion of a 6-DoF vector."""
    if flow is None:
        return x
    b, c, h, w = x.shape
    angle = flow[:, 5].clone()
    translation = flow[:, :2].clone()
    translation[:, 0] /= spatial_extent[0]
    translation[:, 1] /= spatial_extent[1]
    translation[:, 0] *= -1

    cos_theta = torch.cos(angle)
    sin_theta = torch.sin(angle)
    transformation = torch.stack(
        [cos_theta, -sin_theta, translation[:, 1], sin_theta, cos_theta, translation[:, 0]],
        dim=-1,
    ).view(b, 2, 3)
    grid = torch.nn.functional.affine_grid(transformation, size=x.shape, align_corners=False)
    return torch.nn.functional.grid_sample(
        x, grid.float(), mode=mode, padding_mode="zeros", align_corners=False
    )


def cumulative_warp_features(x, flow, mode="nearest", spatial_extent=None):
    """Warp a sequence of feature maps by accumulating incremental 2D flow."""
    sequence_length = x.shape[1]
    if sequence_length == 1:
        return x

    flow = pose_vec2mat(flow)
    out = [x[:, -1]]
    cum_flow = flow[:, -2]
    for t in reversed(range(sequence_length - 1)):
        out.append(warp_features(x[:, t], mat2pose_vec(cum_flow), mode=mode, spatial_extent=spatial_extent))
        cum_flow = flow[:, t - 1] @ cum_flow
    return torch.stack(out[::-1], 1)
