import math
import random

import numpy as np
import torch
from nav_msgs.msg import OccupancyGrid


def get_random_goal(occupancy_map: OccupancyGrid, pose_cur, robot_radius):
    dist = 21.0
    while dist >= 4.5 or dist < 4.0:
        x, y, theta = get_random_pos_on_map(occupancy_map, robot_radius)
        dist = np.linalg.norm(np.array([pose_cur[0] - x, pose_cur[1] - y]))
    return x, y, theta


def get_random_pos_on_map(occupancy_map: OccupancyGrid, robot_radius):
    map_width = occupancy_map.info.width * occupancy_map.info.resolution + occupancy_map.info.origin.position.x
    map_height = occupancy_map.info.height * occupancy_map.info.resolution + occupancy_map.info.origin.position.y
    radius = robot_radius + 0.5
    x = random.uniform(0.0, map_width)
    y = random.uniform(0.0, map_height)
    while not is_pos_valid(x, y, radius, occupancy_map):
        x = random.uniform(0.0, map_width)
        y = random.uniform(0.0, map_height)
    return x, y, random.uniform(-math.pi, math.pi)


def is_pos_valid(x, y, radius, occupancy_map: OccupancyGrid):
    cell_radius = int(radius / occupancy_map.info.resolution)
    y_index = int((y - occupancy_map.info.origin.position.y) / occupancy_map.info.resolution)
    x_index = int((x - occupancy_map.info.origin.position.x) / occupancy_map.info.resolution)
    for i in range(x_index - cell_radius, x_index + cell_radius):
        for j in range(y_index - cell_radius, y_index + cell_radius):
            index = j * occupancy_map.info.width + i
            if index < 0 or index >= len(occupancy_map.data):
                return False
            if occupancy_map.data[index] != 0:
                return False
    return True


def get_depth(keypoints, depth_img):
    n_person = keypoints.shape[0]
    height, width = depth_img.shape
    conf = keypoints[:, :, -1]
    best_joint_idx = conf.argmax(dim=1)
    rows = torch.arange(n_person)
    best_uv = keypoints[rows, best_joint_idx, :2].long()
    u = torch.clamp(best_uv[:, 0], 0, width - 1)
    v = torch.clamp(best_uv[:, 1], 0, height - 1)
    depth_values = depth_img[v, u].reshape(-1, 1, 1).repeat(1, 17, 1)
    return depth_values


def pixel_to_camera(keypoints_uv, depth, k_mat):
    n_person, n_joints, _ = keypoints_uv.shape
    device = keypoints_uv.device
    conf = keypoints_uv[..., -1]
    ones = torch.ones((n_person, n_joints, 1), device=device)
    uv1 = torch.cat([keypoints_uv[..., :2], ones], dim=-1)
    k_inv = torch.inverse(k_mat.to(device))
    cam_coords = (k_inv @ uv1.reshape(-1, 3).T).T.reshape(n_person, n_joints, 3)
    cam_coords *= depth
    cam_coords *= (conf > 0.2).unsqueeze(-1)
    cam_coords = cam_coords[:, :, [2, 0, 1]]
    cam_coords[:, :, 1:] *= -1
    return cam_coords


def align_cameras(points_cam: torch.Tensor, angle_deg: float) -> torch.Tensor:
    theta = math.radians(angle_deg)
    rot = torch.tensor(
        [[math.cos(theta), -math.sin(theta), 0.0],
         [math.sin(theta), math.cos(theta), 0.0],
         [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=points_cam.device,
    )
    return torch.matmul(points_cam, rot.T)


def keypoints_preprocess(keypoints_data, agent_nums):
    if keypoints_data:
        keypoints_data = (
            keypoints_data[0]
            if len(keypoints_data) == 1
            else torch.concat([keypoints_data[0], keypoints_data[1]], dim=0)
        )
        is_zero = keypoints_data.abs().sum(dim=(1, 2)) == 0
        x_values = keypoints_data[:, :, 0]
        x_nonzero = x_values.masked_fill(x_values == 0, float("inf"))
        x_min = x_nonzero.min(dim=1).values
        x_min[is_zero] = 1e9
        keypoints_data = keypoints_data[torch.argsort(x_min)]
    else:
        keypoints_data = torch.zeros(agent_nums, 17, 3)

    if keypoints_data.shape[0] < agent_nums:
        pad = torch.zeros(
            [agent_nums - keypoints_data.shape[0], 17, 3],
            device=keypoints_data.device,
        )
        keypoints_data = torch.concat([keypoints_data, pad], dim=0)
    elif keypoints_data.shape[0] > agent_nums:
        keypoints_data = keypoints_data[:agent_nums]
    return keypoints_data


def goal_in_robot_frame(pose_cur, goal):
    yaw = pose_cur[-1]
    dx, dy = goal[0] - pose_cur[0], goal[1] - pose_cur[1]
    rotation = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    return rotation.dot(np.array([dx, dy]))
