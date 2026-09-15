import numpy as np
from geometry_msgs.msg import PoseArray, Twist


def compute_vel_reward(
    robot_velocity: Twist,
    r_vel,
    r_arrival,
    pose_cur,
    goal,
    goal_radius,
    step_nums,
    max_iteration,
):
    dist_to_goal = np.linalg.norm(pose_cur[:2] - goal[:2])
    x_robot, y_robot, yaw = pose_cur[0], pose_cur[1], pose_cur[-1]
    dx, dy = goal[0] - x_robot, goal[1] - y_robot
    rotation = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    goal_r = rotation.dot(np.array([dx, dy]))
    dist_goal = np.linalg.norm(goal_r) + 1e-6
    goal_dir = goal_r / dist_goal
    vel = np.array([[robot_velocity.linear.x], [robot_velocity.linear.y], [0.0]])
    reward = float(np.array([[goal_dir[0], goal_dir[1], 0.0]]).dot(vel) * r_vel)

    if dist_to_goal <= goal_radius:
        return r_arrival
    if step_nums >= max_iteration:
        return -r_arrival
    return reward


def compute_dynamic_collision(
    ped_states: PoseArray,
    pose_cur,
    r_collision,
    r_scan,
    robot_radius,
    person_radius,
    train_mode,
    collision_nums,
):
    collision_flag = False
    private_flag = False
    reward = 0.0
    cn = collision_nums
    dist = np.inf
    x_robot, y_robot, yaw = pose_cur[0], pose_cur[1], pose_cur[-1]
    rotation = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])

    for ped in ped_states.poses:
        dx = ped.position.x - x_robot
        dy = ped.position.y - y_robot
        ped_r = rotation.dot(np.array([dx, dy]))
        dist_tmp = np.linalg.norm(ped_r)
        if ped_r[0] > 0 and abs(ped_r[1] / (ped_r[0] + 1e-6)) < 1.73 and dist_tmp < dist:
            dist = dist_tmp

    if dist < (person_radius + robot_radius):
        reward = r_collision
        cn += 1
        collision_flag = True
    elif dist <= 4 * robot_radius:
        reward = -r_scan * (4 * robot_radius - dist)

    if not train_mode and dist <= (person_radius + robot_radius + 0.3):
        private_flag = True
    return reward, collision_flag, private_flag, cn


def compute_smooth_reward(velocity: Twist, r_smooth):
    if abs(velocity.angular.z) >= 1.0:
        return -r_smooth * abs(velocity.angular.z)
    return 0.0


def compute_static_obs_punish(r_scan, r_collision, robot_radius, collision_nums, min_obs_dist):
    cn = collision_nums
    if 0.02 <= min_obs_dist <= 1.2 * robot_radius:
        return r_collision, cn + 1
    if min_obs_dist <= 3 * robot_radius:
        return -r_scan * (3 * robot_radius - min_obs_dist), cn
    return 0.0, cn
