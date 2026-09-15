#!/usr/bin/env python
"""Gymnasium environment for iCrowdNav: stereo RGB-D crowd navigation over ROS."""

import itertools
import math
import time
from collections import deque

import numpy as np
import rospy
import torch
from geometry_msgs.msg import Point, PoseArray, PoseStamped, PoseWithCovarianceStamped, Twist
from gymnasium import Env, spaces
from gymnasium.utils import seeding
from nav_msgs.msg import OccupancyGrid, Odometry
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image
from tf2_msgs.msg import TFMessage
from ultralytics import YOLO
from visualization_msgs.msg import Marker, MarkerArray

from bev_perception.bev_perception import BevGenerator
from bev_perception.config import build_camera_matrices, load_policy_config
from mdp.reward import (
    compute_dynamic_collision,
    compute_smooth_reward,
    compute_static_obs_punish,
    compute_vel_reward,
)
from mdp.utils import (
    align_cameras,
    get_depth,
    get_random_goal,
    goal_in_robot_frame,
    keypoints_preprocess,
    pixel_to_camera,
)


class SocBevEnv(Env):
    """Stereo RGB-D crowd navigation. Observation is BEV + robot state (+ human poses)."""

    metadata = {"render_modes": []}

    def __init__(self, train_mode=True, social_mode=False, quality_analysis=False):
        super().__init__()
        rospy.loginfo("Initializing SocBevEnv")
        self.np_random, _ = seeding.np_random(None)

        self.train_mode = train_mode
        self.social_mode = social_mode
        self.quality_analysis = quality_analysis
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cfg = load_policy_config()
        env_cfg = self.cfg["env"]
        bev_cfg = self.cfg["bev"]
        cam_cfg = self.cfg["camera"]

        self.RECEPTIVE_FIELD = bev_cfg["receptive_field"]
        self.AGENT_NUMS = env_cfg["agent_nums"]
        self.GOAL_RADIUS = env_cfg["goal_radius"]
        self.ROBOT_RADIUS = env_cfg["robot_radius"]
        self.PERSON_RADIUS = env_cfg["person_radius"]
        self.MAX_STEPS = env_cfg["max_steps"]
        self.MAX_COLLISIONS = env_cfg["max_collisions"]
        self.CONTROL_HZ = env_cfg["control_hz"]
        self.LINEAR_VELOCITIES = env_cfg["linear_velocities"]
        self.TEST_EPISODES = env_cfg["test_episodes"]
        self.fov_deg = env_cfg["fov_deg"]
        self.collision_pause_s = env_cfg["collision_pause_s"]
        self.depth_crop = cam_cfg["depth_crop"]
        self.reward_weights = env_cfg["reward"]
        self.yolo_weights = env_cfg["yolo_weights"]
        bev_h = int(round((bev_cfg["x_bound"][1] - bev_cfg["x_bound"][0]) / bev_cfg["x_bound"][2]))
        bev_w = int(round((bev_cfg["y_bound"][1] - bev_cfg["y_bound"][0]) / bev_cfg["y_bound"][2]))

        self.step_nums = 0
        self.collision_times = 0
        self.collision_person = False
        self.ped_states = PoseArray()
        self.cur_vel = Twist()
        self.map = OccupancyGrid()
        self.goal = np.array(env_cfg["default_goal"], dtype=np.float32)
        self.reset_error = Point()
        self.pose_cur = None

        self.rgb_l_cur = None
        self.rgb_r_cur = None
        self.depth_l_cur = None
        self.depth_r_cur = None
        self.sensors_buffer = deque(maxlen=self.RECEPTIVE_FIELD)
        self.poses_buffer = deque(maxlen=self.RECEPTIVE_FIELD)

        self.actions = list(
            itertools.product(
                self.LINEAR_VELOCITIES,
                np.linspace(
                    env_cfg["angular_velocity_min"],
                    env_cfg["angular_velocity_max"],
                    env_cfg["angular_velocity_bins"],
                ).tolist(),
            )
        )
        self.action_space = spaces.Discrete(len(self.actions))
        obs_spaces = {
            "bev_features": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1, bev_cfg["encoder_out_channels"], bev_h, bev_w),
                dtype=np.float32,
            ),
            "goal_position_r": spaces.Box(
                low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
            ),
        }
        if self.social_mode:
            obs_spaces["human_poses"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.AGENT_NUMS, 17, 3),
                dtype=np.float32,
            )
        self.observation_space = spaces.Dict(obs_spaces)

        self._init_cameras()
        self._init_ros()
        if self.social_mode:
            self.joints_predictor = YOLO(self.yolo_weights)

        self._init_eval_stats()
        self._wait_for_sensors()

    def _init_cameras(self):
        cam_cfg = self.cfg["camera"]
        self.camera_rotation = cam_cfg["yaw_offset_deg"]
        self.intrinsics, self.extrinsics = build_camera_matrices(cam_cfg, self.device)
        self.bev_generator = BevGenerator(self.intrinsics, self.extrinsics, self.cfg).to(self.device)

    def _init_ros(self):
        rospy.Subscriber("/rgb_left", Image, self._rgb_left_cb)
        rospy.Subscriber("/rgb_right", Image, self._rgb_right_cb)
        rospy.Subscriber("/depth_left", Image, self._depth_left_cb)
        rospy.Subscriber("/depth_right", Image, self._depth_right_cb)
        rospy.Subscriber("/tf_pose", TFMessage, self._tf_cb)
        rospy.Subscriber("/odom", Odometry, self._odom_cb)
        rospy.Subscriber("/map", OccupancyGrid, self._map_cb)
        rospy.Subscriber("/isaacsim/isaacsim_error", Point, self._reset_error_cb)
        rospy.Subscriber("/isaacsim/persons_states", PoseArray, self._ped_cb)

        self.pose_pub = rospy.Publisher("/isaacsim_pose", PoseWithCovarianceStamped, queue_size=10)
        self.vel_cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.goal_pub = rospy.Publisher("/socialbev/goal", MarkerArray, queue_size=10)
        self.set_robot_state_pub = rospy.Publisher("/isaacsim/set_robot_state", Point, queue_size=10)

        if not self.train_mode:
            rospy.Subscriber("/move_base_simple/goal", PoseStamped, self._goal_cb)
            self.movebase_goal_pub = rospy.Publisher(
                "/move_base_simple/goal", PoseStamped, queue_size=10
            )

    def _init_eval_stats(self):
        self.episode_count = 0
        self.collision_free_nums = 0
        self.nav_time_all = 0.0
        self.nav_start_time = rospy.Time.now().to_sec()
        self.velocity_all = 0.0
        self.velocity_cur = []
        self.time_in_private_all = 0.0
        self.time_in_private_cur = 0.0

    def _wait_for_sensors(self, timeout=15.0):
        t0 = rospy.Time.now()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            ready = (
                self.rgb_l_cur is not None
                and self.rgb_r_cur is not None
                and self.depth_l_cur is not None
                and self.depth_r_cur is not None
                and self.pose_cur is not None
            )
            if ready:
                return
            if (rospy.Time.now() - t0).to_sec() > timeout:
                raise RuntimeError(
                    "Timed out waiting for /rgb_*, /depth_* and /tf_pose. "
                    "Start the simulator first."
                )
            rate.sleep()

    def step(self, action):
        self.take_action(action)
        obs = self.get_observation()
        reward = self.compute_reward()
        done = self.is_done()
        self._publish_goal_marker()
        self.step_nums += 1

        if self.train_mode and self.collision_person:
            self.vel_cmd_pub.publish(Twist())
            time.sleep(self.collision_pause_s)
            self.collision_person = False

        rospy.Rate(self.CONTROL_HZ).sleep()
        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        t0 = rospy.Time.now()
        while rospy.Time.now() - t0 <= rospy.Duration(0.2):
            self.vel_cmd_pub.publish(Twist())

        if not self.quality_analysis and self.pose_cur is not None:
            # Compensate the pose error Isaac Sim introduces on reset.
            state_diff = Point()
            state_diff.x = self.pose_cur[0] - self.reset_error.x
            state_diff.y = self.pose_cur[1] - self.reset_error.y
            self.set_robot_state_pub.publish(state_diff)

        self.step_nums = 0
        self.collision_times = 0
        self.time_in_private_cur = 0.0
        self.sensors_buffer.clear()
        self.poses_buffer.clear()
        self._wait_for_sensors()

        if not self.quality_analysis:
            self.goal[0], self.goal[1], _ = get_random_goal(
                self.map, np.array([1.0, 0.0, 0, 0, 0, 0]), self.ROBOT_RADIUS
            )

        if not self.train_mode:
            goal_msg = PoseStamped()
            goal_msg.header.frame_id = "map"
            goal_msg.header.stamp = rospy.Time.now()
            goal_msg.pose.position.x = float(self.goal[0])
            goal_msg.pose.position.y = float(self.goal[1])
            goal_msg.pose.orientation.w = 1.0
            self.movebase_goal_pub.publish(goal_msg)
            self.nav_start_time = rospy.Time.now().to_sec()
            self.velocity_cur = []

        return self.get_observation(), {}

    def take_action(self, action):
        cmd_vel = Twist()
        linear_vel, angular_vel = self.actions[int(np.asarray(action).item())]
        cmd_vel.linear.x = linear_vel
        cmd_vel.angular.z = angular_vel

        if not self.train_mode and self.depth_l_cur is not None and self.depth_r_cur is not None:
            crop_lo, crop_hi = self.depth_crop
            depth_cur = np.concatenate((self.depth_l_cur, self.depth_r_cur), axis=-1)[:, crop_lo:crop_hi]
            if depth_cur.min() <= 2.0 * self.ROBOT_RADIUS:
                cmd_vel.linear.x = 0.0
                min_index = np.argmin(depth_cur) % depth_cur.shape[1]
                cmd_vel.angular.z = -1.5 if min_index < 0.5 * depth_cur.shape[1] else 1.5

        self.vel_cmd_pub.publish(cmd_vel)

    def get_observation(self):
        camera_left = np.concatenate(
            [self.rgb_l_cur, self.depth_l_cur[..., np.newaxis]], axis=-1
        )
        camera_right = np.concatenate(
            [self.rgb_r_cur, self.depth_r_cur[..., np.newaxis]], axis=-1
        )
        sensors_cur = np.stack([camera_left, camera_right], axis=0)
        pose_cur = self.pose_cur
        self.sensors_buffer.append(sensors_cur)
        self.poses_buffer.append(pose_cur)
        while len(self.sensors_buffer) < self.RECEPTIVE_FIELD:
            self.sensors_buffer.append(sensors_cur)
            self.poses_buffer.append(pose_cur)

        image = np.stack(self.sensors_buffer, axis=0).astype(np.float32)
        pose = np.stack(self.poses_buffer, axis=0).astype(np.float32)
        image_tensor = torch.tensor(image, device=self.device).unsqueeze(0)
        pose_tensor = torch.tensor(pose, device=self.device).unsqueeze(0)
        if self.social_mode:
            image_tensor[..., :3] = image_tensor[..., :3] / 255.0

        with torch.no_grad():
            bev_features = self.bev_generator({"image": image_tensor, "pose": pose_tensor})
        bev_features = bev_features.cpu().numpy()

        goal_r = goal_in_robot_frame(self.pose_cur, self.goal)
        dist_goal = np.linalg.norm(goal_r) + 1e-6
        goal_dir = goal_r / dist_goal
        velocity_r = np.array([self.cur_vel.linear.x, self.cur_vel.linear.y])
        robot_states = np.concatenate([goal_dir, [dist_goal], velocity_r]).reshape(1, -1)

        obs = {"bev_features": bev_features, "goal_position_r": robot_states}
        if self.social_mode:
            obs["human_poses"] = self._human_poses(image_tensor)
        return obs

    def _human_poses(self, image_tensor):
        imgs_rgb = image_tensor[0, 0, ..., :3].permute(0, 3, 1, 2)
        imgs_depth = image_tensor[0, 0, ..., 3]
        joint_keypoints = []
        for cam_id, joint_result in enumerate(self.joints_predictor(imgs_rgb, verbose=False)):
            keypoints_xyc = joint_result.keypoints.data
            if keypoints_xyc.shape[1] == 0:
                continue
            depth_values = get_depth(keypoints_xyc, imgs_depth[cam_id])
            cam_pts = pixel_to_camera(keypoints_xyc, depth_values, self.intrinsics)
            joint_keypoints.append(align_cameras(cam_pts, self.camera_rotation[cam_id]))
        joints = keypoints_preprocess(joint_keypoints, self.AGENT_NUMS).unsqueeze(0)
        return joints.cpu().numpy()

    def compute_reward(self):
        r_vel = self.reward_weights["vel"]
        r_arrival = self.reward_weights["arrival"]
        r_scan = self.reward_weights["scan"]
        r_smooth = self.reward_weights["smooth"]
        r_collision = self.reward_weights["collision"]
        vel_reward = compute_vel_reward(
            self.cur_vel, r_vel, r_arrival, self.pose_cur, self.goal,
            self.GOAL_RADIUS, self.step_nums, self.MAX_STEPS,
        )
        min_obs_dist, _ = self.get_nearest_obstacle_in_fov(
            (self.pose_cur[0], self.pose_cur[1], self.pose_cur[-1])
        )
        static_reward, self.collision_times = compute_static_obs_punish(
            r_scan, r_collision, self.ROBOT_RADIUS, self.collision_times, min_obs_dist
        )
        smooth_reward = compute_smooth_reward(self.cur_vel, r_smooth)
        dynamic_reward, self.collision_person, private_flag, self.collision_times = (
            compute_dynamic_collision(
                self.ped_states, self.pose_cur, r_collision, r_scan,
                self.ROBOT_RADIUS, self.PERSON_RADIUS, self.train_mode, self.collision_times,
            )
        )
        if private_flag:
            self.time_in_private_cur += 0.15
        return vel_reward + static_reward + smooth_reward + dynamic_reward

    def is_done(self):
        dist_to_goal = np.linalg.norm(self.pose_cur[:2] - self.goal)
        reached = dist_to_goal <= self.GOAL_RADIUS
        timeout = self.step_nums >= self.MAX_STEPS
        crashed = self.collision_times >= self.MAX_COLLISIONS

        if not self.train_mode:
            self.velocity_cur.append(
                math.hypot(self.cur_vel.linear.x, self.cur_vel.linear.y)
            )
            if reached or crashed or timeout:
                self._update_eval_episode(reached)

        return reached or timeout or crashed

    def _update_eval_episode(self, success):
        self.episode_count += 1
        self.time_in_private_all += self.time_in_private_cur
        self.time_in_private_cur = 0.0
        if success:
            self.collision_free_nums += 1
            self.nav_time_all += rospy.Time.now().to_sec() - self.nav_start_time
            if self.velocity_cur:
                self.velocity_all += float(np.mean(self.velocity_cur))
        self.velocity_cur = []
        if self.episode_count == self.TEST_EPISODES:
            n_ok = max(self.collision_free_nums, 1)
            rospy.loginfo(
                "sr: %f, nav_times: %f, velocity: %f, time_private: %f",
                self.collision_free_nums / self.TEST_EPISODES,
                self.nav_time_all / n_ok,
                self.velocity_all / n_ok,
                self.time_in_private_all / self.TEST_EPISODES,
            )

    def _publish_goal_marker(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "waypoints"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(self.goal[0])
        marker.pose.position.y = float(self.goal[1])
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.2
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 0.5, 0.0, 1.0
        array = MarkerArray()
        array.markers.append(marker)
        self.goal_pub.publish(array)

    def get_nearest_obstacle_in_fov(self, robot_pose, fov_deg=None):
        if fov_deg is None:
            fov_deg = self.fov_deg
        if not self.map.data:
            return float("inf"), 0.0
        width = self.map.info.width
        height = self.map.info.height
        resolution = self.map.info.resolution
        origin = self.map.info.origin
        grid = np.array(self.map.data, dtype=np.int8).reshape((height, width))
        ys, xs = np.where(grid == 100)
        if len(xs) == 0:
            return float("inf"), 0.0

        obs_x = xs * resolution + origin.position.x
        obs_y = ys * resolution + origin.position.y
        rx, ry, rtheta = robot_pose
        dx, dy = obs_x - rx, obs_y - ry
        dists = np.hypot(dx, dy)
        angles = np.arctan2(np.sin(np.arctan2(dy, dx) - rtheta), np.cos(np.arctan2(dy, dx) - rtheta))
        fov_mask = np.abs(angles) <= math.radians(fov_deg / 2.0)
        if not np.any(fov_mask):
            return float("inf"), 0.0
        min_idx = np.argmin(dists[fov_mask])
        return float(dists[fov_mask][min_idx]), float(angles[fov_mask][min_idx])

    def _rgb_left_cb(self, msg):
        self.rgb_l_cur = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))

    def _rgb_right_cb(self, msg):
        self.rgb_r_cur = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))

    def _depth_left_cb(self, msg):
        self.depth_l_cur = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)

    def _depth_right_cb(self, msg):
        self.depth_r_cur = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)

    def _tf_cb(self, msg):
        t = msg.transforms[0].transform
        quat = [t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]
        euler = R.from_quat(quat).as_euler("xyz", degrees=False)
        self.pose_cur = np.array(
            [t.translation.x, t.translation.y, t.translation.z, euler[0], euler[1], euler[2]]
        )
        pose = PoseWithCovarianceStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = rospy.Time.now()
        pose.pose.pose.position.x = self.pose_cur[0]
        pose.pose.pose.position.y = self.pose_cur[1]
        pose.pose.pose.orientation.x = quat[0]
        pose.pose.pose.orientation.y = quat[1]
        pose.pose.pose.orientation.z = quat[2]
        pose.pose.pose.orientation.w = quat[3]
        self.pose_pub.publish(pose)

    def _ped_cb(self, msg):
        self.ped_states = msg

    def _goal_cb(self, msg):
        self.goal[0] = msg.pose.position.x
        self.goal[1] = msg.pose.position.y

    def _reset_error_cb(self, msg):
        self.reset_error = msg

    def _odom_cb(self, msg):
        self.cur_vel = msg.twist.twist

    def _map_cb(self, msg):
        self.map = msg
