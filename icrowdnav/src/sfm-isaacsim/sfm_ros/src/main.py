#!/usr/bin/env python
import os
import rospy
import rospkg
import numpy as np
import cv2
import yaml
import matplotlib.pyplot as plt
import random
import math
from scipy.ndimage import distance_transform_edt
from geometry_msgs.msg import PoseArray, Pose, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid

class Social_Force:
    def __init__(self, edt):
        self.name = 'sfm'
        self.dt = 0.5
        self.rad = 0.5
        self.max_speed = 1.0

        self.gain_k = 3
        self.gain_a_static = 4
        self.gain_b_static = 0.25

        self.gain_a_agent = 4.0 #2.0
        self.gain_b_agent = 0.5
        
        self.distance_field = edt
        self.grad_map = self.compute_gradient(self.distance_field)
    
        self.avoid_dis = 5
        self.reach_dis = 0.3
        self.safe_dis = 0.75
        
    @staticmethod
    def compute_gradient(distance_field):
        
        grad = np.gradient(distance_field)
        magnitude = np.sqrt(grad[0]**2 + grad[1]**2)
        magnitude[magnitude == 0] = 1e-6
        grad_norm = grad / magnitude
        
        return grad_norm

    def compute_forces(self, agent_state, goal, rel_obs):
        pos, cord_int, vel = agent_state
        nbrs_idx, nbrs_relpos, nbrs_dis = rel_obs
        x, y = cord_int
        
        # 1. Goal attraction
        direction_to_goal = goal - pos
        distance_to_goal = np.linalg.norm(direction_to_goal)
        
        if distance_to_goal > self.reach_dis:
            attractive_force = self.max_speed * direction_to_goal/distance_to_goal
        else:
            attractive_force = np.zeros(2)
        
        # 2. Static obstacle repulsion
        dist = self.distance_field[x,y]/4
        
        if dist < self.avoid_dis:
            grad = np.array([self.grad_map[0,x,y], self.grad_map[1,x,y]])
            repulsive_force = self.gain_a_static * np.exp( (0.5*self.safe_dis - dist)/ self.gain_b_static ) * grad
        else:
            repulsive_force = np.zeros(2)
        
        # 3. Agent interaction force
        if len(nbrs_idx) != 0:

            nbrs_dis = nbrs_dis.reshape(-1,1)
            nbrs_relpos = nbrs_relpos/nbrs_dis
                
            interact_force = self.gain_a_agent * np.exp( (self.safe_dis - nbrs_dis)/ self.gain_b_agent ) * nbrs_relpos
            interact_force = np.sum(interact_force, axis=0)
        else:
            interact_force = np.zeros(2)

        # Combined force
        # Sum of push & pull forces
        d_vel = self.gain_k * (attractive_force - vel)
        interaction_vel = repulsive_force + interact_force
        total_d_vel = (d_vel + interaction_vel) * self.dt
        new_vel = vel + total_d_vel

        act_norm = np.linalg.norm(new_vel)
        if act_norm > self.max_speed:
            return new_vel*self.max_speed / act_norm, [interact_force, repulsive_force, d_vel]
        else:
            return new_vel, [interact_force, repulsive_force, d_vel]
    
    def get_action(self, obs):
        agent_state = obs[:3]
        goal = obs[3]
        rel_obs = obs[-3:]
        return self.compute_forces(agent_state, goal, rel_obs)
        
def load_image(img_path, scale=1/3):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
    scale_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    _, map_free = cv2.threshold(scale_img, 127, 1, cv2.THRESH_BINARY_INV)
    _, map_obs = cv2.threshold(scale_img, 127, 1, cv2.THRESH_BINARY)
    return map_free, map_obs, img

try:
    _map_dir = os.path.join(rospkg.RosPack().get_path("dingo_2dnav"), "map")
except Exception:
    _map_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "scene", "dingo_2dnav", "map"))
img_path = os.path.join(_map_dir, "dingo_warehouse_navigation.png")
yaml_path = os.path.join(_map_dir, "dingo_warehouse_navigation.yaml")
img_scale = 1/3
map_free, map_obs, map_img = load_image(img_path, img_scale)
inflated_map = cv2.dilate(map_free.astype(np.uint8), np.ones((5, 5), np.uint8))
dis_field = distance_transform_edt(map_obs)
with open(yaml_path, 'r') as file:
    map_data = yaml.safe_load(file)
resolution = map_data['resolution'] / img_scale
origin = map_data['origin'][:2]
origin[1] = origin[1] / (int(abs(origin[1])/resolution)/(map_free.shape[0] - int(abs(origin[1])/resolution)))

class SocialForceROS:
    def __init__(self, agent_nums):
        self.sfm = Social_Force(dis_field)
        self.agent_state = np.zeros([agent_nums, 6]) # pos2, vel2 and goal2
        self.robot_pose = np.zeros([1, 2])
        self.goal_narray = np.array([[[0.0, 0.0, 1.0],[-5.0, 0.0, 1.0]],
                                    [[-3.5, 1.5, 1.0],[1.0, 1.5, 1.0]],
                                    [[-2.0, 1.0, 1.0],[-6.0, 0.0, 1.0]],
                                    [[4.5, 3.0, 1.0],[4.5, 7.0, 1.0]],
                                    [[3.0, 7.0, 1.0],[3.0, 3.0, 1.0]],
                                    ])
        # map
        self.map = OccupancyGrid()
        self.map.info.resolution = resolution
        self.map.info.origin.position.x = origin[0]
        self.map.info.origin.position.y = origin[1]
        self.map.info.height = map_obs.shape[0]
        self.map.info.width = map_obs.shape[1]
        self.map_data = inflated_map
        self.GOAL_INIT = False
        self.W_to_M_T = np.array([
            [0, -1, -origin[1]],
            [1, 0, -origin[0]],
            [0, 0, 1]
        ])
        self.goal_narray = (self.goal_narray.reshape(-1,3)@self.W_to_M_T.T).reshape(-1,2,3)
        self.goal_indices = np.ones([agent_nums, 1], dtype=int)

        self.agent_nums = agent_nums
        self.AGENT_RADIUS = 0.5
        self.state_sub = rospy.Subscriber('/isaacsim/persons_states', PoseArray, self.state_callback)
        self.robot_pose_sub = rospy.Subscriber('/isaacsim_pose', PoseWithCovarianceStamped, self.robot_pose_callback)
        self.sfm_pub = rospy.Publisher('/sfm_cmd', PoseArray, queue_size=10)
        self.dt = 0.02

    def robot_pose_callback(self, _msg:PoseWithCovarianceStamped):
        p_x, p_y = _msg.pose.pose.position.x, _msg.pose.pose.position.y
        p_w = np.array([p_x, p_y, 1])
        p_m = self.W_to_M_T @ p_w
        self.robot_pose[0, :2] = p_m[:2]

    def state_callback(self, _msgs:PoseArray):
        # get pos and vel
        id = 0
        for pose in _msgs.poses:
            p_x, p_y = pose.position.x, pose.position.y
            p_w = np.array([p_x, p_y, 1])
            p_m = self.W_to_M_T @ p_w
            self.agent_state[id, 2] = (p_m[0] - self.agent_state[id, 0]) / self.dt
            self.agent_state[id, 3] = (p_m[1] - self.agent_state[id, 1]) / self.dt
            self.agent_state[id, :2] = p_m[:2]
            self.agent_state[id, -2:] = self.goal_narray[id, self.goal_indices[id], :2]
            if np.sqrt((self.agent_state[id, 0]-self.agent_state[id, -2])**2+\
                       (self.agent_state[id, 1]-self.agent_state[id, -1])**2) < 0.5:
                self.goal_indices[id] += 1
                self.goal_indices[id] = self.goal_indices[id] % 2
            id += 1
        
        # compute social force model
        sfm_cmd = PoseArray()
        id = 0
        for state in self.agent_state:
            pose = Pose()
            agent_state = [self.agent_state[id, :2],
                           (self.agent_state[id, :2]/resolution).astype(int),
                           self.agent_state[id, 2:4]]
            goal = self.agent_state[id, 4:]
            nbrs_pos = np.delete(self.agent_state, id, axis=0)[:, :2]
            nbrs_pos = np.concatenate((nbrs_pos, self.robot_pose), axis=0)
            nbrs_pos = self.agent_state[id, :2].reshape(1, -1) - nbrs_pos
            nbrs_dis = np.linalg.norm(nbrs_pos, axis=-1)
            rel_obs = [np.array([self.agent_nums-1]), nbrs_pos, nbrs_dis]
            vel_cmd, _ = self.sfm.compute_forces(agent_state, goal, rel_obs)
            vel_cmd = vel_cmd / np.sqrt(vel_cmd[0]**2 + vel_cmd[1]**2)

            waypoint_m = self.agent_state[id, :2] + vel_cmd
            waypoint_m_hom = np.append(waypoint_m, 1.0)
            waypoint_w_hom = np.linalg.inv(self.W_to_M_T) @ waypoint_m_hom
            waypoint_w = waypoint_w_hom[:2]
            pose.position.x = waypoint_w[0]
            pose.position.y = waypoint_w[1]
            sfm_cmd.poses.append(pose)
            id += 1
        self.sfm_pub.publish(sfm_cmd)

    def find_random_goal(self, pos):
        theta = random.uniform(0, 2 * math.pi)
        dist = random.uniform(5, 5.5)
        x = pos[0] + dist * math.cos(theta)
        y = pos[1] + dist * math.sin(theta)
        while not self._is_pos_valid(x, y, self.map):
            theta = random.uniform(0, 2 * math.pi)
            dist = random.uniform(5, 5.5)
            x = pos[0] + dist * math.cos(theta)
            y = pos[1] + dist * math.sin(theta)
        return (x, y)

    def _is_pos_valid(self, x, y, map:OccupancyGrid):
        x_index = int(x/map.info.resolution)
        y_index = int(y/map.info.resolution)

        if y_index < self.map.info.width and x_index < self.map.info.height and \
            self.sample_corner[0, 0]< x < self.sample_corner[1, 0] and \
            self.sample_corner[1, 1]< y < self.sample_corner[0, 1] and \
            self.map_data[x_index, y_index] == 0:
            return True
        else:
            return False

if __name__ == '__main__':
    rospy.init_node("sfm_node")
    try:
        SocialForceROS(agent_nums=5)
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("sfm starts")


