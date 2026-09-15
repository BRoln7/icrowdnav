#!/usr/bin/env python
import os
import carb
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.timeline
from omni.isaac.core.world import World
from omni.isaac.core.utils.extensions import disable_extension, enable_extension

EXTENSIONS_PEOPLE = [
    'omni.anim.people',
    'omni.anim.navigation.bundle',
    'omni.anim.timeline',
    'omni.anim.graph.bundle',
    'omni.anim.graph.core',
    'omni.anim.graph.ui',
    'omni.anim.retarget.bundle',
    'omni.anim.retarget.core',
    'omni.anim.retarget.ui',
    'omni.kit.scripting',
    'omni.graph.io',
    'omni.anim.curve.core',
]

for ext_people in EXTENSIONS_PEOPLE:
    enable_extension(ext_people)

enable_extension("omni.isaac.ros_bridge")
disable_extension("omni.isaac.ros2_bridge")
simulation_app.update()

import omni.usd
omni.usd.get_context().new_stage()

import numpy as np
import rospy
import yaml
from geometry_msgs.msg import Pose, PoseArray, Point

from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.people.person import Person
from pegasus.simulator.logic.people.person_controller import PersonController
from omni.isaac.core.articulations import ArticulationView


def _load_scene_config():
    path = os.environ.get("ICROWDNAV_SCENE_CONFIG")
    if not path:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "warehouse.yaml")
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


class ROSController(PersonController):

    def __init__(self, id, init_pos):
        super().__init__()
        self.target_pos = init_pos
        self.sfm_sub = rospy.Subscriber('/sfm_cmd', PoseArray, self.callback)
        self.id = id

    def callback(self, _msgs):
        self.target_pos = [_msgs.poses[self.id].position.x,
                           _msgs.poses[self.id].position.y,
                           0]

    def update(self, dt: float):
        self._person.update_target_position(self.target_pos)


class PegasusApp:

    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        usd_path = os.environ.get("ICROWDNAV_USD", "")
        scene_cfg = _load_scene_config()
        robot_prim = os.environ.get("ICROWDNAV_ROBOT_PRIM") or scene_cfg.get("robot_prim", "/World/layout/dingo")
        if not usd_path or not os.path.isfile(usd_path):
            raise RuntimeError(
                "USD scene assets are not shipped with this repository. "
                "Set ICROWDNAV_USD to a local ROS-navigation USD that includes a robot "
                "and a stereo RGB-D camera. See the README section 'Simulation Assets'."
            )
        self.pg.load_asset(usd_path, scene_cfg.get("stage_prim", "/World/layout"))

        people_assets_list = Person.get_character_asset_list()
        for person in people_assets_list:
            print(person)

        self.people = []
        for idx, pedestrian in enumerate(scene_cfg["pedestrians"]):
            controller = ROSController(id=idx, init_pos=pedestrian["init_pos"])
            self.people.append(
                Person(
                    pedestrian["name"],
                    pedestrian["character"],
                    init_pos=pedestrian["init_pos"],
                    init_yaw=pedestrian.get("init_yaw", 1.0),
                    controller=controller,
                )
            )

        rospy.init_node('scene_ros_node')
        self.set_robot_state_sub = rospy.Subscriber('/isaacsim/set_robot_state', Point, self.robotstate_callback)
        self.isaacsim_error_pub = rospy.Publisher('/isaacsim/isaacsim_error', Point, queue_size=10)
        self.peds_states_pub = rospy.Publisher('/isaacsim/persons_states', PoseArray, queue_size=10)

        self._robot = ArticulationView(prim_paths_expr=robot_prim, name="robot_view")
        reset_poses = np.array(scene_cfg["reset_poses"], dtype=np.float64)
        if reset_poses.ndim == 1:
            reset_poses = reset_poses.reshape(1, -1)
        self.reset_pose_list = reset_poses
        self.pg.set_viewport_camera(scene_cfg["viewport_eye"], scene_cfg["viewport_target"])
        self.state_diff = Point()
        self.reset_robot_flag = False
        self.world.reset()
        self.stop_sim = False

    def robotstate_callback(self, _msg):
        self.reset_robot_flag = True
        self.state_diff = _msg

    def run(self):
        self.timeline.play()
        while simulation_app.is_running() and not self.stop_sim:
            if self.reset_robot_flag:
                rand_int = np.random.randint(low=0, high=self.reset_pose_list.shape[0])
                self._robot.set_world_poses(
                    np.array([self.reset_pose_list[rand_int, :3] - [self.state_diff.x, self.state_diff.y, 0]])
                )
                positions, orientations = self._robot.get_world_poses()
                error_point = Point()
                error_point.x = positions[0, 0]
                error_point.y = positions[0, 1]
                self.isaacsim_error_pub.publish(error_point)
                self.reset_robot_flag = False
            else:
                self.world.step(render=True)

            pose_array_msg = PoseArray()
            pose_array_msg.header.stamp = rospy.Time.now()
            pose_array_msg.header.frame_id = "map"
            for person in self.people:
                position = person._state.position
                orientation = person._state.attitude
                pose = Pose()
                pose.position.x = position[0]
                pose.position.y = position[1]
                pose.position.z = position[2]
                pose.orientation.x = orientation[0]
                pose.orientation.y = orientation[1]
                pose.orientation.z = orientation[2]
                pose.orientation.w = orientation[3]
                pose_array_msg.poses.append(pose)
            self.peds_states_pub.publish(pose_array_msg)

        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()


def main():
    pg_app = PegasusApp()
    pg_app.run()


if __name__ == "__main__":
    main()
