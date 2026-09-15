#!/usr/bin/env python
import carb
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.timeline
from omni.isaac.core.world import World
from omni.isaac.core.utils.extensions import disable_extension, enable_extension

from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.articulations import ArticulationView
import omni.isaac.core.utils.prims as prim_utils

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

from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.people.person import Person
from pegasus.simulator.logic.people.person_controller import PersonController
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

class CirclePersonController(PersonController):

    def __init__(self, center):
        super().__init__()

        self._radius = 3.0
        self._center = center
        self.gamma = 0.0
        self.gamma_dot = 0.3
        
    def update(self, dt: float):
        self.gamma += self.gamma_dot * dt
        self._person.update_target_position([self._radius * np.cos(self.gamma)+self._center[0], 
                                             self._radius * np.sin(self.gamma)+self._center[1], 0.0])

class GoBackPersonController(PersonController):

    def __init__(self, pos1, pos2):
        super().__init__()

        self.pos1 = pos1
        self.pos2 = pos2
        self.status = 0
        self.radius = 0.2
        
    def update(self, dt: float):
        dis1_2d = self._person._state.position - self.pos1
        dis2_2d = self._person._state.position - self.pos2

        dis1 = np.sqrt(dis1_2d[0]**2 + dis1_2d[1]**2)
        dis2 = np.sqrt(dis2_2d[0]**2 + dis2_2d[1]**2)

        if dis1 < self.radius:
            self.status = 1
        if dis2 < self.radius:
            self.status = 0

        if self.status == 0:
            self._person.update_target_position(self.pos1)
        if self.status == 1:
            self._person.update_target_position(self.pos2)

class PegasusApp:

    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        self.pg.load_asset('/home/bingyi/socialbev_ws/src/scene/assets/warehouse.usd', "/World/layout")
        people_assets_list = Person.get_character_asset_list()
        for person in people_assets_list:
            print(person)

        center1 = [-1.0, -2.0, 0.0]
        circle_controller = CirclePersonController(center1)
        self.p1 = Person("person1", "original_male_adult_construction_05", init_pos=center1, init_yaw=1.0, controller=circle_controller)

        center2 = [-2.0, 4.0, 0.0]
        circle_controller = CirclePersonController(center2)
        self.p2 = Person("person2", "original_male_adult_construction_03", init_pos=center2, init_yaw=1.0, controller=circle_controller)

        p3_pos1 = [-4.0, 0.0, 0.0]
        p3_pos2 = [2.0, 2.0, 0.0]
        goback_controller = GoBackPersonController(p3_pos1, p3_pos2)
        self.p3 = Person("person3", "original_male_adult_construction_01", init_pos=p3_pos1, controller=goback_controller)

        p4_pos1 = [3.0, 4.0, 0.0]
        p4_pos2 = [-4.0, 5.0, 0.0]
        goback_controller = GoBackPersonController(p4_pos1, p4_pos2)
        self.p4 = Person("person4", "original_male_adult_medical_01", init_pos=p4_pos1, controller=goback_controller)

        p5_pos1 = [-3.0, 6.0, 0.0]
        p5_pos2 = [0.0, 2.0, 0.0]
        goback_controller = GoBackPersonController(p5_pos1, p5_pos2)
        self.p5 = Person("person5", "original_female_adult_business_02", init_pos=p5_pos1, controller=goback_controller)

        self.pg.set_viewport_camera([5.0, 9.0, 6.5], [0.0, 0.0, 0.0])
        self.world.reset()
        self.stop_sim = False

    def run(self):
        self.timeline.play()
        while simulation_app.is_running() and not self.stop_sim:
            self.world.step(render=True)
        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()

def main():
    pg_app = PegasusApp()
    pg_app.run()

if __name__ == "__main__":
    main()