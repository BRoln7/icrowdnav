cmake_minimum_required(VERSION 3.0.2)
project(scene_ros)

find_package(catkin REQUIRED COMPONENTS rospy)

catkin_package()

install(DIRECTORY src config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
