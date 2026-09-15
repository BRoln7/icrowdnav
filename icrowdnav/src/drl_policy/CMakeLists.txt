cmake_minimum_required(VERSION 3.0.2)
project(drl_policy)

find_package(catkin REQUIRED COMPONENTS rospy)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
  policy_train.py
  policy_inference.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
