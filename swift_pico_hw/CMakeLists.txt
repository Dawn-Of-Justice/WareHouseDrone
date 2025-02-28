cmake_minimum_required(VERSION 3.8)
project(swift_pico_hw)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rc_msgs REQUIRED)
find_package(pid_msg REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclpy REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(waypoint_navigation REQUIRED)

# Install launch files.
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

#Install scripts
install(PROGRAMS
  src/pico_controller.py
  src/arm.py
  src/pico_client_5.py
  src/pico_server_5.py
  src/waypoint_service.py
  src/path_planning_service.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()