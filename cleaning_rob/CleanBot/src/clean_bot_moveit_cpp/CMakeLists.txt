cmake_minimum_required(VERSION 3.8)
project(clean_bot_moveit_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(control_msgs REQUIRED)
find_package(interactive_markers REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)

add_executable(clean_bot_moveit_cpp src/clean_bot_moveit_cpp.cpp)
target_include_directories(clean_bot_moveit_cpp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
target_compile_features(clean_bot_moveit_cpp PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(
  clean_bot_moveit_cpp
  "moveit_ros_planning_interface"
  "rclcpp"
)

# Zero pose interactive node
add_executable(zero_pose_interactive src/zero_pose_interactive.cpp)
target_compile_features(zero_pose_interactive PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(
  zero_pose_interactive
  "moveit_ros_planning_interface"
  "rclcpp"
  "rclcpp_action"
  "control_msgs"
  "interactive_markers"
  "visualization_msgs"
  "trajectory_msgs"
)

install(TARGETS clean_bot_moveit_cpp zero_pose_interactive
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
