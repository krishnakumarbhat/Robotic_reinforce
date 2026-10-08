// clean_bot_moveit_cpp.cpp
#include <memory>
#include <iostream>
#include <chrono>
#include <functional>
#include <unordered_map>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "moveit/move_group_interface/move_group_interface.hpp"
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>

using namespace std::chrono_literals;

class MoveItCommander
{
public:
  MoveItCommander(rclcpp::Node::SharedPtr node, const std::string &group_name)
  : node_(std::move(node)), logger_(rclcpp::get_logger("clean_bot_moveit_cpp")),
    move_group_(this->node_, group_name)
  {
    RCLCPP_INFO(logger_, "MoveItCommander created for group '%s'", group_name.c_str());
  }

  bool addCollisionObject(const std::string id, const std::unordered_map<std::string, double>& dimensions, const std::unordered_map<std::string, double>& pose) {
    // Create collision object for the robot to avoid
    auto const collision_object = [frame_id = move_group_.getPlanningFrame(), id, dimensions, pose] {
      moveit_msgs::msg::CollisionObject collision_object;
      collision_object.header.frame_id = frame_id;
      collision_object.id = id;

      // Define the size of the box in meters
      shape_msgs::msg::SolidPrimitive primitive;
      primitive.type = primitive.BOX;
      primitive.dimensions.resize(3);
      primitive.dimensions[primitive.BOX_X] = dimensions.at("x");
      primitive.dimensions[primitive.BOX_Y] = dimensions.at("y");
      primitive.dimensions[primitive.BOX_Z] = dimensions.at("z");

      // Define the pose of the box (relative to the frame_id)
      geometry_msgs::msg::Pose box_pose;
      box_pose.orientation.w = pose.at("w");  // We can leave out the x, y, and z components of the quaternion since they are initialized to 0
      box_pose.position.x = pose.at("x");
      box_pose.position.y = pose.at("y");
      box_pose.position.z = pose.at("z");

      collision_object.primitives.push_back(primitive);
      collision_object.primitive_poses.push_back(box_pose);
      collision_object.operation = collision_object.ADD;

      return collision_object;
    }();

    // Add the collision object to the scene
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
    return planning_scene_interface.applyCollisionObject(collision_object);
  }

  bool planAndExecute(const geometry_msgs::msg::Pose &target)
  {
    move_group_.setPoseTarget(target);
    move_group_.setStartStateToCurrentState();
    rclcpp::sleep_for(50ms);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    RCLCPP_INFO(logger_, "Planning trajectory...");

    bool ok = static_cast<bool>(move_group_.plan(plan));
    if (!ok) {
      RCLCPP_ERROR(logger_, "Planning failed.");
      move_group_.clearPoseTargets();
      return false;
    }

    RCLCPP_INFO(logger_, "Plan succeeded, executing trajectory...");
    moveit::core::MoveItErrorCode exec_result = move_group_.execute(plan);

    move_group_.clearPoseTargets();

    if (exec_result == moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_INFO(logger_, "Execution succeeded!");
      return true;
    } else {
      RCLCPP_ERROR(logger_, "Execution failed with code: %d", exec_result.val);
      return false;
    }
  }

private:
  rclcpp::Node::SharedPtr node_;
  rclcpp::Logger logger_;
  moveit::planning_interface::MoveGroupInterface move_group_;
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>(
    "clean_bot_moveit_cpp",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)
  );

  auto logger = rclcpp::get_logger("clean_bot_moveit_cpp");
  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node);

  RCLCPP_INFO(logger, "Delaying for 3 seconds to allow system sync...");
  rclcpp::sleep_for(3s);

  MoveItCommander commander(node, "arm");

  const std::unordered_map<std::string, double> dimensions = {
    {"x", 0.4},
    {"y", 0.4},
    {"z", 1.0}
  };

  const std::unordered_map<std::string, double> pose1 = {
    {"w", 1.0},
    {"x", 0.5},
    {"y", 0.25},
    {"z", 0.5}
  };

  const std::unordered_map<std::string, double> pose2 = {
    {"w", 1.0},
    {"x", -0.5},
    {"y", 0.25},
    {"z", 0.5}
  };

  RCLCPP_INFO(logger, "Adding Collision object(s) to planning scene");
  if(!commander.addCollisionObject("obstacle1", dimensions, pose1)) {
    RCLCPP_ERROR(logger, "Error: Failed to add collision object 1");
    rclcpp::shutdown();
    return 0;
  }
  RCLCPP_INFO(logger, "Collision Object 1 successfully added");

  if(!commander.addCollisionObject("obstacle2", dimensions, pose2)) {
    RCLCPP_ERROR(logger, "Error: Failed to add collision object 2");
    rclcpp::shutdown();
    return 0;
  }
  RCLCPP_INFO(logger, "Collision Object 2 successfully added");
  
  RCLCPP_INFO(logger, "Delaying for 3 seconds to allow system sync...");
  rclcpp::sleep_for(3s);

  const geometry_msgs::msg::Pose before_pose = [] {
    geometry_msgs::msg::Pose p;
    p.orientation.w = 0.6108;
    p.orientation.x = 0.6118;
    p.orientation.y = -0.3589;
    p.orientation.z = -0.3518;
    p.position.x = 0.2432;
    p.position.y = -0.3952;
    p.position.z = 0.3495;
    return p;
  }();

  const geometry_msgs::msg::Pose after_pose = [] {
    geometry_msgs::msg::Pose p;
    p.orientation.w = 0.4144;
    p.orientation.x = 0.4128;
    p.orientation.y = 0.5711;
    p.orientation.z = 0.5760;
    p.position.x = -0.1664;
    p.position.y = 0.4355;
    p.position.z = 0.3466;
    return p;
  }();

  RCLCPP_INFO(logger, "Moving to pre-collision pose...");
  if (!commander.planAndExecute(before_pose)) {
    RCLCPP_WARN(logger, "Failed to move to before_pose; continuing...");
    rclcpp::shutdown();
    return 0;
  }

  RCLCPP_INFO(logger, "Waiting for robot to settle after first movement...");
  rclcpp::sleep_for(5s);

  RCLCPP_INFO(logger, "Moving to post-collision pose...");
  if (!commander.planAndExecute(after_pose)) {
    RCLCPP_ERROR(logger, "Failed to move to after_pose.");
    rclcpp::shutdown();
    return 0;
  }

  RCLCPP_INFO(logger, "Movement successful. Shutting down...");
  rclcpp::shutdown();
  return 0;
}
