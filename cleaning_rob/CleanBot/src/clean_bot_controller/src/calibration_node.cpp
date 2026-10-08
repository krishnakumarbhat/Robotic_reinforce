#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include <map>
#include <mutex>
#include <memory>
#include <iostream>
#include <thread>
#include <string>
#include <sstream>
#include <clean_bot_controller/utils.hpp>

inline constexpr double kStsResolution = 4096.;
inline constexpr double PI = 3.14159;

int from_radians(const double angle) {
  return static_cast<int>(angle * kStsResolution / (2.0 * PI));
}

class CalibrationNode : public rclcpp::Node
{
public:
  CalibrationNode()
    : Node("calibration_node")
  {
    subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states",
      10,
      std::bind(&CalibrationNode::jointStateCallback, this, std::placeholders::_1)
    );

    RCLCPP_INFO(this->get_logger(), "CalibrationNode initialized. Waiting for joint states...");
  }

  std::map<std::string, int> getCurrentJointTicks()
  {
    std::lock_guard<std::mutex> lock(joint_state_mutex_);
    std::map<std::string, int> joint_map;

    if (!latest_joint_state_)
    {
      RCLCPP_WARN(this->get_logger(), "No joint state received yet.");
      return joint_map;
    }

    for (size_t i = 0; i < latest_joint_state_->name.size(); ++i)
    {
      if (i < latest_joint_state_->position.size())
      {
        double radians = latest_joint_state_->position[i];
        joint_map[latest_joint_state_->name[i]] = from_radians(radians);
      }
    }

    return joint_map;
  }

private:
  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(joint_state_mutex_);
    latest_joint_state_ = msg;
  }

  sensor_msgs::msg::JointState::SharedPtr latest_joint_state_;
  std::mutex joint_state_mutex_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr subscription_;
};

void printJointMap(const std::map<std::string, int>& joint_map, const rclcpp::Logger& logger)
{
  if (joint_map.empty())
  {
    RCLCPP_WARN(logger, "No joint data captured.");
    return;
  }

  std::stringstream ss;
  ss << "{";
  for (auto it = joint_map.begin(); it != joint_map.end(); ++it)
  {
    ss << it->first << ": " << it->second;
    if (std::next(it) != joint_map.end())
      ss << ", ";
  }
  ss << "}";

  RCLCPP_INFO(logger, "Captured Joint Ticks: %s", ss.str().c_str());
}

int main(int argc, char* argv[])
{
  std::map<std::string, int> homing_offset;

  rclcpp::init(argc, argv);
  auto node = std::make_shared<CalibrationNode>();

  std::thread spin_thread([&]() {
    rclcpp::spin(node);
    });

  // Get zero position
  std::cout << "Move arm to zero position, then press [ENTER]..." << std::endl;
  std::cin.get();

  auto zero_pos = node->getCurrentJointTicks();
  printJointMap(zero_pos, node->get_logger());

  // Get zero offset 0 - pos => offset
  std::map<std::string, int> target_zero_ticks;
  for (const auto& [joint, _] : target_zero_ticks)
  {
    target_zero_ticks[joint] = from_radians(0);  // 0 degrees
  }

  for(const auto& [joint, zero_val] : zero_pos)
  {
    if (target_zero_ticks.count(joint))
    {
      homing_offset[joint] = target_zero_ticks[joint] - zero_val;
    }
  }

  // Rotated Position Calibration
  std::cout << "\nMove arm to rotated target position" << std::endl;
  std::cout << "Press [ENTER] to continue..." << std::endl;
  std::cin.get();

  auto rotated_pos = node->getCurrentJointTicks();
  printJointMap(rotated_pos, node->get_logger());

  std::map<std::string, int> rotated_target_ticks;
  for (const auto& [joint, _] : rotated_pos)
  {
    rotated_target_ticks[joint] = from_radians(PI / 2.0);  // 90 degrees
  }

  std::map<std::string, int> rotated_adjusted;
  std::map<std::string, int> drive_mode_map;

  for (const auto& [joint, rot_val] : rotated_pos)
  {
    if (zero_pos.count(joint) == 0) continue;

    int zero_val = zero_pos[joint];
    int signed_mode = (rot_val < zero_val) ? -1 : 1;
    drive_mode_map[joint] = signed_mode;

    rotated_adjusted[joint] = rot_val * signed_mode;

    if (rotated_target_ticks.count(joint))
    {
      homing_offset[joint] = rotated_target_ticks[joint] - rotated_adjusted[joint];
    }
  }

  // Print homing offsets
  std::stringstream ss;
  ss << "Computed Homing Offsets:\n{";
  for (auto it = homing_offset.begin(); it != homing_offset.end(); ++it)
  {
    ss << it->first << ": " << it->second;
    if (std::next(it) != homing_offset.end()) ss << ", ";
  }
  ss << "}";
  RCLCPP_INFO(node->get_logger(), "%s", ss.str().c_str());

  // Print drive modes
  std::stringstream dss;
  dss << "Resolved Drive Modes:\n{";
  for (auto it = drive_mode_map.begin(); it != drive_mode_map.end(); ++it)
  {
    dss << it->first << ": " << it->second;
    if (std::next(it) != drive_mode_map.end()) dss << ", ";
  }
  dss << "}";
  RCLCPP_INFO(node->get_logger(), "%s", dss.str().c_str());

  save("/home/hegde-aryan/dev/CleanBot/src/clean_bot_controller/config/calibration_data.yaml", homing_offset, zero_pos, rotated_pos, drive_mode_map);
  RCLCPP_INFO(node->get_logger(), "Calibration data saved to calibration_data.yaml");

  rclcpp::shutdown();
  spin_thread.join();
  return 0;
}
