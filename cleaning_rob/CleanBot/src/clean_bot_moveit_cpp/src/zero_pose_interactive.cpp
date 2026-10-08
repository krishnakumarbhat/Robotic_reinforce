// zero_pose_interactive.cpp
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <interactive_markers/interactive_marker_server.hpp>
#include <visualization_msgs/msg/interactive_marker.hpp>
#include <visualization_msgs/msg/interactive_marker_control.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

class ZeroPoseInteractive : public rclcpp::Node
{
public:
  ZeroPoseInteractive() : Node("zero_pose_interactive")
  {
    // Initialize action client for the arm controller
    action_client_ = rclcpp_action::create_client<control_msgs::action::FollowJointTrajectory>(
        this, "/arm_controller/follow_joint_trajectory");

    // Initialize interactive marker server using raw pointer like in moveit
    int_marker_server_ = new interactive_markers::InteractiveMarkerServer("zero_pose_button", this);

    // Create the interactive button
    createInteractiveButton();

    RCLCPP_INFO(this->get_logger(), "Zero Pose Interactive node started");
    RCLCPP_INFO(this->get_logger(), "Click the green button in RViz to move arm to zero pose");

    // Wait for action server to be available
    if (!action_client_->wait_for_action_server(std::chrono::seconds(5))) {
      RCLCPP_ERROR(this->get_logger(), "Action server not available after waiting");
      return;
    }

    RCLCPP_INFO(this->get_logger(), "Action server is available");
  }

private:
  rclcpp_action::Client<control_msgs::action::FollowJointTrajectory>::SharedPtr action_client_;
  interactive_markers::InteractiveMarkerServer* int_marker_server_;

  void createInteractiveButton()
  {
    // Create interactive marker
    visualization_msgs::msg::InteractiveMarker marker;
    marker.header.frame_id = "base_link";
    marker.header.stamp = this->now();
    marker.name = "zero_pose_button";
    marker.description = "Click to move arm to zero pose";
    
    // Position the button in front of the robot
    marker.pose.position.x = 0.5;
    marker.pose.position.y = 0.0;
    marker.pose.position.z = 1.0;
    marker.pose.orientation.w = 1.0;

    // Create button control
    visualization_msgs::msg::InteractiveMarkerControl button_control;
    button_control.always_visible = true;
    button_control.interaction_mode = visualization_msgs::msg::InteractiveMarkerControl::BUTTON;

    // Create visual representation (green box)
    visualization_msgs::msg::Marker box_marker;
    box_marker.type = visualization_msgs::msg::Marker::CUBE;
    box_marker.scale.x = 0.3;
    box_marker.scale.y = 0.2;
    box_marker.scale.z = 0.1;
    box_marker.color.r = 0.0;
    box_marker.color.g = 0.8;
    box_marker.color.b = 0.2;
    box_marker.color.a = 1.0;
    
    // Add text label
    visualization_msgs::msg::Marker text_marker;
    text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text_marker.text = "ZERO POSE";
    text_marker.scale.z = 0.08;
    text_marker.color.r = 1.0;
    text_marker.color.g = 1.0;
    text_marker.color.b = 1.0;
    text_marker.color.a = 1.0;
    text_marker.pose.position.z = 0.15;

    button_control.markers.push_back(box_marker);
    button_control.markers.push_back(text_marker);
    marker.controls.push_back(button_control);

    // Set feedback callback
    int_marker_server_->insert(marker, 
        std::bind(&ZeroPoseInteractive::buttonCallback, this, std::placeholders::_1));
    
    // Apply changes
    int_marker_server_->applyChanges();
  }

  void buttonCallback(const visualization_msgs::msg::InteractiveMarkerFeedback::ConstSharedPtr& feedback)
  {
    if (feedback->event_type == visualization_msgs::msg::InteractiveMarkerFeedback::BUTTON_CLICK) {
      RCLCPP_INFO(this->get_logger(), "Zero pose button clicked! Sending trajectory to arm controller...");
      
      // Send zero pose trajectory
      sendZeroPoseTrajectory();
    }
  }

  void sendZeroPoseTrajectory()
  {
    // Create trajectory goal
    auto goal_msg = control_msgs::action::FollowJointTrajectory::Goal();
    
    // Set joint names (from your controller.yaml)
    goal_msg.trajectory.joint_names = {
      "shoulder_pan_joint", 
      "shoulder_lift_joint", 
      "elbow_joint", 
      "wrist_yaw_joint", 
      "wrist_pitch_joint", 
      "wrist_roll_joint"
    };
    
    // Create trajectory point with zero positions (from your SRDF)
    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.positions = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};  // Zero pose from SRDF
    point.time_from_start = rclcpp::Duration::from_seconds(3.0);  // 3 second execution
    
    goal_msg.trajectory.points.push_back(point);
    goal_msg.trajectory.header.stamp = this->now();
    
    // Set up goal options with callbacks
    auto send_goal_options = rclcpp_action::Client<control_msgs::action::FollowJointTrajectory>::SendGoalOptions();
    send_goal_options.goal_response_callback =
        std::bind(&ZeroPoseInteractive::goalResponseCallback, this, std::placeholders::_1);
    send_goal_options.feedback_callback =
        std::bind(&ZeroPoseInteractive::feedbackCallback, this, std::placeholders::_1, std::placeholders::_2);
    send_goal_options.result_callback =
        std::bind(&ZeroPoseInteractive::resultCallback, this, std::placeholders::_1);
    
    // Send goal
    action_client_->async_send_goal(goal_msg, send_goal_options);
  }

  void goalResponseCallback(const rclcpp_action::ClientGoalHandle<control_msgs::action::FollowJointTrajectory>::SharedPtr& goal_handle)
  {
    if (!goal_handle) {
      RCLCPP_ERROR(this->get_logger(), "Goal was rejected by server");
    } else {
      RCLCPP_INFO(this->get_logger(), "Goal accepted by server, executing zero pose movement...");
    }
  }

  void feedbackCallback(
      const rclcpp_action::ClientGoalHandle<control_msgs::action::FollowJointTrajectory>::SharedPtr&,
      const std::shared_ptr<const control_msgs::action::FollowJointTrajectory::Feedback>)
  {
    // Optional: Handle feedback during execution
    RCLCPP_DEBUG(this->get_logger(), "Executing trajectory...");
  }

  void resultCallback(const rclcpp_action::ClientGoalHandle<control_msgs::action::FollowJointTrajectory>::WrappedResult& result)
  {
    switch (result.code) {
      case rclcpp_action::ResultCode::SUCCEEDED:
        RCLCPP_INFO(this->get_logger(), "✅ Zero pose movement completed successfully!");
        break;
      case rclcpp_action::ResultCode::ABORTED:
        RCLCPP_ERROR(this->get_logger(), "❌ Zero pose movement was aborted");
        if (result.result) {
          RCLCPP_ERROR(this->get_logger(), "Error code: %d, Error string: %s", 
                      result.result->error_code, result.result->error_string.c_str());
        }
        break;
      case rclcpp_action::ResultCode::CANCELED:
        RCLCPP_WARN(this->get_logger(), "⚠️ Zero pose movement was canceled");
        break;
      default:
        RCLCPP_ERROR(this->get_logger(), "❓ Unknown result code: %d", static_cast<int>(result.code));
        break;
    }
  }
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ZeroPoseInteractive>();
  
  // Spin the node
  rclcpp::spin(node);
  
  rclcpp::shutdown();
  return 0;
}