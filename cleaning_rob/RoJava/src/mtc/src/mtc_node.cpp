#include <rclcpp/rclcpp.hpp>

#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>

#include <moveit/task_constructor/task.h>
#include <moveit/task_constructor/solvers.h>
#include <moveit/task_constructor/stages.h>
#include <moveit/task_constructor/trajectory_execution_info.h>

#include <string>
#include <vector>

#include <geometric_shapes/shape_operations.h>
#include <boost/variant.hpp>
#if __has_include(<tf2_geometry_msgs/tf2_geometry_msgs.hpp>)
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#else
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#endif
#if __has_include(<tf2_eigen/tf2_eigen.hpp>)
#include <tf2_eigen/tf2_eigen.hpp>
#else
#include <tf2_eigen/tf2_eigen.h>
#endif

static const rclcpp::Logger LOGGER = rclcpp::get_logger("mtc_node");
namespace mtc = moveit::task_constructor;

void generate_grasp_transform(Eigen::Isometry3d& T);

Eigen::Isometry3d vectorToEigen(const std::vector<double>& values) {
  return Eigen::Translation3d(values[0], values[1], values[2]) *
         Eigen::AngleAxisd(values[3], Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(values[4], Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(values[5], Eigen::Vector3d::UnitX());
}

class MTCTaskNode : public rclcpp::Node {
public:
  MTCTaskNode(const rclcpp::NodeOptions& options);

  // rclcpp::node_interfaces::NodeBaseInterface::SharedPtr getNodeBaseInterface();

  void doTask();

  void setupPlanningScene();

private:
  // Compose an MTC task from a series of stages.
  mtc::Task createTask();
  mtc::Task task_;
  //rclcpp::Node::SharedPtr node_;

  
};

MTCTaskNode::MTCTaskNode(const rclcpp::NodeOptions& options)
  // : node_(std::make_shared<rclcpp::Node>("mtc_node", options)) {
  : Node("mtc_node", options) {

    auto declare_parameter = [this](const std::string& name, const auto& default_value, const std::string& description = "") {
      rcl_interfaces::msg::ParameterDescriptor descriptor;
      descriptor.description = description;

      if (!this->has_parameter(name)) {
        this->declare_parameter(name, default_value, descriptor);
      }
    };

    declare_parameter("controller_names", std::vector<std::string>{"arm_controller", "gripper_controller"}, "Names of the controllers to use");
  }

// rclcpp::node_interfaces::NodeBaseInterface::SharedPtr MTCTaskNode::getNodeBaseInterface()
// {
//   return node_->get_node_base_interface();
// }

void MTCTaskNode::setupPlanningScene()
{
  // Add ground to planning scene
  moveit_msgs::msg::CollisionObject ground;
  ground.id = "ground";
  ground.header.frame_id = "world";
  ground.primitives.resize(1);
  ground.primitives[0].type = shape_msgs::msg::SolidPrimitive::BOX;
  ground.primitives[0].dimensions = { 0.95, 2.0, 0.00001 };

  geometry_msgs::msg::Pose ground_pose;
  ground_pose.position.x = -0.6;
  ground_pose.position.y = 0.0;
  ground_pose.orientation.w = 1.0;
  ground.pose = ground_pose;

  moveit::planning_interface::PlanningSceneInterface psi;
  psi.applyCollisionObject(ground);


  // Adding cup to planning scene
  moveit_msgs::msg::CollisionObject object;
  object.id = "object";
  object.header.frame_id = "world";
  
  // Create mesh from resource file
  shapes::ShapePtr mesh_shape(shapes::createMeshFromResource("package://clean_bot/models/meshes/cup.stl", 
                                                           Eigen::Vector3d(0.001, 0.001, 0.001)));
  
  // Convert shape to mesh message
  shapes::ShapeMsg shape_msg;
  shapes::constructMsgFromShape(mesh_shape.get(), shape_msg);
  object.meshes.push_back(boost::get<shape_msgs::msg::Mesh>(shape_msg));
  object.mesh_poses.push_back(geometry_msgs::msg::Pose());

  geometry_msgs::msg::Pose pose;
  pose.position.x = -0.3;
  pose.position.y = 0.0;
  pose.position.z = 0.033;
  pose.orientation.w = 1.0;
  object.pose = pose;

  psi.applyCollisionObject(object);
}

void MTCTaskNode::doTask()
{
  task_ = createTask();

  try
  {
    task_.init();
  }
  catch (mtc::InitStageException& e)
  {
    RCLCPP_ERROR_STREAM(LOGGER, e);
    return;
  }

  if (!task_.plan(5))
  {
    RCLCPP_ERROR_STREAM(LOGGER, "Task planning failed");
    return;
  }
  RCLCPP_INFO(LOGGER, "Planning successful. \n Waiting to execute");
  
  // task_.introspection().publishSolution(*task_.solutions().front());

  // auto result = task_.execute(*task_.solutions().front());
  // if (result.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS)
  // {
  //   RCLCPP_ERROR_STREAM(LOGGER, "Task execution failed");
  //   return;
  // }
  // RCLCPP_INFO(LOGGER, "Execution successful.");

  return;
}

mtc::Task MTCTaskNode::createTask()
{
  mtc::Task task;
  task.stages()->setName("demo task");
  task.loadRobotModel(shared_from_this(), "robot_description");

  auto controller_names = this->get_parameter("controller_names").as_string_array();

  task.setProperty("trajectory_execution_info",
    mtc::TrajectoryExecutionInfo().set__controller_names(controller_names));

  const auto& arm_group_name = "arm";
  const auto& hand_group_name = "gripper";
  const auto& hand_frame = "gripper_center_link";

  // Set task properties
  task.setProperty("group", arm_group_name);
  task.setProperty("eef", hand_group_name);
  task.setProperty("ik_frame", hand_frame);

  mtc::Stage* current_state_ptr = nullptr;  // Forward current_state on to grasp pose generator
  // currently two generator stages. Generator stages are stored for refernce but don't get added to task

  auto stage_state_current = std::make_unique<mtc::stages::CurrentState>("current");
  current_state_ptr = stage_state_current.get();
  task.add(std::move(stage_state_current));

  std::unordered_map<std::string, std::string> ompl_map_arm = {
      {"ompl", std::string(arm_group_name) + "[RRTConnectkConfigDefault]"}
    };

  auto sampling_planner = std::make_shared<mtc::solvers::PipelinePlanner>(this->shared_from_this(), ompl_map_arm);
  auto interpolation_planner = std::make_shared<mtc::solvers::JointInterpolationPlanner>();

  auto cartesian_planner = std::make_shared<mtc::solvers::CartesianPath>();
  cartesian_planner->setMaxVelocityScalingFactor(1.0);
  cartesian_planner->setMaxAccelerationScalingFactor(1.0);
  cartesian_planner->setStepSize(.01);

  {
    auto stage_open_hand =
      std::make_unique<mtc::stages::MoveTo>("open hand", interpolation_planner);
    stage_open_hand->setGroup(hand_group_name);
    stage_open_hand->setGoal("open");

    stage_open_hand->properties().set("trajectory_execution_info",
                    mtc::TrajectoryExecutionInfo().set__controller_names(controller_names));
    task.add(std::move(stage_open_hand));
  }
  
  {
    auto stage_move_to_pick = std::make_unique<mtc::stages::Connect>(
      "move to pick",
      mtc::stages::Connect::GroupPlannerVector{ { arm_group_name, sampling_planner } });
    stage_move_to_pick->setTimeout(5.0);
    stage_move_to_pick->properties().configureInitFrom(mtc::Stage::PARENT);
    stage_move_to_pick->properties().set("trajectory_execution_info",
                  mtc::TrajectoryExecutionInfo().set__controller_names(controller_names));
  
    task.add(std::move(stage_move_to_pick));
  }

  // mtc::Stage* attach_object_stage = nullptr;  // Forward attach_object_stage to place pose generator

  // container for grasping stage: with subtasks 
  {
    auto grasp = std::make_unique<mtc::SerialContainer>("pick object");
    task.properties().exposeTo(grasp->properties(), { "eef", "group", "ik_frame" });
    grasp->properties().configureInitFrom(mtc::Stage::PARENT,
                                          { "eef", "group", "ik_frame" });

    {
      auto stage =
          std::make_unique<mtc::stages::MoveRelative>("approach object", cartesian_planner);
      stage->properties().set("marker_ns", "approach_object");
      stage->properties().set("link", hand_frame);
      stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
      stage->setMinMaxDistance(0.0, 0.5);
      stage->properties().set("trajectory_execution_info",
                  mtc::TrajectoryExecutionInfo().set__controller_names(controller_names));

      // Set hand forward direction
      geometry_msgs::msg::Vector3Stamped vec;
      vec.header.frame_id = hand_frame;
      vec.vector.x = -1.0;
      stage->setDirection(vec);
      grasp->insert(std::move(stage));
    }

    {
      // Sample grasp pose
      auto stage = std::make_unique<mtc::stages::GenerateGraspPose>("generate grasp pose");
      stage->properties().configureInitFrom(mtc::Stage::PARENT);
      stage->properties().set("marker_ns", "grasp_pose");
      stage->setPreGraspPose("open");
      stage->setObject("object");
      stage->setAngleDelta(M_PI / 12);
      stage->setMonitoredStage(current_state_ptr);  // Hook into current state  
      
      // Eigen::Isometry3d grasp_frame_transform = Eigen::Isometry3d::Identity();
      // generate_grasp_transform(grasp_frame_transform);

      auto grasp_frame_transform = std::vector<double>{-0.055, 0.0, 0.0, 0.0, 0.0, 0.0};

      // Compute IK
      auto wrapper =
          std::make_unique<mtc::stages::ComputeIK>("grasp pose IK", std::move(stage));
      wrapper->setMaxIKSolutions(8);
      wrapper->setMinSolutionDistance(1.0);
      // wrapper->setIKFrame(grasp_frame_transform, hand_frame);
      wrapper->setIKFrame(vectorToEigen(grasp_frame_transform), hand_frame); // Transform from gripper frame to tool center point (TCP)
      wrapper->properties().configureInitFrom(mtc::Stage::PARENT, { "eef", "group" });
      wrapper->properties().configureInitFrom(mtc::Stage::INTERFACE, { "target_pose" });
      grasp->insert(std::move(wrapper));
    }

    {
      auto stage =
          std::make_unique<mtc::stages::ModifyPlanningScene>("allow collision (hand,object)");
      stage->allowCollisions("object",
                            task.getRobotModel()
                                ->getJointModelGroup(hand_group_name)
                                ->getLinkModelNamesWithCollisionGeometry(),
                            true);
      grasp->insert(std::move(stage));
    }

    {
      auto stage = std::make_unique<mtc::stages::MoveTo>("close hand", interpolation_planner);
      stage->setGroup(hand_group_name);
      stage->setGoal("closed");
      stage->properties().set("trajectory_execution_info",
                    mtc::TrajectoryExecutionInfo().set__controller_names(controller_names));

      grasp->insert(std::move(stage));
    }

    {
      // Allows the planner to generate valid trajectories where the object remains in contact
      // with the support surface until it's lifted.
      auto stage = std::make_unique<mtc::stages::ModifyPlanningScene>("allow collision (object,ground)");
      stage->allowCollisions({ std::string("object") }, { std::string("ground") }, true);
      grasp->insert(std::move(stage));
    }

    {
      auto stage = std::make_unique<mtc::stages::ModifyPlanningScene>("attach object");
      stage->attachObject("object", hand_frame);
      // attach_object_stage = stage.get();
      grasp->insert(std::move(stage));
    }

    // {
    //   auto stage =
    //       std::make_unique<mtc::stages::MoveRelative>("lift object", cartesian_planner);
    //   stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
    //   stage->setMinMaxDistance(0.1, 0.3);
    //   stage->setIKFrame(hand_frame);
    //   stage->properties().set("marker_ns", "lift_object");
    //   stage->properties().set("trajectory_execution_info",
    //                 mtc::TrajectoryExecutionInfo().set__controller_names(controller_names));


    //   // Set upward direction
    //   geometry_msgs::msg::Vector3Stamped vec;
    //   vec.header.frame_id = "world";
    //   vec.vector.z = 1.0;
    //   stage->setDirection(vec);
    //   grasp->insert(std::move(stage));
    // }

    // {
    //   // Forbid collisions between the picked object and the support surface.
    //   // This is important after the object has been lifted to ensure it doesn't accidentally
    //   // collide with the surface during subsequent movements.
    //   auto stage = std::make_unique<mtc::stages::ModifyPlanningScene>("forbid collision (object,ground)");
    //   stage->allowCollisions({ std::string("object") }, { std::string("ground") }, false);
    //   grasp->insert(std::move(stage));
    // }


    task.add(std::move(grasp));
  }

  return task;
}


void generate_grasp_transform(Eigen::Isometry3d& T) {
  // desired axes in IK_frame coordinates (what G should look like)
  // choose: G.z = approach_dir (in G coords it's (0,0,1))
  //        G.x = bite_dir_in_Gcoords (we can choose x to be the bite axis)
  // For construction we want the IK axes in hand frame (call them ik_x_in_hand, ik_y_in_hand, ik_z_in_hand)

  // If we decide: IK_frame.x corresponds to hand +X (bite), IK_frame.z = some approach (perp)
  Eigen::Vector3d hand_bite_in_hand(1, 0, 0); // hand X in hand coords
  Eigen::Vector3d ik_z_in_hand(0, 0, 1);

  // Option A: If you want IK.x == hand.x, and IK.z some known orth perpendicular to it:
  Eigen::Vector3d ik_x_in_hand = hand_bite_in_hand.normalized();

  // now build rotation R whose columns are the IK axes in hand frame
  Eigen::Matrix3d R;
  R.col(0) = ik_x_in_hand; // IK x as expressed in hand frame
  R.col(2) = ik_z_in_hand;
  R.col(1) = R.col(2 ).cross(R.col(0));
  

  // translation: pick offset from hand frame to the contact point in hand frame
  Eigen::Vector3d t_in_hand(-0.05, 0.0, 0.0); // e.g., 0.1 m along hand +X (bite direction)

  // compose transform
  T.linear() = R;
  T.translation() = t_in_hand;
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  int ret = 0;

  try {
    rclcpp::NodeOptions options;
    options.automatically_declare_parameters_from_overrides(true);

    auto mtc_task_node = std::make_shared<MTCTaskNode>(options);
    rclcpp::executors::SingleThreadedExecutor executor;

    // auto spin_thread = std::make_unique<std::thread>([&executor, &mtc_task_node]() {
    //   executor.add_node(mtc_task_node->getNodeBaseInterface());
    //   executor.spin();
    //   executor.remove_node(mtc_task_node->getNodeBaseInterface());
    // });

    // mtc_task_node->setupPlanningScene();
    // mtc_task_node->doTask();

    // spin_thread->join();

    executor.add_node(mtc_task_node);

    // Set up the planning scene and execute the task
    try {
      RCLCPP_INFO(mtc_task_node->get_logger(), "[LOG] Setting up planning scene");
      mtc_task_node->setupPlanningScene();  
      RCLCPP_INFO(mtc_task_node->get_logger(), "[LOG] Executing task");
      mtc_task_node->doTask();

      // Keep the node running until Ctrl+C is pressed
      executor.spin();
    } catch (const std::runtime_error& e) {
      RCLCPP_ERROR(mtc_task_node->get_logger(), "[LOG] Runtime error occurred: %s", e.what());
      ret = 1;
    } catch (const std::exception& e) {
      RCLCPP_ERROR(mtc_task_node->get_logger(), "[LOG] An error occurred: %s", e.what());
      ret = 1;
    }
  } catch (const std::exception& e) {
    RCLCPP_ERROR(rclcpp::get_logger("main"), "[LOG] Error during node setup: %s", e.what());
    ret = 1;
  }

  rclcpp::shutdown();
  return ret;
}