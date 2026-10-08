#include <fmt/ranges.h>

#include <algorithm>
#include <feetech_driver/common.hpp>
#include <feetech_driver/communication_protocol.hpp>
#include <feetech_ros2_driver/feetech_ros2_driver.hpp>
#include <hardware_interface/types/hardware_interface_return_values.hpp>
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/all.hpp>
#include <rclcpp/rclcpp.hpp>
#include <string>
#include <string_view>
#include <vector>
#include <optional>

namespace feetech_ros2_driver {

std::unique_ptr<MotorCalibrator> motor_calibrator_;


#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

// TODO: function signature now deprecated. Refactor to newer, supported API 
CallbackReturn FeetechHardwareInterface::on_init(const hardware_interface::HardwareInfo& info) {
  if (hardware_interface::SystemInterface::on_init(info) != CallbackReturn::SUCCESS) {
    return CallbackReturn::ERROR;
  }

  // Get parameter usb_port
  const auto usb_port_it = info_.hardware_parameters.find("usb_port");
  if (usb_port_it == info_.hardware_parameters.end()) {
    spdlog::error(
        "FeetechHardware::on_init Hardware parameter [{}] not found!. "
        "Make sure to have <param name=\"usb_port\">/dev/XXXX</param>");
    return CallbackReturn::ERROR;
  }

  // get parameter disable_torque
  const auto disable_torque_it = info_.hardware_parameters.find("disable_torque");
  if (disable_torque_it != info_.hardware_parameters.end()) {
    disable_torque_globally_ = (disable_torque_it->second == "true");
    spdlog::info("Global torque disable parameter set to: {}", disable_torque_globally_);
  }

  // get use_offsets parameter
  const auto use_offsets_it = info_.hardware_parameters.find("use_offsets");
  if (use_offsets_it != info_.hardware_parameters.end()) {
    use_offsets_ = (use_offsets_it->second == "true");
    spdlog::info("use_offsets parameter set to: {}", use_offsets_);
  }

  // get parameter disable_torque
  const auto use_drive_modes_it = info_.hardware_parameters.find("use_drive_modes");
  if (use_drive_modes_it != info_.hardware_parameters.end()) {
    use_drive_modes_ = (use_drive_modes_it->second == "true");
    spdlog::info("use_drive_modes parameter set to: {}", use_drive_modes_);
  }

  auto serial_port = std::make_unique<feetech_driver::SerialPort>(usb_port_it->second);

  if (const auto result = serial_port->configure(); !result) {
    spdlog::error("FeetechHardware::on_init -> {}", result.error());
    return CallbackReturn::ERROR;
  }

  communication_protocol_ = std::make_unique<feetech_driver::CommunicationProtocol>(std::move(serial_port));

  joint_ids_.resize(info_.joints.size(), 0);
  joint_offsets_.resize(info_.joints.size(), 0);
  joint_start_pos_.resize(info_.joints.size(), 0);
  joint_end_pos_.resize(info_.joints.size(), 0);
  joint_drive_modes_.resize(info_.joints.size(), false);  // Default to false (positive direction);

  for (uint i = 0; i < info_.joints.size(); i++) {
    const std::string& joint_name = info_.joints[i].name; // CHANGES: Changed to joint_name ( string ) prev assumed to be a vector 
    const auto& joint_params = info_.joints[i].parameters;
    joint_ids_[i] = std::stoi(joint_params.at("id"));
    joint_offsets_[i] = [&] {
      if (!use_offsets_) {
        return 0;  // Default to 0 if offsets are not used
      }
      if (const auto offset_it = joint_params.find("offset"); offset_it != joint_params.end()) {
        return std::stoi(offset_it->second);
      }
      spdlog::info("Joint '{}' does not specify an offset parameter - Setting it to 0", joint_name);
      return 0;
    }();
    joint_start_pos_[i] = [&] {
      if (!use_offsets_) {
        return 0;  // Default to 0 if offsets are not used
      }
      if (const auto joint_start_pos_it = joint_params.find("start_position"); joint_start_pos_it != joint_params.end()) {
        return feetech_driver::from_radians(std::stod(joint_start_pos_it->second));
      }
      spdlog::info("Joint '{}' does not specify an start position parameter - Setting it to 0", joint_name);
      return 0;
    }();
    joint_end_pos_[i] = [&] {
      if (!use_offsets_) {
        return 0;  // Default to 0 if offsets are not used
      }
      if (const auto joint_end_pos_it = joint_params.find("end_position"); joint_end_pos_it != joint_params.end()) {
        return feetech_driver::from_radians(std::stod(joint_end_pos_it->second));
      }
      spdlog::info("Joint '{}' does not specify an end position parameter - Setting it to 0", joint_name);
      return 0;
    }();

    joint_drive_modes_[i] = [&] {
      if (!use_drive_modes_) {
        return false;  // Default to positive direction if drive modes are not used
      }
      if (const auto drive_mode_it = joint_params.find("drive_mode"); drive_mode_it != joint_params.end()) {
        spdlog::info("Joint '{}' drive mode: {}", info_.joints[i].name, drive_mode_it->second);
        auto result = static_cast<bool>((drive_mode_it->second) == "True");
        return result;
      }
      spdlog::info("Joint '{}' does not specify an drive_mode parameter - Setting it to default ( positive / no-drive )", info_.joints[i].name);
      return false;
    }();

    for (const auto& [parameter_name, address] : {std::pair{"p_cofficient", SMS_STS_P_COEF},
                                                  {"d_cofficient", SMS_STS_D_COEF},
                                                  {"i_cofficient", SMS_STS_I_COEF}}) {
      if (const auto param_it = joint_params.find(parameter_name); param_it != joint_params.end()) {
        const auto result = communication_protocol_->write(
            joint_ids_[i], address, std::experimental::make_array(static_cast<uint8_t>(std::stoi(param_it->second))));
        if (!result) {
          spdlog::error("FeetechHardwareInterface::on_init -> {}", result.error());
          return CallbackReturn::ERROR;
        }
      }
    }
    // Disable holding torque for joints that do not have command interfaces.
    if (info_.joints[i].command_interfaces.empty()) {
      spdlog::warn("Joint {} has no command interfaces. Disabling torque.", info_.joints[i].name);
      communication_protocol_->set_torque(joint_ids_[i], false);
    }

    // Disable torque globally if the parameter is set
    if (disable_torque_globally_) {
      spdlog::info("Disabling torque globally for joint ID {}", joint_ids_[i]);
      communication_protocol_->set_torque(joint_ids_[i], false);
    }
  }

  std::vector<std::optional<feetech_driver::ModelSeries>> joint_model_series;
  for (const auto id : joint_ids_) {
      spdlog::info("Reading model number for joint ID: {}", id);
      auto model_number_result = communication_protocol_->read_model_number(id);

      if (!model_number_result) {
          spdlog::error("Failed to read model number for ID: {}", id);
          joint_model_series.emplace_back(std::nullopt);
          continue;
      }

      auto model_number = *model_number_result;
      spdlog::info("Model number for ID {}: {}", id, model_number);

      auto model_name_result = feetech_driver::get_model_name(model_number);
      if (!model_name_result) {
          spdlog::error("Unknown model name for number {} (ID: {})", model_number, id);
          //joint_model_series.emplace_back(std::nullopt);
          spdlog::warn("Using default model name (STS3215) for ID: {}", id);
      }

      model_name_result = feetech_driver::get_model_name(DEFAULT_MODEL_NUM);
      auto model_name = *model_name_result;
      spdlog::info("Model name for ID {}: {}", id, model_name);

      auto model_series_result = feetech_driver::get_model_series(model_name);
      if (!model_series_result) {
          spdlog::error("Unknown model series for model name '{}' (ID: {})", model_name, id);
          joint_model_series.emplace_back(std::nullopt);
          continue;
      }

      auto model_series = *model_series_result;
      spdlog::info("Model series for ID {}: {}", id, static_cast<int>(model_series));

      joint_model_series.emplace_back(model_series);
  }
  
  if (std::ranges::any_of(joint_model_series, [](const auto& series) { return !series.has_value(); })) {
    // spdlog::error("FeetechHardware::on_init [One of the joints has an error]. Input: {}",
    //               ranges::views::zip(joint_ids_, joint_model_series));

    spdlog::error("FeetechHardware::on_init [One of the joints has an error]:");
    for (auto&& [id, series] : ranges::views::zip(joint_ids_, joint_model_series)) {
        spdlog::error("  ID: {}, Series: {}", id, series.has_value() ? std::to_string(static_cast<int>(*series)) : "<none>");
    }
    return CallbackReturn::ERROR;
  }

  const auto js = joint_model_series | ranges::views::transform([](const auto& series) { return series.value(); });

  // TODO: Support other series
  if (ranges::any_of(js, [](const auto& series) { return series != feetech_driver::ModelSeries::kSts; })) {
    // spdlog::error("FeetechHardware::on_init [Only STS series is supported]. Input (id, series): {}",
    //               ranges::views::zip(joint_ids_, js));
    std::string debug_str;
    for (auto&& [id, series] : ranges::views::zip(joint_ids_, js)) {
        debug_str += fmt::format("({}, {}), ", id, static_cast<int>(series));
    }
    spdlog::error("FeetechHardware::on_init [Only STS series is supported]. Input (id, series): [{}]", debug_str);
    return CallbackReturn::ERROR;
  }

  // Initialize motor calibrator
  std::vector<CalibrationEntry> calibration_entries;
  for (size_t i = 0; i < joint_ids_.size(); ++i) {
    calibration_entries.emplace_back(
      CalibrationEntry{
        .motor_name = info_.joints[i].name,
        .homing_offset = joint_offsets_[i],
        .drive_mode = joint_drive_modes_[i],
        .start_pos = joint_start_pos_[i],
        .end_pos = joint_end_pos_[i]
      }
    );
  }

  motor_calibrator_ = std::make_unique<MotorCalibrator>(
    calibration_entries,
    joint_ids_.size()  // Count of motors
  );

  spdlog::info("FeetechHardware::on_init [Motor calibrator initialized with {} motors]. Calibration entry: {}",
               info_.joints.size(),
               ranges::views::zip(info_.joints, joint_ids_) | ranges::views::transform([](const auto& pair) {
                 const auto& [joint, id] = pair;
                 return fmt::format("({}, {})", joint.name, id);
               }) | ranges::to<std::vector>());

  return CallbackReturn::SUCCESS;
}

#pragma GCC diagnostic pop

std::vector<hardware_interface::StateInterface> FeetechHardwareInterface::export_state_interfaces() {
  std::vector<hardware_interface::StateInterface> state_interfaces;
  state_hw_positions_.resize(info_.joints.size(), 0.0);
  state_hw_velocities_.resize(info_.joints.size(), 0.0);
  for (uint i = 0; i < info_.joints.size(); i++) {
    state_interfaces.emplace_back(info_.joints[i].name, hardware_interface::HW_IF_POSITION, &state_hw_positions_[i]);
    state_interfaces.emplace_back(info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &state_hw_velocities_[i]);
  }

  return state_interfaces;
}

std::vector<hardware_interface::CommandInterface> FeetechHardwareInterface::export_command_interfaces() {
  std::vector<hardware_interface::CommandInterface> command_interfaces;
  hw_positions_.resize(info_.joints.size(), std::numeric_limits<double>::quiet_NaN());
  for (uint i = 0; i < info_.joints.size(); i++) {
    if (!info_.joints[i].command_interfaces.empty()) {
      command_interfaces.emplace_back(info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_positions_[i]);
    }
  }

  return command_interfaces;
}

int raw_readings[6];

hardware_interface::return_type FeetechHardwareInterface::read(const rclcpp::Time& /* time */,
                                                               const rclcpp::Duration& /* period */) {
  // 4 = 2 bytes for position + 2 bytes for speed
  std::vector<std::array<uint8_t, 4>> data;
  data.reserve(joint_ids_.size());
  if (auto result = communication_protocol_->sync_read(joint_ids_, SMS_STS_PRESENT_POSITION_L, &data); !result) {
    spdlog::error("FeetechHardwareInterface::read -> {}", result.error());
    return hardware_interface::return_type::ERROR;
  }

  if (use_offsets_) {
      ranges::for_each(data | ranges::views::enumerate, [&](const auto& values) {
      const auto& [index, readings] = values;
      raw_readings[index] = feetech_driver::from_sts(feetech_driver::WordBytes{.low = readings[0], .high = readings[1]});
      state_hw_positions_[index] = feetech_driver::to_radians(
        motor_calibrator_->apply_calibration(raw_readings[index], joint_ids_[index])
      );      
      state_hw_velocities_[index] = feetech_driver::to_radians(
          feetech_driver::from_sts(feetech_driver::WordBytes{.low = readings[2], .high = readings[3]}));
    });
  }
  else {
    ranges::for_each(data | ranges::views::enumerate, [&](const auto& values) {
      const auto& [index, readings] = values;
      state_hw_positions_[index] = feetech_driver::to_radians(
        feetech_driver::from_sts(feetech_driver::WordBytes{.low = readings[0], .high = readings[1]})
      );
      state_hw_velocities_[index] = feetech_driver::to_radians(
          feetech_driver::from_sts(feetech_driver::WordBytes{.low = readings[2], .high = readings[3]}));
    });
  }
  
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type FeetechHardwareInterface::write(const rclcpp::Time& /* time */,
                                                                const rclcpp::Duration& /* period */) {
  // Create vectors only for joints that have command interfaces
  std::vector<uint8_t> commanded_joint_ids;
  std::vector<int> commanded_positions;
  std::vector<int> commanded_speeds;
  std::vector<int> commanded_accelerations;

  for (uint i = 0; i < info_.joints.size(); i++) {
    // Only include joints with command interfaces
    if (!info_.joints[i].command_interfaces.empty()) {
      commanded_joint_ids.push_back(joint_ids_[i]);
      commanded_speeds.push_back(480);       // Default speed (2400)
      commanded_accelerations.push_back(20);  // Default acceleration (50)

      if (use_offsets_) {
        commanded_positions.push_back(
          motor_calibrator_->revert_calibration(
            static_cast<int>(hw_positions_[i]), 
            joint_ids_[i]
          )
        );
      }
      else {
        commanded_positions.push_back(
            feetech_driver::from_radians(hw_positions_[i])
        );
      }
    }
  }

  // Only send commands if there are joints to command
  if (!commanded_joint_ids.empty() && !disable_torque_globally_) {
    // spdlog::info("Writing positions: Torque disable globally: {}", disable_torque_globally_);
    const auto write_result = communication_protocol_->sync_write_position(
        commanded_joint_ids, commanded_positions, commanded_speeds, commanded_accelerations);
    if (!write_result) {
      spdlog::error("FeetechHardwareInterface::write -> {}", write_result.error());
      return hardware_interface::return_type::ERROR;
    }
  }

  return hardware_interface::return_type::OK;
}

CallbackReturn FeetechHardwareInterface::on_activate(const rclcpp_lifecycle::State& /* previous_state */) {
  // Time/Duration are not used
  read(rclcpp::Time{}, rclcpp::Duration::from_seconds(0));
  // Set the initial command to current joint positions
  hw_positions_ = state_hw_positions_;
  return CallbackReturn::SUCCESS;
}

CallbackReturn FeetechHardwareInterface::on_deactivate(const rclcpp_lifecycle::State& /* previous_state */) {
  // all joints torque off
  const auto torque_disable_parameters =
      std::vector(joint_ids_.size(), std::experimental::make_array(static_cast<uint8_t>(0)));
  if (const auto result =
          communication_protocol_->sync_write(joint_ids_, SMS_STS_TORQUE_ENABLE, torque_disable_parameters);
      !result) {
    spdlog::error("FeetechHardwareInterface::on_deactivate -> {}", result.error());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

}  // namespace feetech_ros2_driver

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(feetech_ros2_driver::FeetechHardwareInterface, hardware_interface::SystemInterface)
