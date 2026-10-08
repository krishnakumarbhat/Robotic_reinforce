#include "feetech_ros2_driver/motor_calibrator.hpp"
#include "feetech_driver/common.hpp"


MotorCalibrator::MotorCalibrator(std::vector<CalibrationEntry> calib, size_t count)
    : calibration_data_(std::move(calib)), motor_count_(count) {}



int MotorCalibrator::apply_calibration(int value, size_t motor_index) {
    try {
        if (motor_index == 0 || motor_index > calibration_data_.size()) {
            spdlog::error("apply_calibration failed - Invalid motor index: {}", motor_index);
            return value;
        }

        size_t i = motor_index - 1;  // Adjust for zero-based index
        const auto& calib = calibration_data_[i];
        
        // Step 1: Add homing offset
        double result = add_offset(static_cast<double>(value), static_cast<double>(calib.homing_offset));
        spdlog::debug("[{}] add_offset: {} + {} = {}", calib.motor_name, value, calib.homing_offset, result);
        
        // Step 2: Normalize to 0-4095 range
        result = normalize(result);
        spdlog::debug("[{}] normalize: {} % 4095 = {}", calib.motor_name, result, result);
        
        // Step 3: Remap from 0→2π to -π→π range
        result = remap_to_centered(result);
        spdlog::debug("[{}] remap_to_centered: {} - 2048 = {}", calib.motor_name, result, result);

        // Step 4: Apply drive mode x -> -x
        result = apply_drive_mode(result, calib.drive_mode);
        
        return static_cast<int>(result);
    }
    catch (const std::exception& e) {
        spdlog::error("apply_calibration failed for motor {}: {}", motor_index, e.what());
        return value; // Return original value on error
    }
}

int MotorCalibrator::revert_calibration(int value, size_t motor_index) {
    try {
        if (motor_index == 0 || motor_index > calibration_data_.size()) {
            spdlog::error("revert_calibration failed - Invalid motor index: {}", motor_index);
            return value;
        }

        size_t i = motor_index - 1;  // Adjust for zero-based index
        const auto& calib = calibration_data_[i];

        // Step 1: Remove drive mode:
        double result = apply_drive_mode(static_cast<double>(value), calib.drive_mode);
        
        // Step 2: Convert from radians to ticks using existing utility
        result = feetech_driver::from_radians(result);
        spdlog::debug("[{}] rad_to_ticks: {} rad -> {} ticks", calib.motor_name, result, result);
        
        // Step 3: Remap from -π→π to 0→2π range
        result = remap_to_zero_based(result);
        spdlog::debug("[{}] remap_to_zero_based: {} + 2048 = {}", calib.motor_name, result, result);
        
        // Step 4: Normalize to 0-4095 range
        result = normalize(result);
        spdlog::debug("[{}] normalize: {} % 4095 = {}", calib.motor_name, result, result);
        
        // Step 5: Remove homing offset
        result = remove_offset(result, static_cast<double>(calib.homing_offset));
        spdlog::debug("[{}] remove_offset: {} - {} = {}", calib.motor_name, result, calib.homing_offset, result);
        
        return static_cast<int>(result);
    }
    catch (const std::exception& e) {
        spdlog::error("revert_calibration failed for motor {}: {}", motor_index, e.what());
        return value; // Return original value on error
    }
}

// Helper function implementations
double MotorCalibrator::add_offset(double value, double offset) {
    return value + offset;
}

int MotorCalibrator::normalize(int value) {
    return value % 4095;
}

double MotorCalibrator::normalize(double value) {
    return std::fmod(value, 4095.0);
}

double MotorCalibrator::remap_to_centered(double value) {
    return value - 2048.0;
}

int MotorCalibrator::remap_to_zero_based(int value) {
    return value + 2048;
}

int MotorCalibrator::remove_offset(int value, int offset) {
    return value - offset;
}

double MotorCalibrator::apply_drive_mode(double val, bool drive_mode) {
    return drive_mode ? val * -1 : val;
}

