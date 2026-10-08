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
        if (result > 2100 || result < 2000) {
            spdlog::debug("[{}] add_offset: {} + {} = {}", calib.motor_name, value, calib.homing_offset, result);
        }
        // Step 2: Normalize to 0-4095 range
        result = normalize(result);
        if (result > 2100 || result < 2000) {
            spdlog::debug("[{}] normalize: x % 4095 = {}", calib.motor_name, result);
        }
        // Step 3: Remap from 0→2π to -π→π range
        result = remap_to_centered(result);
        if (result > 50 || result < -50) {
            spdlog::debug("[{}] remap_to_centered: x - 2048 = {}", calib.motor_name, result);
        }
        // Step 4: Apply drive mode x -> -x
        result = apply_drive_mode(result, calib.drive_mode);
        if (result > 50 || result < -50) {
            spdlog::debug("[{}] drive_mode applied: {}", calib.motor_name, calib.drive_mode);
        }
        
        return static_cast<int>(result);
    }
    catch (const std::exception& e) {
        spdlog::error("apply_calibration failed for motor {}: {}", motor_index, e.what());
        return value; // Return original value on error
    }
}

int MotorCalibrator::revert_calibration(double value, size_t motor_index) {
    try {
        if (motor_index == 0 || motor_index > calibration_data_.size()) {
            spdlog::error("revert_calibration failed - Invalid motor index: {}", motor_index);
            return value;
        }

        size_t i = motor_index - 1;  // Adjust for zero-based index
        const auto& calib = calibration_data_[i];

        // Step 1: Remove drive mode:
        double val = apply_drive_mode(static_cast<double>(value), calib.drive_mode);
        
        // Step 2: Convert from radians to ticks using existing utility
        double result = feetech_driver::from_radians(val);
        if (result > 50 || result < -50) {
            spdlog::debug("[{}] rad_to_ticks: {} rad -> {} ticks", calib.motor_name, val, result);
        }
        // Step 3: Remap from -π→π to 0→2π range
        result = remap_to_zero_based(result);
        if (result > 2100 || result < 2000) {
            spdlog::debug("[{}] remap_to_zero_based: x + 2048 = {}", calib.motor_name, result);
        }
        // Step 4: Normalize to 0-4095 range
        result = normalize(result);
        if (result > 2100 || result < 2000) {
            spdlog::debug("[{}] normalize: x % 4095 = {}", calib.motor_name, result);
        }
        // Step 5: Remove homing offset
        result = remove_offset(result, static_cast<double>(calib.homing_offset));
        if (result > 2100 || result < 2000) {
            spdlog::debug("[{}] remove_offset: x - {} = {}", calib.motor_name, calib.homing_offset, result);
        }
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
    return drive_mode ? val * -1.0 : val;
}

