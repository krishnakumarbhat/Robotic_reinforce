#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <optional>
#include <spdlog/spdlog.h>

#if __has_include(<tl/expected.hpp>)
#include <tl/expected.hpp>
#else
#include <tl_expected/expected.hpp>
#endif



constexpr double kHalfTurnValue = 2048.0F;
constexpr double kModelResolution = 4096.0F;

struct JointOutOfRangeError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct CalibrationEntry {
    std::string motor_name;
    int homing_offset;
    bool drive_mode;
    int start_pos;
    int end_pos;
};

class MotorCalibrator {
private:
    std::vector<CalibrationEntry> calibration_data_;
    size_t motor_count_;
    std::vector<std::optional<int>> prev_positions_;

    // Helper functions for apply_calibration
    double add_offset(double value, double offset);
    double normalize(double value); 
    double remap_to_centered(double value);

    // Helper functions for revert_calibration  
    int normalize(int value); 
    int remap_to_zero_based(int value);
    int remove_offset(int value, int offset);

    double apply_drive_mode(double val, bool drive_mode);

public:
    MotorCalibrator(std::vector<CalibrationEntry> calib, size_t count);

    int apply_calibration(int value, size_t motor_index);
    int revert_calibration(double value, size_t motor_index);
 };
