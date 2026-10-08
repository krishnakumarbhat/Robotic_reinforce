#pragma once

#include <yaml-cpp/yaml.h>
#include <fstream>

void save(
  const std::string& filename,
  const std::map<std::string, int>& homing_offset,
  const std::map<std::string, int>& start_pos,
  const std::map<std::string, int>& end_pos,
  const std::map<std::string, int>& drive_mode_map
);