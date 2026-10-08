#include <clean_bot_controller/utils.hpp>

void save(
  const std::string& filename,
  const std::map<std::string, int>& homing_offset,
  const std::map<std::string, int>& start_pos,
  const std::map<std::string, int>& end_pos,
  const std::map<std::string, int>& drive_mode_map
) {
  YAML::Node root;
  YAML::Node offset_node;
  YAML::Node start_pos_node;
  YAML::Node end_pos_node;
  YAML::Node drive_mode_node;

  for (const auto& [joint, offset] : homing_offset) {
    offset_node[joint] = offset;
  }

  for (const auto& [joint, start] : start_pos) {
    start_pos_node[joint] = start;
  }

  for (const auto& [joint, end] : end_pos) {
    end_pos_node[joint] = end;
  }

  for (const auto& [joint, mode] : drive_mode_map) {
    drive_mode_node[joint] = mode;
  }

  root["homing_offset"] = offset_node;
  root["start_position"] = start_pos_node;
  root["end_position"] = end_pos_node;
  root["drive_mode"] = drive_mode_node;

  std::ofstream fout(filename);
  fout << root;
  fout.close();
}
