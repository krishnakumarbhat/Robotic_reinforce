# CleanBot Development Guidelines

## Build/Test Commands
```bash
# Build all packages
colcon build --symlink-install

# Build specific package
colcon build --packages-up-to <package_name>

# Run tests
colcon test && colcon test-result

# Run tests for specific package
colcon test --packages-select <package_name>

# Run single test
colcon test --packages-select <package_name> --ctest-args -R <test_name>
```

## Code Style Guidelines

### C++ (Google Style)
- Format: Google style with 120-character limit, Allman braces
- Pointers/References: Left-aligned (`int* ptr`, `int& ref`)
- Qualifiers: Left-aligned
- Naming: `CamelCase` classes, `snake_case` functions/variables, `kCamelCase` constants
- Imports: Use standard ROS2 includes, avoid `using namespace`
- Pre-commit: clang-format auto-applied on commit

### Python
- Linter: ruff (flake8 compatible) with auto-fix
- Formatter: ruff-format
- Docstrings: PEP 257 compliant
- Naming: `snake_case` for functions/variables, `CamelCase` classes
- Imports: Group stdlib, third-party, and local imports
- Pre-commit: ruff check/format auto-applied on commit

### Formatting Commands
```bash
# C++ formatting (auto-applied by pre-commit)
cd src/feetech_ros2_driver && clang-format -i src/**/*.cpp include/**/*.hpp

# Python formatting (auto-applied by pre-commit)  
cd src/move_api && ruff check --fix . && ruff format .

# Run all pre-commit hooks manually
cd src/feetech_ros2_driver && pre-commit run --all-files
```

## Project Structure
- ROS2 workspace using colcon build system
- C++14 standard (C++17 for MoveIt packages)
- Submodule: `feetech_ros2_driver` for servo control
- Testing: ament_lint for C++, pytest for Python
- XML/URDF formatting: prettier with @prettier/plugin-xml

## Error Handling
- Use ROS2 logging (`RCLCPP_*`, `RCLCPP_ERROR`)
- Return appropriate error codes from functions
- Validate parameters in launch files and nodes