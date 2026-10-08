# User Guide

## ros2_control urdf tag

The feetech system interface has a few `ros2_control` urdf tags to customize its behavior.

#### Parameters

* usb_port: (required). Example: `<param name="usb_port">/dev/my_robot</param>`.

#### Per-joint Parameters

Make sure to look at [Memory table](https://docs.google.com/spreadsheets/d/1GVs7W1VS1PqdhA1nW-abeyAHhTUxKUdR/edit?gid=364516031#gid=364516031) for a detailed explanation of the parameters.

* id: (required). ID of the servo. Example: `<param name="id">1</param>`.
* offset (default: 0). Offset of the servo's zero position. Example: `<param name="offset">2048</param>`.
* p_cofficient (optional): Proportional coefficient of the PID controller. Example: `<param name="p_cofficient">8</param>`.
* i_cofficient (optional): Integral coefficient of the PID controller. Example: `<param name="i_cofficient">0</param>`.
* d_cofficient (optional): Derivative coefficient of the PID controller. Example: `<param name="d_cofficient">32</param>`.

### Example

Take a look at [ros2_so_arm100](https://github.com/JafarAbdi/ros2_so_arm100/blob/main/so_arm100_description/control/so_arm100.ros2_control.xacro) for an example of how to use the tags.
