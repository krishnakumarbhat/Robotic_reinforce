# SLAM Robot Orchestrator

A ROS2-based Simultaneous Localization and Mapping (SLAM) application with RealSense camera support, designed to run on any device including Raspberry Pi using Docker.

## Features

- **SLAM Mapping**: Create 3D maps of environments using RealSense cameras
- **Map Analysis**: Analyze saved maps for path planning and object recognition
- **Cross-Platform**: Runs on any device with Docker support (x86, ARM64, Raspberry Pi)
- **RealSense Integration**: Full support for Intel RealSense cameras
- **ROS2 Ecosystem**: Built on ROS2 Humble with RTAB-Map

## Prerequisites

- Docker and Docker Compose installed
- Intel RealSense camera (D415, D435i, etc.)
- USB 3.0 port for camera connection
- X11 forwarding support (for GUI applications)

### Installing Docker on Raspberry Pi

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt install docker-compose-plugin

# Reboot to apply changes
sudo reboot
```

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd slam-robot-orchestrator
   ```

2. **Build the Docker image**:
   ```bash
   docker-compose build
   ```

3. **Run the application**:
   ```bash
   docker-compose up
   ```

4. **Access the application**:
   - The main menu will appear in the terminal
   - Choose option 1 to start SLAM mapping
   - Choose option 2 to analyze saved maps

## Usage

### Running SLAM Mapping

1. Select option 1 from the main menu
2. The system will launch:
   - RealSense camera driver
   - RTAB-Map SLAM algorithm
   - RViz2 visualization (if GUI available)
3. Physically move the camera around the environment
4. Press Enter when finished mapping
5. The map will be exported as a .ply file

### Analyzing Maps

1. Select option 2 from the main menu
2. Choose analysis options:
   - Path planning (conceptual)
   - Object recognition (conceptual)

## Docker Commands

### Basic Usage

```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f
```

### Advanced Usage

```bash
# Run with GUI support (RViz2)
docker-compose --profile gui up

# Run without RealSense camera (for testing)
docker-compose run --rm slam-app python3 /workspace/scripts/main.py

# Access container shell
docker-compose exec slam-app bash

# Run specific script
docker-compose run --rm slam-app python3 /workspace/scripts/slam_manager.py
```

## Configuration

### Environment Variables

- `ROS_DOMAIN_ID`: ROS2 domain ID (default: 0)
- `DISPLAY`: X11 display for GUI applications

### Volume Mounts

- `./data:/workspace/data`: Persistent data storage
- `~/.ros:/root/.ros`: ROS configuration and maps
- `/dev:/dev`: Device access for cameras

### Device Access

The container runs with privileged access to enable:
- USB device access for RealSense cameras
- Video device access
- Network access for ROS2 communication

## Troubleshooting

### Camera Not Detected

```bash
# Check USB devices
lsusb

# Check video devices
ls /dev/video*

# Restart USB services
sudo systemctl restart udev
```

### GUI Issues

```bash
# Allow X11 connections
xhost +local:docker

# Check DISPLAY variable
echo $DISPLAY
```

### Permission Issues

```bash
# Fix Docker permissions
sudo chmod 666 /var/run/docker.sock

# Add user to docker group
sudo usermod -aG docker $USER
```

### Memory Issues (Raspberry Pi)

```bash
# Increase swap space
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Set CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

## File Structure

```
slam-robot-orchestrator/
├── Dockerfile                 # Docker image definition
├── docker-compose.yml         # Docker Compose configuration
├── package.xml               # ROS2 package definition
├── CMakeLists.txt            # ROS2 build configuration
├── scripts/                  # Python application scripts
│   ├── main.py              # Main orchestrator
│   ├── slam_manager.py      # SLAM process management
│   └── analysis_manager.py  # Map analysis tools
├── launch/                   # ROS2 launch files
│   └── slam_launch.py       # SLAM system launch
├── config/                   # Configuration files
│   └── realsense/           # RealSense utilities
└── data/                     # Persistent data storage
```

## Development

### Building for Different Architectures

```bash
# Build for ARM64 (Raspberry Pi)
docker buildx build --platform linux/arm64 -t slam-app:arm64 .

# Build for x86_64
docker buildx build --platform linux/amd64 -t slam-app:amd64 .

# Build multi-platform
docker buildx build --platform linux/amd64,linux/arm64 -t slam-app:latest .
```

### Customizing the Build

1. Modify `Dockerfile` for different base images or dependencies
2. Update `docker-compose.yml` for different configurations
3. Add new launch files in the `launch/` directory
4. Extend the Python scripts in the `scripts/` directory

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with Docker
5. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section
- Review Docker and ROS2 documentation
- Open an issue on GitHub
