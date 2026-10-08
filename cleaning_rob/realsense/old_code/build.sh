#!/bin/bash

# SLAM Robot Orchestrator Build Script
# This script builds and runs the Docker container for the SLAM application

set -e

echo "🚀 SLAM Robot Orchestrator - Docker Build Script"
echo "================================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if RealSense camera is connected
echo "🔍 Checking for RealSense camera..."
if lsusb | grep -q "Intel Corp"; then
    echo "✅ RealSense camera detected"
else
    echo "⚠️  No RealSense camera detected. The application will still run but camera features may not work."
fi

# Check X11 forwarding for GUI
if [ -n "$DISPLAY" ]; then
    echo "✅ X11 forwarding available for GUI applications"
    xhost +local:docker 2>/dev/null || echo "⚠️  Could not configure X11 forwarding"
else
    echo "⚠️  No X11 forwarding detected. GUI applications may not work."
fi

# Create data directory if it doesn't exist
mkdir -p data

echo ""
echo "🔨 Building Docker image..."
docker compose build

echo ""
echo "🎯 Starting SLAM Robot Orchestrator..."
echo "Press Ctrl+C to stop the application"
echo ""

# Run the application
docker compose up

echo ""
echo "👋 Application stopped. Run './build.sh' again to restart." 