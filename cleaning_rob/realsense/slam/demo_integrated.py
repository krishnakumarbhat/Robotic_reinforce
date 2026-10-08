#!/usr/bin/env python3
"""Integrated demo showing all SLAM components working together.

This script demonstrates how the five projects integrate:
1. Change Detection Monitor
2. Semantic Mapping Pipeline
3. Dynamic Object Tracking
4. Autonomous Navigation (conceptual)
5. Real-time Indoor Mapping

Run without ROS/hardware for simulation mode.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np

# Import all core components
from realsense.slam import (
    # Core ASMTCDR
    IntegratedMappingAndMonitoringSystem,
    MapUpdate,
    # Room mapping
    RoomMapLogger,
    run_autonomous_room_mapping,
    # Dynamic tracking
    DynamicObjectTracker,
    # Semantic mapping
    SemanticMappingPipeline,
    Detection2D,
)


def demo_integrated_system(duration_s: int = 30) -> Dict[str, Any]:
    """Run the complete integrated ASMTCDR system."""
    
    print("\n" + "="*70)
    print("ASMTCDR Integrated System Demo")
    print("="*70)
    print(f"\n🤖 Starting {duration_s}-second autonomous mapping session...")
    print("\nThis demo combines all 5 projects:")
    print("  1. Change Detection Monitor")
    print("  2. Semantic Mapping Pipeline")
    print("  3. Dynamic Object Tracking")
    print("  4. Autonomous Navigation (conceptual)")
    print("  5. Real-time Indoor Mapping\n")
    
    # Run the integrated system
    summary = run_autonomous_room_mapping(
        duration_s=duration_s,
        output_dir=Path("artifacts/demo_integrated")
    )
    
    print("\n" + "="*70)
    print("✅ Session Complete!")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"  - Duration: {summary.get('elapsed_seconds', 0):.1f}s")
    print(f"  - Map updates: {summary.get('map_update_count', 0)}")
    print(f"  - Semantic events: {len(summary.get('semantic_events', []))}")
    print(f"  - Change events: {len(summary.get('change_events', []))}")
    print(f"  - Dynamic objects: {len(summary.get('dynamic_objects', []))}")
    
    return summary


def demo_semantic_mapping() -> None:
    """Demonstrate semantic mapping pipeline standalone."""
    
    print("\n" + "="*70)
    print("Semantic Mapping Pipeline Demo (Project 2)")
    print("="*70 + "\n")
    
    # Create pipeline
    pipeline = SemanticMappingPipeline(
        classes=["chair", "table", "monitor", "person", "backpack"]
    )
    
    # Simulate RGB-D frames
    print("🎥 Processing simulated RGB-D frames...")
    for i in range(5):
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth = np.random.randint(500, 5000, (480, 640), dtype=np.uint16)
        
        summary = pipeline.build_semantic_map_step(rgb, depth)
        
        print(f"\n  Frame {i+1}:")
        print(f"    - Objects detected: {summary.get('detected_objects', 0)}")
        print(f"    - Points labeled: {summary.get('labeled_points', 0)}")
        if 'labels' in summary:
            print(f"    - Labels: {', '.join(summary['labels'])}")
    
    print(f"\n✅ Total points in map: {len(pipeline.get_current_map().points)}")


def demo_dynamic_tracking() -> None:
    """Demonstrate dynamic object tracking standalone."""
    
    print("\n" + "="*70)
    print("Dynamic Object Tracking Demo (Project 3)")
    print("="*70 + "\n")
    
    tracker = DynamicObjectTracker()
    
    print("🎯 Tracking dynamic objects across frames...")
    for i in range(5):
        depth = np.random.randint(500, 5000, (480, 640), dtype=np.uint16)
        imu = np.array([
            np.random.uniform(-0.2, 0.2),  # Accel X
            np.random.uniform(-0.2, 0.2),  # Accel Y
            9.8  # Accel Z (gravity)
        ])
        
        print(f"\n  Frame {i+1}:")
        tracker.track_dynamic_objects(depth, imu)
    
    print(f"\n✅ Tracked objects: {len(tracker.tracked_objects)}")
    for obj_id, obj_data in tracker.tracked_objects.items():
        print(f"  - Object {obj_id}: {len(obj_data['history'])} observations")


def demo_all_components() -> None:
    """Run all component demos in sequence."""
    
    print("\n🚀 Running All Component Demos\n")
    
    # Demo 1: Semantic Mapping
    demo_semantic_mapping()
    time.sleep(1)
    
    # Demo 2: Dynamic Tracking
    demo_dynamic_tracking()
    time.sleep(1)
    
    # Demo 3: Integrated System
    demo_integrated_system(duration_s=10)
    
    print("\n" + "="*70)
    print("🎉 All Demos Complete!")
    print("="*70)
    print("\nNext steps:")
    print("  - Check artifacts/ directory for saved session data")
    print("  - Run with real RealSense hardware using ROS launch")
    print("  - Explore individual modules in realsense/slam/")
    print()


def main() -> None:
    """Main entry point with command-line options."""
    
    parser = argparse.ArgumentParser(
        description="ASMTCDR Integrated System Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all demos
  python3 demo_integrated.py --all

  # Run integrated system for 60 seconds
  python3 demo_integrated.py --integrated --duration 60

  # Run semantic mapping demo only
  python3 demo_integrated.py --semantic

  # Run dynamic tracking demo only
  python3 demo_integrated.py --tracking
        """
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all component demos"
    )
    parser.add_argument(
        "--integrated",
        action="store_true",
        help="Run integrated system demo"
    )
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Run semantic mapping demo"
    )
    parser.add_argument(
        "--tracking",
        action="store_true",
        help="Run dynamic tracking demo"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Duration for integrated demo (seconds)"
    )
    
    args = parser.parse_args()
    
    # If no specific demo selected, run all
    if not any([args.all, args.integrated, args.semantic, args.tracking]):
        args.all = True
    
    # Run selected demos
    if args.all:
        demo_all_components()
    else:
        if args.semantic:
            demo_semantic_mapping()
        if args.tracking:
            demo_dynamic_tracking()
        if args.integrated:
            demo_integrated_system(duration_s=args.duration)


if __name__ == "__main__":
    main()
