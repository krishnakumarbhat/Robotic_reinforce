import subprocess
import sys
import time

EXPECTED_TOPICS = [
    "/camera/camera/color/image_raw",
    "/camera/camera/aligned_depth_to_color/image_raw",
    "/camera/camera/color/camera_info",
    "/camera/camera/imu"
]

def get_topic_list():
    raw = subprocess.getoutput("ros2 topic list")
    return [t.strip() for t in raw.splitlines()]

def find_missing_topics(topic_list):
    return [t for t in EXPECTED_TOPICS if t not in topic_list]

def echo_topic(topic, n=1, timeout=3):
    """Returns first n messages (as text) from the topic."""
    cmd = f"timeout {timeout} ros2 topic echo -n {n} {topic}"
    result = subprocess.getoutput(cmd)
    return result

def parse_tf_static(tf_msg):
    """Checks for expected structure in a /tf_static message."""
    return all(s in tf_msg for s in ["header:", "child_frame_id:", "frame_id:"])

def main():
    print("\n##### ROS2 Realsense/IMU/TF Health Check #####\n")

    time.sleep(0.5)
    topics = get_topic_list()
    print("Available Topics:")
    for t in topics:
        print(f"- {t}")

    time.sleep(0.5)
    missing = find_missing_topics(topics)
    print("\nMissing Expected Topics:")
    if missing:
        for t in missing:
            print(f"!! {t}")
    else:
        print("None")

    # Check static transform on /tf_static
    print("\nTF Static message check:")
    tf_msg = echo_topic('/tf_static')
    if parse_tf_static(tf_msg):
        print("TF Static: Available and broadcasting!")
        # Print out the transform summary
        for l in tf_msg.splitlines():
            if 'parent:' in l or 'child_frame_id' in l or 'frame_id' in l:
                print("  " + l.strip())
    else:
        print("TF Static: NOT found or message malformed!")

    # Check IMU data
    print("\nIMU Data publishing:")
    imu_msg = echo_topic('/camera/camera/imu')
    if 'linear_acceleration' in imu_msg or 'angular_velocity' in imu_msg:
        print("IMU: Raw data publishing OK.")
    else:
        print("IMU: No data, check sensor and topic!")

    # Optionally check /tf (dynamic, may be empty)
    print("\nTF Dynamic topic check (can be empty):")
    tf_msg_dyn = echo_topic('/tf')
    if tf_msg_dyn and 'header:' in tf_msg_dyn:
        print("TF Dynamic: At least one message published.")
    else:
        print("TF Dynamic: No data (OK for static transform only).")

    print("\n##### Check Complete #####\n")

if __name__ == "__main__":
    main()
