import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import time

def main():
    # ---------------------------------------------------------------------------- #
    # 1. Configure and start the RealSense pipeline                                #
    # ---------------------------------------------------------------------------- #
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    # Enable depth and color streams
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Get camera intrinsics for 3D reconstruction
    intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    # ---------------------------------------------------------------------------- #
    # 2. Setup Open3D for visualization and SLAM                                   #
    # ---------------------------------------------------------------------------- #
    # This is a core component of SLAM, tracking frame-to-frame movement
    odometry_option = o3d.pipelines.odometry.OdometryOption()
    
    # Initialize the global map and the current camera pose
    global_pcd = o3d.geometry.PointCloud()
    current_pose = np.identity(4) # 4x4 identity matrix

    # Create a visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window("Live SLAM")
    is_first_frame = True

    print("Move the camera around to build the map. Press Esc to exit.")

    try:
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert images to Open3D format
            depth_image = o3d.geometry.Image(np.asanyarray(depth_frame.get_data()))
            color_image = o3d.geometry.Image(np.asanyarray(color_frame.get_data()))
            
            # Create an RGBD image from color and depth
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_image, depth_image, convert_rgb_to_intensity=False)
            
            # Process the first frame
            if is_first_frame:
                # Create a point cloud from the first frame
                current_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                    rgbd_image, pinhole_camera_intrinsic)
                
                # Add the first point cloud to the visualizer
                global_pcd = current_pcd
                vis.add_geometry(global_pcd)
                
                source_rgbd_image = rgbd_image
                is_first_frame = False
            
            # Process subsequent frames
            else:
                target_rgbd_image = rgbd_image

                # Compute the transformation (odometry) from the previous frame to the current one
                success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
                    source_rgbd_image, target_rgbd_image, pinhole_camera_intrinsic,
                    np.identity(4), o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), odometry_option)

                # If odometry is successful, update the camera pose and the global map
                if success:
                    # Update the overall camera pose
                    current_pose = np.dot(current_pose, trans)

                    # Create a point cloud from the current frame
                    current_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                        target_rgbd_image, pinhole_camera_intrinsic)
                    
                    # Transform the current point cloud to the global coordinate frame
                    current_pcd.transform(current_pose)
                    
                    # Add the new point cloud to the global map
                    global_pcd += current_pcd
                    
                    # Downsample the global map to keep it manageable
                    global_pcd = global_pcd.voxel_down_sample(voxel_size=0.01)

                    # Update the visualizer
                    vis.update_geometry(global_pcd)

                # The current frame becomes the source for the next iteration
                source_rgbd_image = target_rgbd_image

            # Update the rendering
            vis.poll_events()
            vis.update_renderer()
            
    finally:
        pipeline.stop()
        vis.destroy_window()

if __name__ == "__main__":
    main()