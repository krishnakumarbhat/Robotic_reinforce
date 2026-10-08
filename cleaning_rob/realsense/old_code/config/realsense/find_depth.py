import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

# To subscribe to two topics at once, we use the message_filters library
import message_filters

class DepthFinder(Node):
    def __init__(self):
        super().__init__('depth_finder_node')
        
        # Create a CvBridge to convert ROS images to OpenCV format
        self.bridge = CvBridge()

        # === IMPORTANT ===
        # We need to subscribe to the ALIGNED depth image and the color image.
        # The 'message_filters' library helps us get messages from both topics
        # that have the same timestamp. This is crucial for matching pixels.
        
        # Subscriber for the color image
        # self.color_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        
        # Subscriber for the aligned depth image
        # self.depth_sub = message_filters.Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')
        self.color_sub = message_filters.Subscriber(self, Image, '/camera/camera/color/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw')
        # Create a synchronizer to get both frames at the same time
        self.ts = message_filters.TimeSynchronizer([self.color_sub, self.depth_sub], 10)
        
        # Register the callback that will be executed when we have a synced pair of messages
        self.ts.registerCallback(self.image_callback)
        
        self.get_logger().info('Depth Finder Node has been started. Waiting for images...')

    def image_callback(self, color_msg, depth_msg):
        """
        This function is called every time we receive a synced pair of color and depth images.
        """
        try:
            # Convert the ROS Image messages to OpenCV images
            color_image = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
            # The depth image is a 16-bit single-channel image
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Failed to convert images: {e}')
            return

        # Get the dimensions of the image
        height, width, _ = color_image.shape
        
        # --- Pick a point to measure depth ---
        # We will measure the depth at the center of the image.
        center_x = width // 2
        center_y = height // 2

        # Get the depth value at the center pixel
        # The depth image from the D435i gives depth in MILLIMETERS.
        depth_value_mm = depth_image[center_y, center_x]
        
        # Convert millimeters to meters for easier interpretation
        depth_in_meters = depth_value_mm / 1000.0

        # --- Display the information on the image ---
        
        # Define the text to display
        if depth_in_meters > 0:
            # If the depth is valid (not 0), show the distance
            text = f"Distance: {depth_in_meters:.2f} meters"
        else:
            # If depth is 0, it means the camera couldn't get a reading at that point
            text = "Distance: N/A"

        # Draw a circle at the center point so we know where we are measuring
        cv2.circle(color_image, (center_x, center_y), 5, (0, 0, 255), -1) # Red circle

        # Put the text on the image
        cv2.putText(color_image, text, (center_x + 10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 255, 255), 2, cv2.LINE_AA) # White text

        # Show the final image in a window
        cv2.imshow("Depth Finder", color_image)
        
        # Wait a little bit. If the user presses 'q', exit.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info('Shutting down...')
            self.destroy_node()
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    depth_finder_node = DepthFinder()
    rclpy.spin(depth_finder_node)
    
    # Destroy the node explicitly
    depth_finder_node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()