import rclpy
from rclpy.node import Node
from rtabmap_msgs.srv import PublishMap
from pynput import keyboard
import time

class MapSaver(Node):
    def __init__(self):
        super().__init__('map_saver_key_node')
        self.cli = self.create_client(PublishMap, '/rtabmap/rtabmap/publish_map')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('RTAB-Map "publish_map" service not available, waiting...')
        
        self.req = PublishMap.Request()
        self.req.global_map = True
        self.req.optimized = True
        self.req.graph_only = False
        
        self.get_logger().info("Map Saver node is running. Press 's' in this terminal's window or anywhere to save the map.")
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        try:
            if key.char == 's':
                self.get_logger().info("'s' key pressed. Calling service to publish the map...")
                self.future = self.cli.call_async(self.req)
                self.get_logger().info("Map publish request sent. The map will be saved to ~/.ros/rtabmap.db")

        except AttributeError:
            # This handles special keys like 'shift', 'ctrl', etc.
            pass

def main(args=None):
    rclpy.init(args=args)
    map_saver_node = MapSaver()
    rclpy.spin(map_saver_node)
    map_saver_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()