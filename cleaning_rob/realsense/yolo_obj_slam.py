"""
YOLOv8 ROS2 detection node integrating with RealSense camera topics launched by `launch_slam_rtmap.py`.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

# Optional OpenCV visualisation support
try:
    import cv2
except ImportError:  # pragma: no cover - OpenCV may be unavailable in headless envs
    cv2 = None  # type: ignore

# --- ultralytics is the correct library ---
from ultralytics import YOLO

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
except ImportError as exc:  # pragma: no cover - ROS2 not always installed in dev envs
    rclpy = None  # type: ignore
    ROS_IMPORT_ERROR = exc
else:
    ROS_IMPORT_ERROR = None

try:
    from cv_bridge import CvBridge
except ImportError:  # pragma: no cover
    CvBridge = None  # type: ignore


@dataclass
class DetectorConfig:
    """Configuration tying YOLO inference to the RealSense SLAM launch."""

    # --- Changed default model to a real one: yolov8n.pt ---
    model_path: str = os.getenv("YOLO_MODEL", "yolov8n.pt")
    image_topic: str = os.getenv("YOLO_IMAGE_TOPIC", "/camera/camera/color/image_raw")
    detection_topic: str = os.getenv("YOLO_DETECTIONS_TOPIC", "/yolo/detections")
    confidence_threshold: float = float(os.getenv("YOLO_CONF", "0.25"))
    max_results: int = int(os.getenv("YOLO_MAX_DETECTIONS", "100"))
    publish_raw_json: bool = bool(int(os.getenv("YOLO_PUBLISH_JSON", "1")))
    display_window: bool = bool(int(os.getenv("YOLO_DISPLAY", "0")))


class YOLODetector:
    """Thin wrapper around the YOLO model to normalise predictions."""

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        # --- Changed yolo11.YOLO to just YOLO ---
        self.model = YOLO(config.model_path)

    def infer(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Runs inference and returns serialisable detections."""

        # The ultralytics predict method accepts these arguments directly
        raw_results = self.model.predict(
            frame,
            conf=self.config.confidence_threshold,
            max_det=self.config.max_results,
        )

        return self._serialise_results(raw_results)

    def _serialise_results(self, raw_results: Any) -> List[Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []
        if raw_results is None:
            return detections

        iterator: Iterable = raw_results if isinstance(raw_results, Iterable) else [raw_results]

        for result in iterator:
            # ultralytics results object has a convenient .tojson() method
            if hasattr(result, "tojson"):
                try:
                    json_str = result.tojson()
                    if json_str:
                        parsed = json.loads(json_str)
                        if isinstance(parsed, list):
                            detections.extend(parsed)
                            continue
                except Exception:
                    pass # Fallback to manual parsing if tojson fails

            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            # Extract data from the boxes object
            xyxy_tensors = getattr(boxes, "xyxy", [])
            conf_tensors = getattr(boxes, "conf", [])
            cls_tensors = getattr(boxes, "cls", [])
            id_tensors = getattr(boxes, "id", None) # Tracker IDs

            for i in range(len(xyxy_tensors)):
                detections.append(
                    {
                        "name": self.model.names[int(cls_tensors[i])],
                        "class_id": int(cls_tensors[i]),
                        "confidence": float(conf_tensors[i]),
                        "box": {
                            "x1": float(xyxy_tensors[i][0]),
                            "y1": float(xyxy_tensors[i][1]),
                            "x2": float(xyxy_tensors[i][2]),
                            "y2": float(xyxy_tensors[i][3]),
                        },
                        "track_id": int(id_tensors[i]) if id_tensors is not None else None,
                    }
                )

        return detections


class YoloObjSlamNode(Node):
    """ROS2 node that bridges RealSense imagery to YOLO detections."""

    def __init__(self, config: DetectorConfig) -> None:
        super().__init__("yolo_obj_slam")
        self.config = config
        self.detector = YOLODetector(config)
        self._display_enabled = bool(config.display_window and cv2 is not None)
        if config.display_window and cv2 is None:
            self.get_logger().warning(
                "YOLO_DISPLAY was requested but OpenCV (cv2) is not available. "
                "Install opencv-python-headless to enable visualisation."
            )

        if CvBridge is None:
            raise ImportError(
                "cv_bridge is required to convert ROS Image messages to OpenCV arrays. "
                "Install 'ros-humble-cv-bridge' or matching distro."
            )
        self._bridge = CvBridge()

        self._publisher = self.create_publisher(String, config.detection_topic, 10)
        self._subscription = self.create_subscription(
            Image,
            config.image_topic,
            self._image_callback,
            10,
        )

        self.get_logger().info(
            f"YOLO detector ready. Listening to '{config.image_topic}', "
            f"publishing detections to '{config.detection_topic}'."
        )

    def _image_callback(self, ros_image: Image) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(ros_image, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"Failed to convert image message: {exc}")
            return

        detections = self.detector.infer(frame)

        if not detections:
            return

        payload = json.dumps(
            {
                "header_stamp": ros_image.header.stamp.sec + ros_image.header.stamp.nanosec * 1e-9,
                "frame_id": ros_image.header.frame_id,
                "detections": detections,
            }
        ) if self.config.publish_raw_json else json.dumps(detections)

        self._publisher.publish(String(data=payload))

        if self._display_enabled:
            self._show_detections(frame, detections)

        self.get_logger().debug(
            f"Published {len(detections)} detections for frame {ros_image.header.frame_id}"
        )

    def _show_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> None:
        if cv2 is None:
            return

        annotated = frame.copy()
        for det in detections:
            box = det.get("box")
            if box is not None:
                x1, y1, x2, y2 = box.values()
            elif "bbox_xyxy" in det:
                x1, y1, x2, y2 = det["bbox_xyxy"]
            else:
                continue

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            label = det.get("name") or str(det.get("class_id", "?"))
            score = det.get("confidence")
            caption = f"{label}" if score is None else f"{label}: {score:.2f}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                caption,
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        cv2.imshow("YOLO detections", annotated)
        cv2.waitKey(1)


def main() -> None:
    if rclpy is None:
        raise ImportError(
            "rclpy is not available. Install ROS2 Python client libraries to run the YOLO node."
        ) from ROS_IMPORT_ERROR

    rclpy.init()
    config = DetectorConfig()
    node = YoloObjSlamNode(config)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down YOLO detector...")
    finally:
        node.destroy_node()
        if cv2 is not None and config.display_window:
            cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()