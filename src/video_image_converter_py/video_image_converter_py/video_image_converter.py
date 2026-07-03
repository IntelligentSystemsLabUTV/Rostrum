"""
Generic video file to ROS 2 Image converter node.

dotX Automation s.r.l. <info@dotxautomation.com>

July 2, 2026
"""

# Copyright 2026 dotX Automation s.r.l.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time

import cv2
import rclpy
from ament_index_python.packages import PackageNotFoundError
from ament_index_python.packages import get_package_share_directory
import dua_qos_py.dua_qos_besteffort as dua_qos_besteffort
from dua_node_py.dua_node import NodeBase
import dua_qos_py.dua_qos_reliable as dua_qos_reliable
from sensor_msgs.msg import Image


def _get_params_file_path() -> str:
    """
    Returns the node parameter-definition file path.

    :return: Parameter-definition file path.
    """
    try:
        return os.path.join(
            get_package_share_directory('video_image_converter_py'),
            'video_image_converter_params.yaml'
        )
    except PackageNotFoundError:
        return os.path.join(os.path.dirname(__file__), 'video_image_converter_params.yaml')


class VideoImageConverter(NodeBase):
    """Publishes video frames as sensor_msgs/msg/Image messages."""

    _PARAMS_FILE_PATH = _get_params_file_path()

    def __init__(self):
        """Constructor."""
        self._cap = None
        self._image_pub = None
        self._publish_timer = None
        self._publish_fps = 0.0
        self._frames_published = 0
        self._stopping = False

        super().__init__("video_image_converter", True)

        self._open_video()
        if self._wait_for_subscribers:
            self._wait_until_matched()
        self._publish_timer = self.dua_create_timer(
            "publish_frame",
            1000.0 / self._publish_fps,
            self._publish_next_frame,
            self._timer_cgroup
        )
        self.get_logger().info("Node initialized")

    def init_cgroups(self) -> None:
        """Initializes callback groups."""
        self._timer_cgroup = self.dua_create_exclusive_cgroup()

    def init_publishers(self) -> None:
        """Initializes publishers."""
        self._image_pub = self.dua_create_publisher(
            Image,
            "/image",
            self._make_qos_profile()
        )

    def validate_qos_reliability(self, parameter) -> bool:
        """
        Validates the QoS reliability selector.

        :param parameter: Parameter update.
        :return: Whether the value is valid.
        """
        return str(parameter.value).lower() in (
            "reliable",
            "reliability_reliable",
            "best_effort",
            "besteffort",
            "best-effort",
        )

    def _make_qos_profile(self):
        """
        Creates the configured image QoS profile.

        :return: Image QoS profile.
        """
        reliability = str(self._qos_reliability).lower()
        if self._qos_depth < 1:
            raise RuntimeError("Parameter 'qos.depth' must be greater than zero")
        if reliability in ("reliable", "reliability_reliable"):
            return dua_qos_reliable.get_image_qos(depth=self._qos_depth)
        if reliability in ("best_effort", "besteffort", "best-effort"):
            return dua_qos_besteffort.get_image_qos(depth=self._qos_depth)
        raise RuntimeError("Parameter 'qos.reliability' must be 'reliable' or 'best_effort'")

    def _open_video(self) -> None:
        """Opens the video source and derives the publishing rate."""
        if not self._video_path:
            raise RuntimeError("Parameter 'video_path' is required")

        self._cap = cv2.VideoCapture(self._video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video: {self._video_path}")

        if self._start_frame > 0:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._start_frame)

        video_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self._publish_fps = self._fps if self._fps > 0.0 else video_fps
        if self._publish_fps <= 0.0:
            self._publish_fps = 25.0

        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(
            f"Publishing '{self._video_path}' on '{self._image_pub.topic_name}' "
            f"at {self._publish_fps:.3f} FPS ({total_frames} frames, {width}x{height})"
        )

    def _wait_until_matched(self) -> None:
        """Waits for at least one subscriber before replaying the video."""
        deadline = time.monotonic() + max(0.0, self._wait_timeout_sec)
        self.get_logger().info(
            f"Waiting for a subscriber on '{self._image_pub.topic_name}' "
            f"for up to {self._wait_timeout_sec:.1f} s"
        )
        while rclpy.ok() and self._image_pub.get_subscription_count() == 0:
            if time.monotonic() >= deadline:
                self.get_logger().warn(
                    f"No subscribers matched on '{self._image_pub.topic_name}', "
                    "starting video replay anyway"
                )
                return
            time.sleep(0.1)
        if self._image_pub.get_subscription_count() > 0:
            self.get_logger().info(f"Matched subscriber on '{self._image_pub.topic_name}'")

    def _stop(self, reason: str) -> None:
        """
        Stops video replay and shuts down the ROS context.

        :param reason: Reason to log.
        """
        if self._stopping:
            return
        self._stopping = True
        self.get_logger().info(reason)
        if self._publish_timer is not None:
            self._publish_timer.cancel()
        if rclpy.ok():
            rclpy.shutdown()

    def _restart_video(self):
        """
        Restarts the video from the configured start frame.

        :return: Tuple with read status and frame.
        """
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._start_frame)
        return self._cap.read()

    def _publish_next_frame(self) -> None:
        """Publishes the next video frame."""
        if self._max_frames > 0 and self._frames_published >= self._max_frames:
            self._stop("Reached max_frames, stopping converter")
            return

        ok, frame = self._cap.read()
        if not ok:
            if self._loop:
                ok, frame = self._restart_video()
            if not ok:
                self._stop("Reached end of video, stopping converter")
                return

        if not frame.flags["C_CONTIGUOUS"]:
            frame = frame.copy()

        msg = self._frame_to_image(frame)
        self._image_pub.publish(msg)
        self._frames_published += 1

        if self._verbose:
            self.get_logger().info(
                f"Published frame {self._frames_published}",
                throttle_duration_sec=1.0
            )

    def _frame_to_image(self, frame) -> Image:
        """
        Converts an OpenCV frame into an Image message.

        :param frame: OpenCV image matrix.
        :return: ROS 2 Image message.
        """
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.height = frame.shape[0]
        msg.width = frame.shape[1]
        msg.is_bigendian = False

        if len(frame.shape) == 2:
            msg.encoding = "mono8"
            msg.step = msg.width
        elif frame.shape[2] == 3:
            msg.encoding = "bgr8"
            msg.step = msg.width * 3
        elif frame.shape[2] == 4:
            msg.encoding = "bgra8"
            msg.step = msg.width * 4
        else:
            raise RuntimeError(f"Unsupported frame shape: {frame.shape}")

        msg.data = frame.tobytes()
        return msg

    def destroy_node(self) -> None:
        """Releases video resources before destroying the node."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        super().destroy_node()
