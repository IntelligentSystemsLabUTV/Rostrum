"""
Video image converter node standalone application.

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

import traceback

import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor

from video_image_converter_py.video_image_converter import VideoImageConverter


def main(args=None):
    """Runs the video image converter app."""
    rclpy.init(args=args)
    node = None
    executor = MultiThreadedExecutor()
    try:
        node = VideoImageConverter()
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as e:
        print(f"Exception occurred: {e}")
        traceback.print_exc()
    finally:
        executor.shutdown()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
