"""
ESSLD GPU app launch file.
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

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    config = os.path.join(
        get_package_share_directory('essld_cpp'),
        'config',
        'essld_cpp.yaml'
    )

    ns = LaunchConfiguration('namespace')
    cf = LaunchConfiguration('cf')
    image_topic = LaunchConfiguration('image_topic')

    ld.add_action(DeclareLaunchArgument('namespace', default_value=''))
    ld.add_action(DeclareLaunchArgument('cf', default_value=config))
    ld.add_action(DeclareLaunchArgument('image_topic', default_value='/camera/image_rect_color'))

    node = Node(
        package='essld_cpp',
        executable='essld_cpp_app',
        namespace=ns,
        emulate_tty=True,
        output='both',
        log_cmd=True,
        parameters=[cf],
        remappings=[
            ('/camera/image_rect_color', image_topic),
        ],
    )
    ld.add_action(node)

    return ld
