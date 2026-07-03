"""
Video image converter app launch file.

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

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    """Generates the video image converter launch description."""
    video_path = LaunchConfiguration('video_path')
    image_topic = LaunchConfiguration('image_topic')
    frame_id = LaunchConfiguration('frame_id')
    fps = LaunchConfiguration('fps')
    loop = LaunchConfiguration('loop')
    start_frame = LaunchConfiguration('start_frame')
    max_frames = LaunchConfiguration('max_frames')
    qos_depth = LaunchConfiguration('qos_depth')
    qos_reliability = LaunchConfiguration('qos_reliability')
    wait_for_subscribers = LaunchConfiguration('wait_for_subscribers')
    wait_timeout_sec = LaunchConfiguration('wait_timeout_sec')
    use_sim_time = LaunchConfiguration('use_sim_time')
    verbose = LaunchConfiguration('verbose')
    name = LaunchConfiguration('name')
    namespace = LaunchConfiguration('namespace')

    return LaunchDescription([
        DeclareLaunchArgument('video_path', default_value=''),
        DeclareLaunchArgument('image_topic', default_value='/image'),
        DeclareLaunchArgument('frame_id', default_value='camera'),
        DeclareLaunchArgument('fps', default_value='0.0'),
        DeclareLaunchArgument('loop', default_value='false'),
        DeclareLaunchArgument('start_frame', default_value='0'),
        DeclareLaunchArgument('max_frames', default_value='0'),
        DeclareLaunchArgument('qos_depth', default_value='1'),
        DeclareLaunchArgument('qos_reliability', default_value='reliable'),
        DeclareLaunchArgument('wait_for_subscribers', default_value='true'),
        DeclareLaunchArgument('wait_timeout_sec', default_value='10.0'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('verbose', default_value='false'),
        DeclareLaunchArgument('name', default_value='video_image_converter'),
        DeclareLaunchArgument('namespace', default_value=''),
        Node(
            package='video_image_converter_py',
            executable='video_image_converter',
            name=name,
            namespace=namespace,
            output='screen',
            emulate_tty=True,
            parameters=[{
                'video_path': ParameterValue(video_path, value_type=str),
                'frame_id': ParameterValue(frame_id, value_type=str),
                'fps': ParameterValue(fps, value_type=float),
                'loop': ParameterValue(loop, value_type=bool),
                'start_frame': ParameterValue(start_frame, value_type=int),
                'max_frames': ParameterValue(max_frames, value_type=int),
                'qos.depth': ParameterValue(qos_depth, value_type=int),
                'qos.reliability': ParameterValue(qos_reliability, value_type=str),
                'wait_for_subscribers': ParameterValue(wait_for_subscribers, value_type=bool),
                'wait_timeout_sec': ParameterValue(wait_timeout_sec, value_type=float),
                'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
                'verbose': ParameterValue(verbose, value_type=bool),
            }],
            remappings=[
                ('/image', image_topic),
            ],
        ),
    ])
