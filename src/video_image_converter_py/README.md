# Video Image Converter Python Package

Generic ROS 2 Python node that publishes frames from a video source as
`sensor_msgs/msg/Image` messages.

The node is implemented with DUA's `dua_node_py.NodeBase`, publishes on
`/image` internally, and is intended to be remapped by launch files or the
ROS command line.

## Build

```bash
colcon build \
  --base-paths src \
  --packages-select video_image_converter_py
```

## Run

```bash
source install/setup.zsh

ros2 launch video_image_converter_py video_image_converter.launch.py \
  video_path:=logs/Buoy/buoyGT_2_5_3_4.avi \
  image_topic:=/camera/image_rect_color
```

By default the converter uses the video's native FPS, waits up to 10 seconds
for a subscriber, and stops at end of file. Use `fps:=25.0`, `loop:=true`, or
`wait_for_subscribers:=false` to override those behaviors.
