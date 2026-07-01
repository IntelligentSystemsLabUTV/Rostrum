# HL-TSA C++ CPU Package

ROS 2 / OpenCV C++ implementation of the HL-TSA horizon-line detector. The
package mirrors the structure of `object_detector_cpp_dst`:

- `include/hl_tsa_cpp/`: public node and algorithm declarations.
- `src/hl_tsa/`: detector, ROS node, CSV/video utilities, and generated params input.
- `config/`: runtime parameter YAML.
- `launch/`: ROS 2 launch file.
- `src/hl_tsa_app.cpp`: ROS 2 standalone node application.
- `src/hl_tsa_video_app.cpp`: offline video benchmarking application.

The implementation is CPU-only. It uses OpenCV Canny, connected components,
Hough lines, and histogram operations.

## Build

```bash
colcon build \
  --base-paths src \
  --packages-select hl_tsa_cpp \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

## Offline Video Run

```bash
source install/setup.zsh

ros2 run hl_tsa_cpp hl_tsa_video_app \
  --video logs/Buoy/buoyGT_2_5_3_4.avi \
  --listener-frames 16 \
  --canny-low 20 \
  --canny-high 60 \
  --hough-threshold 15 \
  --states-csv logs/Buoy/hl_tsa_cpp_buoyGT_2_5_3_4_states.csv \
  --frame-timing-csv logs/Buoy/hl_tsa_cpp_buoyGT_2_5_3_4_frame_timing.csv
```

## ROS 2 Node

```bash
source install/setup.zsh

ros2 launch hl_tsa_cpp hl_tsa_cpp.launch.py \
  image_topic:=/camera/image_rect_color
```

In a second terminal, publish an AVI file to that image topic:

```bash
source install/setup.zsh

ros2 launch hl_tsa_cpp video_image_publisher.launch.py \
  video_path:=logs/Buoy/buoyGT_2_5_3_4.avi \
  topic_name:=/camera/image_rect_color
```

The publisher uses the video's native FPS by default. Override it with
`fps:=25.0`, and use `loop:=true` to replay the video continuously. By default
the video publisher waits up to 10 seconds for an image subscriber before it
starts replaying; disable that with `wait_for_subscribers:=false` if you only
want to publish the raw image stream.

The node publishes `~/state` as a `std_msgs/msg/Float64MultiArray` with:

```text
[y, theta_deg, elapsed_ms, iterations, absence_flag]
```

Set `publish_annotated: true` in `config/hl_tsa_cpp.yaml` to publish the
annotated image on `~/annotated`.

## Troubleshooting: apparent message loss

Do not judge frame loss with `ros2 topic echo` (or `ros2 topic hz`) on
`/camera/image_rect_color`. Both tools must first introspect the message
type from the ROS graph before they can subscribe, which races against a
short test clip, and once attached they still have to deserialize and
pretty-print every raw image (hundreds of KB to MB as a byte array) to the
terminal, which cannot keep up with a 25-30 Hz stream; the resulting
backlog can make `echo` itself skip reliable samples with no fault in the
node. This is a limitation of those CLI tools, not of `hl_tsa_cpp`.

To check whether the node itself is actually dropping frames:

- Set `verbose: true` and watch for consecutive `frame=` numbers in the
  log, or set `output.frame_timing_csv` and count rows against the
  video's frame count.
- Watch for the node's own `RCLCPP_WARN`: *"HL-TSA image processing queue
  full; dropped N old image(s)"*. This is the only condition under which
  the node itself discards frames (its internal queue, bounded by
  `subscribers.depth`, is full because processing can't keep up); if it
  never appears, no frames were dropped.
