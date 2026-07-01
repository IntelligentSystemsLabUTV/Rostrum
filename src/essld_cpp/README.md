# ESSLD GPU Package

ROS 2 / TensorRT C++ implementation of the ESSLD (Efficient Sea-Sky Line
Detection) pipeline from `tools/ESSLD`. The package mirrors the structure of
`hl_tsa_cpp`:

- `include/essld_cpp/`: public node, TensorRT engine wrapper, and algorithm
  declarations.
- `src/essld/`: TensorRT engine wrapper, ported processing/refinement code,
  detector, ROS node, and generated params input.
- `config/`: runtime parameter YAML.
- `launch/`: ROS 2 launch file.
- `src/essld_app.cpp`: ROS 2 standalone node application.

Unlike `hl_tsa_cpp` (CPU-only), this package runs the DCEUNet segmentation
network directly through the TensorRT 10.x C++ runtime API
(`nvinfer1::IRuntime`/`ICudaEngine`/`IExecutionContext`, `enqueueV3`), loading
one of the prebuilt `.engine` files already produced by
`tools/ESSLD/export_onnx.py` + `trtexec`. This is a raw-TensorRT integration,
distinct from and much lighter than `object_detector_cpp_dst`'s
DeepStream/NvDsInfer pipeline: no GStreamer, no custom NvDsInfer plugins are
needed here, because both shipped engines deserialize with a bare
`nvinfer1::IRuntime` (verified by loading them directly with the Python
`tensorrt` module).

## What is ported vs. fixed

`src/essld/essld_processing.cpp` ports `tools/ESSLD/utils/processing.py`'s
`get_coarse_line_from_mask` and `refine_horizon_stage3` (and their helpers)
exactly, replacing only what has no direct C++/OpenCV equivalent:

- `np.polyfit(x, y, 1[, w=weights])` → a closed-form weighted least-squares
  line fit (`weighted_line_fit`), used both weighted and unweighted.
- `_non_max_suppression_fast` (numba-jitted) → a plain nested-loop C++
  translation (`non_max_suppression`); the ROI processed here is small
  enough that no numba-equivalent is needed.

**Dead code intentionally not ported**: `utils/processing.py` also defines
`_ray_cast_core_numba`, `_ray_cast_to_contour`, `_is_gourd_shape`, and
`_find_optimal_p`. None of them are called by `get_coarse_line_from_mask`,
`refine_horizon_stage3`, `demo.py`, or `demo_video.py` (confirmed by
grepping the reference tool) -- they are unused leftovers in the Python
reference and have no effect on the detector's behavior, so they were not
reimplemented.

`config.py`'s tunables that are actually exercised by the reference demos
are exposed as ROS parameters under `fusion.*`/`mask_threshold` (see
`config/essld_cpp.yaml`). Internal constants that the Python reference never
varies are left hardcoded in `essld_processing.cpp`: CLAHE
(`clipLimit=4.0`, `tileGridSize=(8,8)`), the fusion-stage
`Canny(50, 150)` thresholds, the coarse-line inlier distance (`< 5.0` px)
and minimum inlier/point counts (`10`), and the fusion connected-component
cleanup area (`>= 50` px).

Temporal smoothing (`smoothing.alpha`, default `0.7`, matching
`demo_video.py`'s `0.7*current + 0.3*last` blend) and hold-last-on-failure
behavior are reproduced in `ESSLDDetector::process_frame`
(`src/essld/essld_algorithm.cpp`).

## TensorRT engine portability (read this before deploying elsewhere)

`.engine` files are **not** portable binaries: they are locked to the exact
GPU architecture, driver, and TensorRT build used to create them. The
engines shipped in `tools/ESSLD/weights/` were built on this workspace's
machine with **TensorRT 10.9.0.34**, driver **580.95.05**, on an **RTX 5090
Laptop GPU**. Loading them on different hardware/driver/TensorRT versions
will fail (or silently misbehave). If you're deploying to a different
machine, re-export from the ONNX model and rebuild the engine there:

```bash
cd tools/ESSLD
python3 export_onnx.py --verify   # -> weights/dceunetex.onnx
trtexec --onnx=weights/dceunetex.onnx --saveEngine=weights/dceunetex.engine
trtexec --onnx=weights/dceunetex.onnx --saveEngine=weights/dceunetex_fp16.engine --fp16
```

For this reason `model.engine_path` is a mandatory, operator-set parameter
(the node fails fast at startup if it's empty or the file doesn't exist) --
the `.engine` files are **not** installed into this package's share
directory as if they were portable artifacts.

Both `dceunetex.engine` (FP32) and `dceunetex_fp16.engine` (FP16) expose the
same FLOAT32 I/O tensor bindings -- only their internal precision differs --
so either can be pointed to by `model.engine_path` with no other config
changes. `dceunetex_fp16.engine` is the recommended default for real-time
use (matches the Python tool's own real-time-oriented default).

## Build

```bash
colcon build \
  --base-paths src \
  --packages-select essld_cpp \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

## ROS 2 Node

Set `model.engine_path` in `config/essld_cpp.yaml` to an absolute path to a
valid engine for your machine, e.g.
`<workspace>/tools/ESSLD/weights/dceunetex_fp16.engine`, then:

```bash
source install/setup.zsh

ros2 launch essld_cpp essld_cpp.launch.py \
  image_topic:=/camera/image_rect_color
```

To feed it a test clip over ROS, reuse `hl_tsa_cpp`'s existing
`video_image_publisher` utility (a generic video-file-to-`Image`-topic
publisher, not specific to horizon detection) instead of duplicating it in
this package:

```bash
source install/setup.zsh

ros2 launch hl_tsa_cpp video_image_publisher.launch.py \
  video_path:=tools/ESSLD/samples/test_13.mp4 \
  topic_name:=/camera/image_rect_color
```

The node publishes `~/state` as a `std_msgs/msg/Float64MultiArray` with:

```text
[y, theta_deg, elapsed_ms, valid_flag]
```

`y`/`theta_deg` describe the detected sea-sky line at the frame's horizontal
midline (matching `demo.py`/`demo_video.py`'s line-drawing convention).
`valid_flag` is `1.0` when this frame produced a fresh detection, `0.0` when
the segmentation mask yielded no plausible line and the published
`y`/`theta_deg` are held over from the last valid frame (see
`smoothing.hold_last_on_failure`) -- mirroring `hl_tsa_cpp`'s
`absence_flag`, inverted.

Set `publish_annotated: true` (default) in `config/essld_cpp.yaml` to
publish the frame with the detected line drawn on `~/annotated`, and
`publish_mask: true` to also publish the thresholded segmentation mask on
`~/mask` (mono8) -- the C++ equivalent of `demo.py`'s mask-visualization
panel.

Set `output.states_csv` and/or `output.frame_timing_csv` (both empty/disabled
by default) to absolute file paths to have the node append, per processed
frame: a `frame,y,theta_deg` row to `states_csv`, and a
`frame,elapsed_seconds,inference_seconds,y,theta_deg,valid_flag` row to
`frame_timing_csv` -- same CSV-logging mechanism as `hl_tsa_cpp`'s
`output.states_csv`/`output.frame_timing_csv` parameters. `elapsed_seconds`
covers everything `ESSLDDetector::process_frame` times: preprocessing,
TensorRT inference, sigmoid/threshold, coarse-line extraction, and (when it
runs) the dual multi-scale fusion refinement -- it does not include ROS
message (de)serialization or the image queue wait. `inference_seconds` is
the subset of that spent purely inside `TRTEngine::infer` (H2D copy +
`enqueueV3` + D2H copy + stream sync) -- comparing it against
`elapsed_seconds` tells you how much time is TensorRT itself vs. the
OpenCV pre/post-processing and refinement around it. Both timings are also
logged with `verbose: true` (`RCLCPP_INFO` per frame). Unlike `hl_tsa_cpp`,
this package has no offline (non-ROS) video-benchmarking app, so these are
the only ways to log timing/detections to disk.

## Troubleshooting: apparent message loss

Do not judge frame loss with `ros2 topic echo`/`ros2 topic hz` on the raw
image topic -- see `hl_tsa_cpp`'s README for why. To check whether this node
itself is dropping frames, set `verbose: true` and watch for its own
`RCLCPP_WARN`: *"ESSLD image processing queue full; dropped N old
image(s)"*. This is the only condition under which the node discards
frames (its internal queue, bounded by `subscribers.depth`, is full because
GPU inference can't keep up). Alternatively, set `output.frame_timing_csv`
and count rows against the input stream's frame count, exactly as suggested
in `hl_tsa_cpp`'s README.
