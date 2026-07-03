# HL-TSA Python Port

Python implementation of `tools/HL-Detection-using-TSA/HL_Detect_TSA.m`.
The C++ CPU ROS 2 package lives in `src/hl_tsa_cpp/`.

The main entry point is `hl_detect_tsa.py`. It returns an `N x 2` state matrix
where column 1 is the vertical horizon position at the image center in pixels,
and column 2 is the horizon angle in degrees.

## Usage

Place the Buoy videos and HorizonGT MAT files under `logs/Buoy/` before running
the Python port. The MAT files are used as horizon-line ground truth when passed
with `--ground-truth`.

```bash
python tools/HL-TSA/hl_detect_tsa.py \
  logs/Buoy/buoyGT_2_5_3_4.avi \
  --listener-frames 16 \
  --canny-low 20 \
  --canny-high 60 \
  --hough-threshold 15 \
  --ground-truth logs/Buoy/buoyGT_2_5_3_4HorizonGT.mat \
  --csv logs/Buoy/buoyGT_2_5_3_4_states.csv \
  --timing-csv logs/Buoy/buoyGT_2_5_3_4_timing.csv
```

Add `--output-video path/to/output.avi` to save frames annotated with the
detected horizon in red and the ROI in yellow. Add `--max-frames N` for quick
smoke tests. Each run prints setup, listener-block, model-fit, main-loop, and
total execution time. Use `--frame-timing-csv` to save per-frame HL detection
runtime rows.

## Notes

The MATLAB version depends on the Econometrics Toolbox for ARIMA/GARCH models
and on the Image Processing Toolbox for Radon-based line detection. This port
uses a small local AR(2) forecaster and OpenCV Hough lines so it can run with
the Python packages already available in this workspace.

## C++ CPU Package

Build the C++ package with:

```bash
colcon build \
  --base-paths src \
  --packages-select hl_tsa_cpp \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

Run the offline video app with the tuned Buoy parameters:

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

To test the ROS node with an AVI file, launch HL-TSA in one terminal:

```bash
source install/setup.zsh

ros2 launch hl_tsa_cpp hl_tsa_cpp.launch.py \
  image_topic:=/camera/image_rect_color
```

Then publish the video frames in a second terminal:

```bash
source install/setup.zsh

ros2 launch hl_tsa_cpp video_image_publisher.launch.py \
  video_path:=logs/Buoy/buoyGT_2_5_3_4.avi \
  topic_name:=/camera/image_rect_color
```
