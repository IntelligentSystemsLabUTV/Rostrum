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
Hough lines, and histogram operations for the horizon-line detector (`HLDA`,
substituting Hough for MATLAB's Radon transform, which has no OpenCV
equivalent), and an exact ARIMA(2,d,0)+GARCH(1,1) time series model for the
dynamic ROI, matching `HL_Detect_TSA.m`'s `Mdly`/`Mdlt` (`arima('ARLags',1:2,
'D',d,'Variance',garch(1,1))`).

## Time series model (`ArimaGarchModel`)

For each of `y` (`D=2`) and `theta` (`D=0`), the model is fit once after the
listener block by jointly maximizing the exact Gaussian conditional
log-likelihood of the ARMA(2) mean equation plus GARCH(1,1) conditional
variance equation, via a dependency-free Nelder-Mead simplex (no
gradient/toolbox optimizer is available in C++, unlike MATLAB's `fmincon`).
One-step-ahead forecasts then reuse those fixed coefficients every frame,
exactly as `HL_Detect_TSA.m` does (`estimate` is called once; `forecast` is
called every frame). The forecast horizon is always 1, so the returned
forecast variance equals the GARCH one-step conditional variance directly
(the MA(1) coefficient on the next innovation is always 1, independent of
`D`).

Two points are exact matches to MATLAB's public documentation
(mathworks.com/help/econ/): the `Constant` term is fixed at 0 when `D>0`
(not estimated) and estimated when `D=0`; the presample window length is
`N - P` frames where `P = p + D` is the compound AR polynomial degree, so
`P=4` for the `y` model and `P=2` for `theta` (not simply the AR lag order).
This `Constant` behavior is what this implementation assumes for `theta`
(`D=0`), but note the paper (Agaoglu & Topaloglu, 2025, Eq. 3) writes the
theta mean equation with no intercept term at all
(`(1-gamma1*B-gamma2*B^2)*theta_k = eps_theta,k`) -- it's ambiguous whether
that is a notational simplification or the actual fitted model has no
constant; this implementation follows the documented MATLAB software
default rather than the paper's possibly-simplified displayed equation.

Two points are principled but **not verifiable against MATLAB's source**
(the Econometrics Toolbox does not publish exact numerics for these):

- **Presample backcast**: MATLAB documents only that presample
  variances/innovations default to "the sample mean of squared response
  series," with no disclosed formula. This is implemented as a *fixed*
  value (mean of squared conditional-least-squares residuals from the
  initial AR fit), held constant across optimization iterations and reused
  for every forecast call of that fitted model.
- **Optimizer**: Nelder-Mead simplex in place of whatever solver MATLAB's
  `estimate` uses internally (undisclosed by MathWorks' public docs, and
  not named in the paper either -- the paper only says the listener-block
  states "are... used to optimize the coefficients of the TSMs", with no
  solver specified). The objective (log-likelihood) and constraints
  (stationary AR(2) triangle via a Durbin-Levinson reparametrization;
  `omega>0`, `alpha, beta>=0`, `alpha+beta<0.999`) are exact; the numerical
  path to the optimum is not the same algorithm.

`min_y_sd`/`min_theta_sd` are pure numerical safety floors (default 0.05 px,
0.01 deg) guarding against a degenerate zero-width ROI on pathological
input; they are not part of the paper's model.

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

ros2 launch video_image_converter_py video_image_converter.launch.py \
  video_path:=logs/Buoy/buoyGT_2_5_3_4.avi \
  image_topic:=/camera/image_rect_color
```

The converter uses the video's native FPS by default. Override it with
`fps:=25.0`, and use `loop:=true` to replay the video continuously. By default
the video converter waits up to 10 seconds for an image subscriber before it
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
