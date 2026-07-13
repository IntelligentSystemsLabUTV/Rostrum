# TRiD-Horizon GPU Package

ROS 2 / TensorRT C++ implementation of the TRiD-Horizon temporal horizon-line
detector from `tools/TRiD` (method `trid` in that tool's
`run_inference.py`/`run_benchmark.py`). The package mirrors the structure of
`essld_cpp` and `hl_tsa_cpp`:

- `include/trid_cpp/`: public node, TensorRT engine wrapper, and algorithm
  declarations.
- `src/trid/`: TensorRT engine wrapper, ported DSAC/heatmap and ROI
  refinement/gate code, detector, ROS node, and generated params input.
- `config/`: runtime parameter YAML.
- `launch/`: ROS 2 launch file.
- `src/trid_app.cpp`: ROS 2 standalone node application.

Like `essld_cpp` (and unlike `hl_tsa_cpp`), this package runs its neural
network directly through the TensorRT 10.x C++ runtime API
(`nvinfer1::IRuntime`/`ICudaEngine`/`IExecutionContext`, `enqueueV3`),
loading one of the prebuilt `.engine` files produced by
`tools/TRiD/export_onnx.py` + `trtexec`. TRiD-Horizon is a *temporal*
model (an 8-frame ConvGRU), unlike ESSLD's single-frame DCEUNet, which
drives most of what is new in this package relative to `essld_cpp` (see
below).

## What the TensorRT engine does and does not compute

`TRiDHorizon` (`tools/TRiD/models/heads.py`) is: DCEUNet backbone (shared
with ESSLD) -> per-frame features -> `ConvGRU` temporal aggregation over an
8-frame clip -> a 1x1 `reduce` conv -> `HorizonColumnHead` (per-column heat
map + confidence logits) -> `DSACLineFit`, a differentiable-RANSAC line fit
over the column heat/confidence maps.

**`DSACLineFit` is not part of the exported engine.** It calls
`torch.multinomial` to sample candidate column pairs, and TensorRT 10.9's
ONNX parser rejects that op outright:

```text
ERROR: onnxOpCheckers.cpp:979 In function checkMultinomial:
[8] false
```

(verified locally: `trtexec --onnx=weights/trid_horizon.onnx
--saveEngine=...` fails parsing with exactly this error before
`export_onnx.py`'s wrapper was introduced). `tools/TRiD/export_onnx.py`
therefore exports a wrapper (`TRiDExportWrapper`) that stops right before
`DSACLineFit`, at the two per-column tensors: `heat_logits`
`(1,1,256,512)` and `confidence_logits` `(1,1,512)`. `heatmap_to_points`
and `DSACLineFit` (`models/heads.py`) are ported to C++ instead
(`src/trid/trid_dsac.cpp`) and run on the engine's raw outputs every frame.

A second optimization is baked into the same wrapper: `TRiDHorizon.forward`
runs `reduce`+`column_head` on all 8 clip steps and then the inference
pipeline (`inference/pipeline.py`) only ever reads the *last* step's
output. The wrapper instead slices to the last ConvGRU step *before*
`reduce`/`column_head`. This is exact, not an approximation:
`reduce`/`column_head`'s `Conv2d`/`BatchNorm2d` layers run in eval mode
(fixed running statistics) with no cross-sample interaction, so per-frame
outputs are bit-identical either way -- verified locally on a random input
(max abs diff `0.0` on both outputs before this optimization was applied).
It cuts that stage's cost 8x.

### RNG note on `DSACLineFit`

`DSACLineFit` samples `hypotheses=64` column pairs per frame from a
confidence-weighted categorical distribution (`torch.multinomial(...,
replacement=True)`), fits a line per hypothesis, and soft-votes across
hypotheses to pick the final line -- this is inherent to the model, not an
artifact of this port: `tools/TRiD/run_inference.py` itself never fixes a
random seed, so the Python reference is already non-deterministic
frame-to-frame. The C++ port (`dsac_line_fit` in `trid_dsac.cpp`) uses
`std::discrete_distribution` seeded from `std::random_device`, which has
the same sampling semantics as `torch.multinomial` but an independent RNG
stream -- results are not bit-exact with a given PyTorch run, but this
does not matter here: the whole point of this differentiable-RANSAC-style
head is that no single sampled hypothesis' identity matters, only the
consensus-weighted average over many of them. The closed-form parts of the
pipeline (`heatmap_to_points`'s softmax expectation, and every formula
inside `DSACLineFit` given a fixed set of sampled indices) were verified
locally against the PyTorch reference to float32 precision (see the git
history of this port for the validation script) before being translated to
C++.

## What is ported vs. fixed vs. deliberately different

`src/trid/trid_algorithm.cpp` ports `inference/postprocess.py`'s
`create_roi`, `dual_fusion_pipeline` (`run_dual_fusion_pipeline` here),
`refine_existing`, and their helpers (`robust_fit_line`,
`weighted_line_fit`, `non_max_suppression`, `calculate_dynamic_padding`)
exactly, using the same C++ replacements as `essld_cpp` for constructs with
no direct OpenCV equivalent (`np.polyfit(..., w=weights)` ->
`weighted_line_fit`; numba's `_non_max_suppression_fast` -> a plain nested
loop). The "dual multi-scale fusion" ROI refinement stage itself is
verbatim-equivalent Python between `tools/ESSLD/utils/processing.py` and
`tools/TRiD/inference/postprocess.py` -- this is not a coincidence, both
tools share the same ROI-refinement design, just tuned with different
default thresholds (see below).

`src/trid/trid_algorithm.cpp` also ports `inference/roi_gate.py`'s
`apply_bounded_roi` (the conservative accept/reject gate around the ROI
refinement) exactly, including all six rejection reasons
(`refined_outside_roi`, `center_shift_too_large`, `endpoint_shift_too_large`,
`angle_change_too_large`, `insufficient_candidates`,
`candidate_span_too_small`). **This has no equivalent in `essld_cpp`**,
which always accepts its ROI refinement whenever it does not throw;
`essld_cpp`'s segmentation-mask-based coarse line has no such gate because
ESSLD is not part of the all-methods comparison harness this gate was
designed for.

One subtlety worth flagging explicitly: `roi_gate.py`'s `candidate_span`
(logged as `candidate_horizontal_span` in `tools/TRiD/run_inference.py`'s
CSV) is computed from `points[:, 1]`, where `points` is
`dual_fusion_pipeline`'s raw `pts_yx = np.argwhere(binary > 0)` -- i.e.
`points[:, 1]` is the **column** index, not the row. It is a horizontal
span, not a vertical one, despite living next to `roi_height`-relative
thresholds. `run_dual_fusion_pipeline` in this package tracks the
candidate columns (`xs`) directly during its Hough-filtered mask scan and
reports `max(xs) - min(xs)` as `candidate_span`, matching this.

**A real bug found and fixed by validating against the Python reference
end-to-end**: `dual_fusion_pipeline`'s candidate-cleanup connected-components
call, `cv2.connectedComponentsWithStats(binary, 4, cv2.CV_32S)`, *looks*
like it requests 4-connectivity. Verified locally (OpenCV 4.11.0, both the
`opencv-python` and C++ `libopencv` bindings, on a synthetic 3-diagonal-pixel
test case where true 4- vs 8-connectivity give different, checkable answers):
that 3-positional-argument Python call does not actually bind its second
argument to `connectivity` at all -- it silently runs with the function's
8-connectivity default regardless of the value passed. The equivalent C++
call with an explicit `4` does apply true 4-connectivity. So the reference
tool's *actual* runtime behavior here is 8-connectivity, contrary to what
its source appears to request. `run_dual_fusion_pipeline` in this package
hardcodes `8` to match that actual behavior, not the misleading literal `4`
in `postprocess.py`. This was not a hypothetical concern: an initial faithful
port using true 4-connectivity was validated end-to-end against
`tools/TRiD/run_inference.py --method trid` on `samples/TMD/TMD_annotated_5.avi`
and measured a mean center error of 9.66 px / 29.9% ROI acceptance vs. the
Python reference's 5.75 px / 87.5% on the same clip -- i.e. a real accuracy
regression, not a cosmetic difference. After hardcoding `8`-connectivity,
the same comparison gives 5.62 px / 87.2% for this package against 5.67 px
/ 87.3% for the Python reference -- statistically equivalent, with the small
remaining spread attributable to `DSACLineFit`'s independent RNG stream (see
above) and TensorRT-vs-PyTorch numerics. **`tools/ESSLD/utils/processing.py`
has the identical `cv2.connectedComponentsWithStats(binary, 4, cv2.CV_32S)`
call in its own dual-fusion refinement stage, and `essld_cpp`'s port of it
(`run_dual_fusion_pipeline` in `essld_cpp`'s `essld_processing.cpp`) was
given the same literal (incorrect) `4`; it was not fixed as part of this
change since it's a different package, but it is very likely subject to the
same discrepancy and worth checking.**

`config.py`'s tunables actually exercised by `dual_fusion_pipeline` are
exposed as ROS parameters under `fusion.*` (see `config/trid_cpp.yaml`);
`roi_gate.py`'s `ROIGateConfig` fields are exposed under `roi_gate.*`.
Note that `tools/TRiD/config.py`'s fusion defaults
(`HOUGH_THRESHOLD=30`, `HOUGH_MIN_LINE_LENGTH=50`, `HOUGH_MAX_LINE_GAP=20`)
differ from `essld_cpp.yaml`'s tuned values (`50`/`60`/`100`) -- this
package defaults to `tools/TRiD/config.py`'s values, since that is the
reference tool being ported here. `refine_existing`'s ROI padding gradient
score is hardcoded to `80` in the Python reference (not read from
`config.py` at all); it is exposed here as `fusion.grad_score` (default
`80.0`) for tunability, matching `essld_cpp`'s equivalent parameter.
Magic numbers the Python reference never varies (CLAHE settings, the
fusion-stage Canny thresholds, the candidate connected-component area
cleanup) are left as fixed constants in `trid_algorithm.cpp`, exactly as
in `essld_cpp`.

**Temporal clip buffer -- deliberate deviation from the Python
reference.** `inference/pipeline.py`'s `MethodRunner` keeps a plain Python
list, appends one preprocessed frame per call, and stacks whatever is in
it (`clip_buffer[-CLIP_LENGTH:]`) -- so for the first `CLIP_LENGTH-1`
frames of a stream it feeds the model a *shorter-than-8* sequence. The
exported TensorRT engine has a fixed input shape `(1, 8, 3, 256, 512)` and
cannot accept a shorter sequence (this is also why the ConvGRU loop had to
be traced, not scripted, with a static clip length -- see
`export_onnx.py`). `TRiDDetector::process_frame`
(`src/trid/trid_detector.cpp`) instead always feeds a full 8-frame window,
front-padding with repeats of the oldest buffered frame while the stream
is warming up. Once 8 frames have been processed, this is an *exact*
match to the Python reference's sliding window (same 8 frames, same
order); only the first 7 frames of a stream differ, and only by using a
replicated-frame warm-up instead of a variable-length one. This is judged
the more useful behavior for a real-time robotics deployment (the node
produces a plausible line from frame 1, rather than a degenerate one or
none at all), and is called out here because it is a genuine behavior
difference, not just a numerics footnote.

**No temporal smoothing needed.** Unlike `essld_cpp`
(`smoothing.alpha`, an exponential blend added by `demo_video.py` on top
of DCEUNet's frame-independent output), TRiD-Horizon's `ConvGRU` already
performs temporal smoothing internally via its hidden state across the
clip window; `TRiDDetector` applies no additional smoothing, matching
`inference/pipeline.py` (which applies none either).

## TensorRT engine portability (read this before deploying elsewhere)

`.engine` files are **not** portable binaries: they are locked to the exact
GPU architecture, driver, and TensorRT build used to create them. The
engines shipped in `tools/TRiD/weights/` were built on this workspace's
machine with **TensorRT 10.9.0.34**, driver **580.95.05**, on an **RTX 5090
Laptop GPU** -- identical toolchain to `essld_cpp`'s engines. Loading them
on different hardware/driver/TensorRT versions will fail (or silently
misbehave). If you're deploying to a different machine, re-export from the
checkpoint and rebuild the engine there:

```bash
cd tools/TRiD
python3 export_onnx.py --verify   # -> weights/trid_horizon.onnx
trtexec --onnx=weights/trid_horizon.onnx --saveEngine=weights/trid_horizon.engine
trtexec --onnx=weights/trid_horizon.onnx --saveEngine=weights/trid_horizon_fp16.engine --fp16
```

For this reason `model.engine_path` is a mandatory, operator-set parameter
(the node fails fast at startup if it's empty or the file doesn't exist) --
the `.engine` files are **not** installed into this package's share
directory as if they were portable artifacts.

Both `trid_horizon.engine` (FP32) and `trid_horizon_fp16.engine` (FP16)
expose the same FLOAT32 I/O tensor bindings (`input`, `heat_logits`,
`confidence_logits`) -- only their internal precision differs -- so either
can be pointed to by `model.engine_path` with no other config changes.
`trid_horizon_fp16.engine` is the recommended default for real-time use
(matches `tools/TRiD/run_inference.py`'s own FP16-by-default-on-CUDA
behavior), and measured ~30% faster GPU compute time than FP32 on the
reference machine (`trtexec` benchmark: FP32 mean GPU compute 5.08 ms vs.
FP16 mean 3.68 ms, at batch 1, clip length 8).

## Build

```bash
colcon build \
  --base-paths src \
  --packages-select trid_cpp \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

## ROS 2 Node

Set `model.engine_path` in `config/trid_cpp.yaml` to an absolute path to a
valid engine for your machine, e.g.
`<workspace>/tools/TRiD/weights/trid_horizon_fp16.engine`, then:

```bash
source install/setup.zsh

ros2 launch trid_cpp trid_cpp.launch.py \
  image_topic:=/camera/image_rect_color
```

To feed it a test clip over ROS, use the generic `video_image_converter_py`
video-file-to-`Image`-topic converter:

```bash
source install/setup.zsh

ros2 launch video_image_converter_py video_image_converter.launch.py \
  video_path:=tools/TRiD/samples/TMD/TMD_annotated_5.avi \
  image_topic:=/camera/image_rect_color
```

The node publishes `~/state` as a `std_msgs/msg/Float64MultiArray` with:

```text
[y, theta_deg, elapsed_ms, valid_flag, roi_accepted_flag]
```

`y`/`theta_deg` describe the detected horizon line at the frame's
horizontal midline. `valid_flag` is always `1.0` (unlike `essld_cpp`'s mask
-based coarse line, which can fail outright, the column-heatmap approach
here always yields a fit). `roi_accepted_flag` is `1.0` when the dual
multi-scale ROI refinement passed `roi_gate`'s acceptance checks and
`y`/`theta_deg` reflect the refined line, `0.0` when the gate rejected it
(or ROI refinement/gating is disabled) and `y`/`theta_deg` reflect the
model's coarse line instead -- see `~/state`'s CSV logging below for the
specific rejection reason on any given frame.

Set `publish_annotated: true` (default) in `config/trid_cpp.yaml` to
publish the frame with the detected line drawn on `~/annotated`, and
`publish_heatmap: true` to also publish a debug visualization of the raw
per-pixel heat logits (sigmoid-scaled, mono8) on `~/heatmap` -- note this
is **not** what the detector uses to extract the line (it applies a
per-column softmax expectation over the raw logits, not a threshold), it
is a debug aid only, analogous to but not equivalent to `essld_cpp`'s
`publish_mask`.

Set `output.states_csv` and/or `output.frame_timing_csv` (both empty/disabled
by default) to absolute file paths to have the node append, per processed
frame: a `frame,y,theta_deg` row to `states_csv`, and a
`frame,elapsed_seconds,inference_seconds,y,theta_deg,roi_accepted,roi_reason`
row to `frame_timing_csv` -- same CSV-logging mechanism as `essld_cpp`'s
equivalent parameters. `elapsed_seconds` covers everything
`TRiDDetector::process_frame` times: preprocessing, TensorRT inference,
heatmap decoding + DSAC line fit, and the dual multi-scale ROI
refinement/gate. `inference_seconds` is the subset spent purely inside
`TRTEngine::infer` (H2D copy + `enqueueV3` + D2H copy + stream sync).
Both timings are also logged with `verbose: true` (`RCLCPP_INFO` per
frame). Measured on the reference machine with the FP16 engine: ~4.5 ms
inference, ~13-15 ms full pipeline per frame (dominated by the dual
multi-scale ROI refinement's OpenCV work, not TensorRT).

## Troubleshooting: apparent message loss

Do not judge frame loss with `ros2 topic echo`/`ros2 topic hz` on the raw
image topic -- see `hl_tsa_cpp`'s README for why. To check whether this
node itself is dropping frames, set `verbose: true` and watch for its own
`RCLCPP_WARN`: *"TRiD image processing queue full; dropped N old
image(s)"*. This is the only condition under which the node discards
frames (its internal queue, bounded by `subscribers.depth`, is full because
GPU inference + ROI refinement can't keep up). Alternatively, set
`output.frame_timing_csv` and count rows against the input stream's frame
count.
