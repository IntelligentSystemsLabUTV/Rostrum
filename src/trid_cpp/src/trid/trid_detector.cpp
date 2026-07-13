/**
 * TRiD-Horizon detector: TensorRT backbone+temporal inference + C++ DSAC
 * line fit + dual multi-scale ROI refinement/gate, tying together
 * trid_trt.cpp, trid_dsac.cpp and trid_algorithm.cpp.
 *
 * dotX Automation s.r.l. <info@dotxautomation.com>
 */

/**
 * Copyright 2026 dotX Automation s.r.l.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <trid_cpp/trid_cpp.hpp>

#include <opencv2/imgproc.hpp>

#include <cstring>

namespace trid_cpp
{

TRiDDetector::TRiDDetector(std::shared_ptr<TRTEngine> engine, TRiDConfig config)
: engine_(std::move(engine)), config_(std::move(config)), rng_(std::random_device{}())
{
  const size_t frame_elems = 3ULL * config_.input_height * config_.input_width;
  input_clip_buffer_.resize(static_cast<size_t>(config_.clip_length) * frame_elems);
  heat_logits_buffer_.resize(1ULL * config_.input_height * config_.input_width);
  confidence_logits_buffer_.resize(1ULL * config_.input_width);
}

FrameResult TRiDDetector::process_frame(const cv::Mat & bgr_frame)
{
  const auto start = std::chrono::steady_clock::now();
  FrameResult result;

  const int orig_h = bgr_frame.rows;
  const int orig_w = bgr_frame.cols;
  const int model_h = config_.input_height;
  const int model_w = config_.input_width;
  const size_t frame_elems = 3ULL * model_h * model_w;

  // Preprocess: resize -> RGB -> CHW float32 in [0,1], matching
  // inference/pipeline.py's `preprocess`.
  cv::Mat resized;
  cv::resize(bgr_frame, resized, cv::Size(model_w, model_h));
  cv::Mat rgb;
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
  cv::Mat rgb_f;
  rgb.convertTo(rgb_f, CV_32FC3, 1.0 / 255.0);

  std::vector<float> frame_chw(frame_elems);
  std::vector<cv::Mat> channels(3);
  cv::split(rgb_f, channels);
  const size_t plane = static_cast<size_t>(model_h) * model_w;
  for (int c = 0; c < 3; ++c) {
    std::memcpy(frame_chw.data() + c * plane, channels[c].ptr<float>(0), plane * sizeof(float));
  }

  // Maintain a rolling window of the last clip_length preprocessed frames,
  // matching inference/pipeline.py's MethodRunner.clip_buffer (a Python
  // list appended every frame and trimmed to `clip_buffer[-CLIP_LENGTH:]`).
  //
  // DEVIATION FROM THE PYTHON REFERENCE: the reference feeds the model a
  // growing (possibly shorter than CLIP_LENGTH) sequence for the first
  // CLIP_LENGTH-1 frames of a stream, since torch.stack() over a Python
  // list has no fixed-length requirement. The exported TensorRT engine has
  // a fixed input shape (1, clip_length, 3, H, W) (see export_onnx.py) and
  // cannot accept a shorter sequence. This detector instead always feeds a
  // full clip_length window, front-padding with repeats of the oldest
  // buffered frame while the stream is still warming up. Once
  // clip_buffer_.size() == clip_length (after clip_length-1 frames), this
  // is an exact match to the Python reference's sliding window; only the
  // first clip_length-1 frames of a stream differ. See README.md.
  clip_buffer_.push_back(std::move(frame_chw));
  if (static_cast<int>(clip_buffer_.size()) > config_.clip_length) {
    clip_buffer_.pop_front();
  }

  const int deficit = config_.clip_length - static_cast<int>(clip_buffer_.size());
  for (int step = 0; step < config_.clip_length; ++step) {
    const int src_idx = step < deficit ? 0 : step - deficit;
    std::memcpy(
      input_clip_buffer_.data() + static_cast<size_t>(step) * frame_elems,
      clip_buffer_[src_idx].data(), frame_elems * sizeof(float));
  }

  const auto inference_start = std::chrono::steady_clock::now();
  engine_->infer(input_clip_buffer_.data(), heat_logits_buffer_.data(), confidence_logits_buffer_.data());
  result.inference_elapsed_s =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - inference_start).count();

  const ColumnPoints points = heatmap_to_points(
    heat_logits_buffer_.data(), confidence_logits_buffer_.data(), model_h, model_w);
  const auto [m, b] = dsac_line_fit(points, config_.dsac, rng_);

  // endpoints_from_mb(m, b) = [b - m, b + m] (line evaluated at x=-1, x=+1
  // in normalized coordinates), then ((. + 1) * 0.5).clamp(0, 1) maps the
  // normalized-y range [-1,1] to [0,1] -- matches models.heads.WLSHL/DSACHL
  // /TRiDHorizon.forward and inference/pipeline.py's
  // `endpoints_norm_to_original`.
  const double y_left_norm01 = std::clamp((b - m + 1.0) * 0.5, 0.0, 1.0);
  const double y_right_norm01 = std::clamp((b + m + 1.0) * 0.5, 0.0, 1.0);

  HorizonLine coarse;
  coarse.y_left = y_left_norm01 * static_cast<double>(orig_h - 1);
  coarse.y_right = y_right_norm01 * static_cast<double>(orig_h - 1);
  coarse.width = orig_w;
  coarse.height = orig_h;

  const RoiGateResult gate = apply_bounded_roi(
    bgr_frame, coarse, config_.roi_enable, config_.roi_gate_enable,
    config_.grad_score, config_.fusion, config_.roi_gate);

  result.y = gate.final.y_center();
  result.theta_deg = gate.final.theta_deg();
  result.valid = true;
  result.roi_accepted = gate.accepted;
  result.roi_reason = gate.reason;

  // Debug visualization only: sigmoid-scaled raw heat logits at model
  // resolution. The detector itself extracts the horizon line via a
  // per-column softmax expectation (heatmap_to_points), not a threshold on
  // this image -- see README.md.
  const cv::Mat heat_mat(model_h, model_w, CV_32F, heat_logits_buffer_.data());
  cv::Mat neg_exp;
  cv::exp(-heat_mat, neg_exp);
  cv::Mat heat_sigmoid;
  cv::divide(1.0, neg_exp + 1.0, heat_sigmoid);
  heat_sigmoid.convertTo(result.heatmap, CV_8UC1, 255.0);

  result.elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return result;
}

} // namespace trid_cpp
