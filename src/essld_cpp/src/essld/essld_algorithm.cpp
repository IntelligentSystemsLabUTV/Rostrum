/**
 * ESSLD detector: TensorRT segmentation + coarse/refined line extraction +
 * temporal smoothing, tying together essld_trt.cpp and essld_processing.cpp.
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

#include <essld_cpp/essld_cpp.hpp>

#include <opencv2/imgproc.hpp>

#include <cstring>

namespace essld_cpp
{

ESSLDDetector::ESSLDDetector(std::shared_ptr<TRTEngine> engine, ESSLDConfig config)
: engine_(std::move(engine)), config_(std::move(config))
{
  input_buffer_.resize(3ULL * config_.input_height * config_.input_width);
  logits_buffer_.resize(1ULL * config_.input_height * config_.input_width);
}

FrameResult ESSLDDetector::process_frame(const cv::Mat & bgr_frame)
{
  const auto start = std::chrono::steady_clock::now();
  FrameResult result;

  const int orig_h = bgr_frame.rows;
  const int orig_w = bgr_frame.cols;

  // Preprocess: resize -> RGB -> CHW float32 in [0,1], matching demo_video.py.
  cv::Mat resized;
  cv::resize(bgr_frame, resized, cv::Size(config_.input_width, config_.input_height));
  cv::Mat rgb;
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
  cv::Mat rgb_f;
  rgb.convertTo(rgb_f, CV_32FC3, 1.0 / 255.0);

  std::vector<cv::Mat> channels(3);
  cv::split(rgb_f, channels);
  const size_t plane = static_cast<size_t>(config_.input_height) * config_.input_width;
  for (int c = 0; c < 3; ++c) {
    std::memcpy(input_buffer_.data() + c * plane, channels[c].ptr<float>(0), plane * sizeof(float));
  }

  engine_->infer(input_buffer_.data(), logits_buffer_.data());

  // Sigmoid on the raw logits, then resize the probability map back up to
  // the original frame size before thresholding (matches demo_video.py's
  // order: sigmoid at model resolution, then cv2.resize, then threshold).
  const cv::Mat logits_mat(config_.input_height, config_.input_width, CV_32F, logits_buffer_.data());
  cv::Mat neg_exp;
  cv::exp(-logits_mat, neg_exp);
  cv::Mat prob_map;
  cv::divide(1.0, neg_exp + 1.0, prob_map);

  cv::Mat prob_map_orig;
  cv::resize(prob_map, prob_map_orig, cv::Size(orig_w, orig_h));

  cv::Mat binary_mask;
  cv::threshold(prob_map_orig, binary_mask, config_.mask_threshold, 255, cv::THRESH_BINARY);
  binary_mask.convertTo(binary_mask, CV_8UC1);
  result.mask = binary_mask;

  const auto coarse = get_coarse_line_from_mask(binary_mask);
  if (coarse) {
    double final_y = coarse->first;
    double final_angle = coarse->second;

    try {
      const auto refined = refine_horizon_stage3(bgr_frame, final_y, final_angle, config_);
      if (refined) {
        final_y = refined->first;
        final_angle = refined->second;
      }
    } catch (const std::exception &) {
      // Matches the Python reference's bare try/except: pass around
      // refine_horizon_stage3 -- fall back to the coarse detection.
    }

    if (has_last_) {
      final_y = config_.smoothing_alpha * final_y + (1.0 - config_.smoothing_alpha) * last_y_;
      final_angle = config_.smoothing_alpha * final_angle + (1.0 - config_.smoothing_alpha) * last_theta_;
    }
    last_y_ = final_y;
    last_theta_ = final_angle;
    has_last_ = true;

    result.y = final_y;
    result.theta_deg = final_angle;
    result.valid = true;
  } else if (config_.hold_last_on_failure && has_last_) {
    result.y = last_y_;
    result.theta_deg = last_theta_;
    result.valid = false;
  } else {
    result.valid = false;
  }

  result.elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return result;
}

} // namespace essld_cpp
