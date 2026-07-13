/**
 * TRiD-Horizon GPU temporal horizon-line detector.
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

#pragma once

#include <dua_cv_bridge/dua_cv_bridge.hpp>
#include <dua_node_cpp/dua_node.hpp>
#include <dua_qos_cpp/dua_qos.hpp>

#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cuda_runtime.h>
#include <NvInfer.h>

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using sensor_msgs::msg::Image;
using std_msgs::msg::Float64MultiArray;

namespace trid_cpp
{

/**
 * Routes TensorRT log messages through rclcpp logging.
 */
class TRTLogger : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char * msg) noexcept override
  {
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
      case Severity::kERROR:
        RCLCPP_ERROR(rclcpp::get_logger("trid_trt"), "%s", msg);
        break;
      case Severity::kWARNING:
        RCLCPP_WARN(rclcpp::get_logger("trid_trt"), "%s", msg);
        break;
      default:
        // Ignore kINFO and kVERBOSE.
        break;
    }
  }
};

/**
 * Wraps a deserialized TensorRT engine for the TRiD-Horizon backbone +
 * temporal (ConvGRU) + per-column heat/confidence heads, exported by
 * tools/TRiD/export_onnx.py (see that file's TRiDExportWrapper docstring for
 * why the model's DSACLineFit head is *not* part of the engine: it calls
 * torch.multinomial, which TensorRT 10.9's ONNX parser rejects).
 *
 * Expects exactly one FLOAT input tensor "input" shaped
 * (1, clip_length, 3, H, W) and two FLOAT output tensors: "heat_logits"
 * shaped (1, 1, H, W) and "confidence_logits" shaped (1, 1, W) (all engines
 * built by export_onnx.py + trtexec, FP32 and FP16, expose FLOAT32 I/O
 * bindings; only their internal precision differs). Fails fast in the
 * constructor if a loaded engine does not match this layout, or does not
 * match the clip length/width/height the node was configured with.
 */
class TRTEngine
{
public:
  TRTEngine(
    const std::string & engine_path, int gpu_id, int clip_length,
    int expected_width, int expected_height);
  ~TRTEngine();

  TRTEngine(const TRTEngine &) = delete;
  TRTEngine & operator=(const TRTEngine &) = delete;

  int input_width() const { return input_width_; }
  int input_height() const { return input_height_; }
  int clip_length() const { return clip_length_; }

  /**
   * @brief Runs one synchronous inference pass.
   *
   * @param input_clip NCHW-per-frame, RGB, [0,1]-normalized clip buffer of
   *   clip_length() stacked frames in temporal order (oldest first); must
   *   have clip_length() * 3 * input_height() * input_width() elements.
   * @param heat_logits Output buffer for the raw per-pixel heat logits of
   *   the last clip frame; must have input_height() * input_width() elements.
   * @param confidence_logits Output buffer for the raw per-column confidence
   *   logits of the last clip frame; must have input_width() elements.
   */
  void infer(const float * input_clip, float * heat_logits, float * confidence_logits);

private:
  TRTLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  std::string input_name_;
  std::string heat_name_;
  std::string conf_name_;
  int clip_length_ = 0;
  int input_width_ = 0;
  int input_height_ = 0;
  size_t input_elems_ = 0;
  size_t heat_elems_ = 0;
  size_t conf_elems_ = 0;

  float * host_input_ = nullptr;
  float * host_heat_ = nullptr;
  float * host_conf_ = nullptr;
  void * device_input_ = nullptr;
  void * device_heat_ = nullptr;
  void * device_conf_ = nullptr;
  cudaStream_t stream_ = nullptr;
};

/**
 * A horizon line expressed by its y-coordinate at the leftmost and
 * rightmost frame columns, mirroring tools/TRiD/inference/geometry.py's
 * HorizonLine.
 */
struct HorizonLine
{
  double y_left = 0.0;
  double y_right = 0.0;
  int width = 0;
  int height = 0;

  double slope() const
  {
    return (y_right - y_left) / std::max(static_cast<double>(width - 1), 1.0);
  }
  double intercept() const { return y_left; }
  double y_center() const { return slope() * (static_cast<double>(width) / 2.0) + intercept(); }
  double theta_deg() const;
};

/**
 * @brief Direct port of geometry.line_from_center_angle.
 */
HorizonLine line_from_center_angle(double y_center, double theta_deg, int width, int height);

/**
 * @brief Direct port of models.heads.heatmap_to_points, operating on one
 *   engine output pair (the last clip frame's raw logits).
 *
 * @param heat_logits input_height() * input_width() raw logits, row-major.
 * @param confidence_logits input_width() raw logits.
 * @return {x, y, confidence}, each of length width, with x/y in [-1,1] and
 *   confidence in [0,1] (post-sigmoid).
 */
struct ColumnPoints
{
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> confidence;
};
ColumnPoints heatmap_to_points(
  const float * heat_logits, const float * confidence_logits, int height, int width);

/**
 * Tunable parameters for models.heads.DSACLineFit, mirroring the fixed
 * construction `DSACLineFit(hypotheses=64)` used by TRiDHorizon(64, 64) in
 * tools/TRiD/inference/pipeline.py (the other three fields keep that class's
 * own defaults, since the reference tool never overrides them).
 */
struct DSACConfig
{
  int hypotheses = 64;
  double temperature = 0.1;
  double inlier_threshold = 0.05;
  double min_column_delta = 0.05;
};

/**
 * @brief Direct port of models.heads.DSACLineFit.forward: samples
 *   `hypotheses` pairs of columns from a confidence-weighted categorical
 *   distribution (torch.multinomial with replacement -> std::discrete_
 *   distribution here, same semantics, independent RNG stream -- see
 *   README.md for why this is not bit-exact with the PyTorch reference and
 *   why that does not matter for this differentiable-RANSAC-style head).
 *
 * @return {m, b} for the fitted line y = m*x + b in normalized [-1,1]
 *   coordinates.
 */
std::pair<double, double> dsac_line_fit(
  const ColumnPoints & points, const DSACConfig & config, std::mt19937 & rng);

/**
 * @brief Direct port of utils/processing.py's `_robust_fit_line` (as also
 *   ported in essld_cpp).
 */
std::optional<std::pair<double, double>> robust_fit_line(
  const std::vector<cv::Point2f> & points,
  cv::DistanceTypes dist_type);

/**
 * @brief Closed-form weighted least-squares line fit, replicating
 *   np.polyfit(x, y, 1, w=weights) (weights may be empty for the unweighted
 *   case), as also ported in essld_cpp.
 */
std::optional<std::pair<double, double>> weighted_line_fit(
  const std::vector<double> & x,
  const std::vector<double> & y,
  const std::vector<double> & weights);

/**
 * @brief Direct translation of utils/processing.py's
 *   `_non_max_suppression_fast` (numba-jitted in the reference; the ROI
 *   processed here is small enough that a plain nested loop suffices), as
 *   also ported in essld_cpp.
 */
cv::Mat non_max_suppression(const cv::Mat & magnitude, const cv::Mat & angle);

/**
 * Tunable ROI dual-fusion refinement parameters, mirroring
 * tools/TRiD/config.py's module-level constants (MEDIAN_FILTER_SIZES,
 * CANNY_FUSION_WEIGHTS, CONFIDENCE_MAP_SIGMAS, CONFIDENCE_MAP_WEIGHTS,
 * FUSED_MAP_FINAL_THRESHOLD, HOUGH_THRESHOLD, HOUGH_MIN_LINE_LENGTH,
 * HOUGH_MAX_LINE_GAP). Magic numbers the Python reference never varies
 * (CLAHE settings, the fusion-stage Canny thresholds, inlier/area cleanup
 * thresholds) are left as fixed constants in trid_algorithm.cpp instead.
 */
struct FusionConfig
{
  std::vector<int64_t> median_filter_sizes {1, 5, 7};
  std::vector<double> canny_fusion_weights {0.5, 0.3, 0.2};
  std::vector<double> confidence_map_sigmas {10.0, 15.0, 20.0};
  std::vector<double> confidence_map_weights {0.5, 0.3, 0.2};
  int fused_map_final_threshold = 60;
  int hough_threshold = 30;
  int hough_min_line_length = 50;
  int hough_max_line_gap = 20;
};

/**
 * @brief Direct port of utils/processing.py's `_calculate_dynamic_padding`,
 *   as also ported in essld_cpp.
 */
int calculate_dynamic_padding(double grad_score);

/**
 * @brief Direct port of inference/postprocess.py's `create_roi`, taking the
 *   coarse line's center/angle (rather than a raw y/angle pair) as input;
 *   otherwise identical to essld_cpp's `create_roi`.
 */
struct RoiResult
{
  cv::Mat roi;
  cv::Mat m;
  cv::Mat m_inv;
  int roi_height = 0;
  int width = 0;
  std::array<cv::Point2f, 4> corners {};  // lt, rt, rb, lb, matching create_roi's src_pts order
};
RoiResult create_roi(const cv::Mat & frame, double y_coarse, double angle_coarse, int padding);

/**
 * @brief Direct port of inference/postprocess.py's `dual_fusion_pipeline`,
 *   as also ported in essld_cpp (`run_dual_fusion_pipeline`).
 *
 * @return {y_mid, angle_deg} local to roi_image, candidate point count, and
 *   candidate vertical span, or std::nullopt (with count/span still filled
 *   in) on failure -- matching roi_gate.py's need for those diagnostics even
 *   when the fit itself fails.
 */
struct FusionResult
{
  std::optional<std::pair<double, double>> line;
  int candidate_count = 0;
  double candidate_span = 0.0;
};
FusionResult run_dual_fusion_pipeline(const cv::Mat & roi_image, const FusionConfig & config);

/**
 * @brief Direct port of inference/postprocess.py's `refine_existing`.
 */
struct RefineResult
{
  std::optional<HorizonLine> refined;
  std::array<cv::Point2f, 4> roi_pts {};
  int padding = 0;
  int roi_height = 0;
  int candidate_count = 0;
  double candidate_span = 0.0;
};
RefineResult refine_existing(
  const cv::Mat & frame, const HorizonLine & coarse, double grad_score, const FusionConfig & config);

/**
 * Tunable ROI acceptance-gate parameters, mirroring
 * tools/TRiD/inference/roi_gate.py's ROIGateConfig defaults.
 */
struct RoiGateConfig
{
  double min_inside_roi_fraction = 0.98;
  double max_center_shift_half_height_frac = 0.80;
  double max_endpoint_shift_roi_height_frac = 1.00;
  double max_angle_change_deg = 4.0;
  int min_candidate_count = 10;
  double min_candidate_span_frac = 0.20;
};

/**
 * @brief Direct port of inference/roi_gate.py's `apply_bounded_roi`.
 */
struct RoiGateResult
{
  std::optional<HorizonLine> existing_refined;
  HorizonLine final;
  bool accepted = false;
  std::string reason;
  std::optional<std::array<cv::Point2f, 4>> roi_pts;
  double inside_roi_fraction = std::numeric_limits<double>::quiet_NaN();
  double center_correction = std::numeric_limits<double>::quiet_NaN();
  double endpoint_correction = std::numeric_limits<double>::quiet_NaN();
  double angle_correction = std::numeric_limits<double>::quiet_NaN();
  int candidate_count = 0;
  double candidate_span = 0.0;
};
RoiGateResult apply_bounded_roi(
  const cv::Mat & frame,
  const HorizonLine & coarse,
  bool enable_roi,
  bool enable_gate,
  double grad_score,
  const FusionConfig & fusion_config,
  const RoiGateConfig & gate_config);

/**
 * Full detector configuration.
 */
struct TRiDConfig
{
  int input_width = 512;
  int input_height = 256;
  int clip_length = 8;

  DSACConfig dsac;
  FusionConfig fusion;
  RoiGateConfig roi_gate;
  double grad_score = 80.0;

  bool roi_enable = true;
  bool roi_gate_enable = true;
};

/**
 * Result of processing a single frame.
 */
struct FrameResult
{
  double y = 0.0;
  double theta_deg = 0.0;
  bool valid = false;
  bool roi_accepted = false;
  std::string roi_reason;
  double elapsed_s = 0.0;
  double inference_elapsed_s = 0.0;
  cv::Mat heatmap;  // optional visualization: sigmoid(heat_logits), mono8, model resolution
};

/**
 * Ties the TensorRT backbone+temporal+column-head engine together with the
 * C++-side DSAC line fit and dual multi-scale ROI refinement/gate steps
 * ported from the tools/TRiD/inference Python modules. Has no ROS dependency, mirroring
 * essld_cpp::ESSLDDetector / hl_tsa_cpp::HLTSADetector.
 */
class TRiDDetector
{
public:
  TRiDDetector(std::shared_ptr<TRTEngine> engine, TRiDConfig config);

  FrameResult process_frame(const cv::Mat & bgr_frame);

  const TRiDConfig & config() const { return config_; }

private:
  std::shared_ptr<TRTEngine> engine_;
  TRiDConfig config_;

  std::deque<std::vector<float>> clip_buffer_;  // preprocessed CHW frames, oldest first, capped at clip_length
  std::vector<float> input_clip_buffer_;
  std::vector<float> heat_logits_buffer_;
  std::vector<float> confidence_logits_buffer_;
  std::mt19937 rng_;
};

class TRiD : public dua_node::NodeBase
{
public:
  TRiD(const rclcpp::NodeOptions & node_options = rclcpp::NodeOptions());
  ~TRiD();

private:
  void init_parameters() override;
  void init_publishers() override;
  void init_subscribers() override;
  void init_service_servers() override {}

  void callback_image(const Image::ConstSharedPtr & image_msg);
  void process_image(const Image::ConstSharedPtr & image_msg);
  void image_processing_loop();
  void write_csv_headers();
  void append_state_csv(const FrameResult & result);
  void append_frame_timing_csv(const FrameResult & result);

  /* Parameters. */
  bool autostart_;
  bool verbose_;
  bool publish_annotated_;
  bool publish_heatmap_;
  bool subscribers_best_effort_qos_;
  int64_t subscribers_depth_;
  std::string subscribers_topic_name_image_;
  std::string subscribers_transport_;
  std::string output_states_csv_;
  std::string output_frame_timing_csv_;
  std::string model_engine_path_;
  int64_t model_gpu_id_;
  int64_t model_input_width_;
  int64_t model_input_height_;
  int64_t model_clip_length_;
  int64_t dsac_hypotheses_;
  double dsac_temperature_;
  double dsac_inlier_threshold_;
  double dsac_min_column_delta_;
  bool roi_enable_;
  bool roi_gate_enable_;
  std::vector<int64_t> fusion_median_filter_sizes_;
  std::vector<double> fusion_canny_fusion_weights_;
  std::vector<double> fusion_confidence_map_sigmas_;
  std::vector<double> fusion_confidence_map_weights_;
  int64_t fusion_fused_map_final_threshold_;
  int64_t fusion_hough_threshold_;
  int64_t fusion_hough_min_line_length_;
  int64_t fusion_hough_max_line_gap_;
  double fusion_grad_score_;
  double roi_gate_min_inside_roi_fraction_;
  double roi_gate_max_center_shift_half_height_frac_;
  double roi_gate_max_endpoint_shift_roi_height_frac_;
  double roi_gate_max_angle_change_deg_;
  int64_t roi_gate_min_candidate_count_;
  double roi_gate_min_candidate_span_frac_;

  std::shared_ptr<TRTEngine> engine_;
  std::unique_ptr<TRiDDetector> detector_;
  bool running_ = false;

  rclcpp::Publisher<Float64MultiArray>::SharedPtr state_pub_;
  std::shared_ptr<image_transport::Publisher> annotated_pub_;
  std::shared_ptr<image_transport::Publisher> heatmap_pub_;
  std::shared_ptr<image_transport::Subscriber> image_sub_;

  std::mutex image_queue_mutex_;
  std::condition_variable image_queue_cv_;
  std::deque<Image::ConstSharedPtr> image_queue_;
  std::thread image_processing_thread_;
  bool image_processing_stop_ = false;
  size_t dropped_images_ = 0;
  size_t frame_count_ = 0;

  std::ofstream states_csv_;
  std::ofstream frame_timing_csv_;
};

/**
 * @brief Draws the detected horizon line across the full frame width, like
 *   essld_cpp::draw_horizon / hl_tsa_cpp::draw_horizon.
 */
cv::Mat draw_horizon(const cv::Mat & frame, double y, double theta_deg);

} // namespace trid_cpp
