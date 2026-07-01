/**
 * ESSLD GPU sea-sky line detector.
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

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

using sensor_msgs::msg::Image;
using std_msgs::msg::Float64MultiArray;

namespace essld_cpp
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
        RCLCPP_ERROR(rclcpp::get_logger("essld_trt"), "%s", msg);
        break;
      case Severity::kWARNING:
        RCLCPP_WARN(rclcpp::get_logger("essld_trt"), "%s", msg);
        break;
      default:
        // Ignore kINFO and kVERBOSE.
        break;
    }
  }
};

/**
 * Wraps a deserialized TensorRT engine for the DCEUNet segmentation network.
 *
 * Expects exactly one FLOAT input tensor shaped (1,3,H,W) and one FLOAT
 * output tensor shaped (1,1,H,W) (both engines shipped with the ESSLD tool,
 * FP32 and FP16, expose FLOAT32 I/O bindings; only their internal precision
 * differs). Fails fast in the constructor if a loaded engine does not match
 * this layout, or does not match the width/height the node was configured
 * with.
 */
class TRTEngine
{
public:
  TRTEngine(const std::string & engine_path, int gpu_id, int expected_width, int expected_height);
  ~TRTEngine();

  TRTEngine(const TRTEngine &) = delete;
  TRTEngine & operator=(const TRTEngine &) = delete;

  int input_width() const { return input_width_; }
  int input_height() const { return input_height_; }

  /**
   * @brief Runs one synchronous inference pass.
   *
   * @param input_chw NCHW (N=1), RGB, [0,1]-normalized input buffer; must
   *   have 3 * input_height() * input_width() elements.
   * @param output_logits Output buffer for the raw (pre-sigmoid) logits;
   *   must have input_height() * input_width() elements.
   */
  void infer(const float * input_chw, float * output_logits);

private:
  TRTLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  std::string input_name_;
  std::string output_name_;
  int input_width_ = 0;
  int input_height_ = 0;
  size_t input_elems_ = 0;
  size_t output_elems_ = 0;

  float * host_input_ = nullptr;
  float * host_output_ = nullptr;
  void * device_input_ = nullptr;
  void * device_output_ = nullptr;
  cudaStream_t stream_ = nullptr;
};

/**
 * Tunable algorithm parameters, mirroring tools/ESSLD/config.py for the
 * knobs actually used by get_coarse_line_from_mask/refine_horizon_stage3.
 * Magic numbers that the Python reference never varies (CLAHE settings, the
 * fusion-stage Canny thresholds, inlier/area cleanup thresholds) are left as
 * fixed constants in essld_processing.cpp instead of being duplicated here.
 */
struct ESSLDConfig
{
  int input_width = 512;
  int input_height = 256;
  double mask_threshold = 0.5;

  std::vector<int64_t> median_filter_sizes {1, 5, 7};
  std::vector<double> canny_fusion_weights {0.5, 0.3, 0.2};
  std::vector<double> confidence_map_sigmas {10.0, 15.0, 20.0};
  std::vector<double> confidence_map_weights {0.5, 0.3, 0.2};
  int fused_map_final_threshold = 60;
  int hough_threshold = 50;
  int hough_min_line_length = 60;
  int hough_max_line_gap = 100;
  double grad_score = 80.0;

  double smoothing_alpha = 0.7;
  bool hold_last_on_failure = true;
};

/**
 * Result of processing a single frame.
 */
struct FrameResult
{
  double y = 0.0;
  double theta_deg = 0.0;
  bool valid = false;
  double elapsed_s = 0.0;
  double inference_elapsed_s = 0.0;
  cv::Mat mask;
};

/**
 * Ties the TensorRT segmentation engine together with the coarse-line and
 * dual multi-scale fusion refinement steps ported from
 * tools/ESSLD/utils/processing.py, plus temporal smoothing. Has no ROS
 * dependency, mirroring hl_tsa_cpp::HLTSADetector.
 */
class ESSLDDetector
{
public:
  ESSLDDetector(std::shared_ptr<TRTEngine> engine, ESSLDConfig config);

  FrameResult process_frame(const cv::Mat & bgr_frame);

  const ESSLDConfig & config() const { return config_; }

private:
  std::shared_ptr<TRTEngine> engine_;
  ESSLDConfig config_;

  bool has_last_ = false;
  double last_y_ = 0.0;
  double last_theta_ = 0.0;

  std::vector<float> input_buffer_;
  std::vector<float> logits_buffer_;
};

class ESSLD : public dua_node::NodeBase
{
public:
  ESSLD(const rclcpp::NodeOptions & node_options = rclcpp::NodeOptions());
  ~ESSLD();

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
  bool publish_mask_;
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
  double mask_threshold_;
  std::vector<int64_t> fusion_median_filter_sizes_;
  std::vector<double> fusion_canny_fusion_weights_;
  std::vector<double> fusion_confidence_map_sigmas_;
  std::vector<double> fusion_confidence_map_weights_;
  int64_t fusion_fused_map_final_threshold_;
  int64_t fusion_hough_threshold_;
  int64_t fusion_hough_min_line_length_;
  int64_t fusion_hough_max_line_gap_;
  double fusion_grad_score_;
  double smoothing_alpha_;
  bool smoothing_hold_last_on_failure_;

  std::shared_ptr<TRTEngine> engine_;
  std::unique_ptr<ESSLDDetector> detector_;
  bool running_ = false;

  rclcpp::Publisher<Float64MultiArray>::SharedPtr state_pub_;
  std::shared_ptr<image_transport::Publisher> annotated_pub_;
  std::shared_ptr<image_transport::Publisher> mask_pub_;
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

/* --- Ported from tools/ESSLD/utils/processing.py (dead ray-cast/gourd-shape
   helpers, unused by get_coarse_line_from_mask/refine_horizon_stage3 in the
   Python reference, are intentionally not ported; see README.md). --- */

/**
 * @brief Fits a line to points via cv::fitLine, replicating
 *   utils.processing._robust_fit_line (including its near-vertical guard).
 *
 * @return {k, b} for y = k*x + b, or std::nullopt if fewer than 2 points.
 */
std::optional<std::pair<double, double>> robust_fit_line(
  const std::vector<cv::Point2f> & points,
  cv::DistanceTypes dist_type);

/**
 * @brief Closed-form weighted least-squares line fit, replicating
 *   np.polyfit(x, y, 1, w=weights) (weights may be empty for the unweighted
 *   case, replicating a plain np.polyfit(x, y, 1) call).
 *
 * @return {k, b} for y = k*x + b, or std::nullopt if fewer than 2 points or
 *   a degenerate (zero-variance) x distribution.
 */
std::optional<std::pair<double, double>> weighted_line_fit(
  const std::vector<double> & x,
  const std::vector<double> & y,
  const std::vector<double> & weights);

/**
 * @brief Direct translation of utils.processing._non_max_suppression_fast:
 *   Sobel-gradient-direction non-max suppression.
 */
cv::Mat non_max_suppression(const cv::Mat & magnitude, const cv::Mat & angle);

/**
 * @brief Direct port of utils.processing.get_coarse_line_from_mask.
 *
 * @return {y_mid, angle_deg} at the frame's horizontal midline, or
 *   std::nullopt if no plausible line could be extracted from the mask.
 */
std::optional<std::pair<double, double>> get_coarse_line_from_mask(
  const cv::Mat & binary_mask);

/**
 * @brief Direct port of utils.processing._run_dual_fusion_pipeline.
 *
 * @return {y_mid, angle_deg} local to roi_image, or std::nullopt on failure.
 */
std::optional<std::pair<double, double>> run_dual_fusion_pipeline(
  const cv::Mat & roi_image,
  const ESSLDConfig & config);

/**
 * @brief Direct port of utils.processing._calculate_dynamic_padding.
 */
int calculate_dynamic_padding(double grad_score);

/**
 * @brief Direct port of utils.processing._create_roi.
 *
 * @return roi image, forward perspective transform M, inverse transform
 *   M_inv, and (roi_height, width); roi image is empty on failure.
 */
struct RoiResult
{
  cv::Mat roi;
  cv::Mat m;
  cv::Mat m_inv;
  int roi_height = 0;
  int width = 0;
};
RoiResult create_roi(
  const cv::Mat & frame,
  double y_coarse,
  double angle_coarse,
  int padding);

/**
 * @brief Direct port of utils.processing.refine_horizon_stage3.
 *
 * @return {final_y, final_angle_deg} in frame coordinates, or std::nullopt
 *   on failure (caller should then fall back to the coarse result, matching
 *   the Python reference's bare try/except: pass).
 */
std::optional<std::pair<double, double>> refine_horizon_stage3(
  const cv::Mat & frame,
  double coarse_y,
  double coarse_angle_deg,
  const ESSLDConfig & config);

/**
 * @brief Draws the detected sea-sky line across the full frame width, like
 *   hl_tsa_cpp::draw_horizon.
 */
cv::Mat draw_horizon(const cv::Mat & frame, double y, double theta_deg);

} // namespace essld_cpp
