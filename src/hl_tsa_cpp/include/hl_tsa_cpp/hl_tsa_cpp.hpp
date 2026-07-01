/**
 * HL-TSA CPU horizon-line detector.
 *
 * dotX Automation s.r.l. <info@dotxautomation.com>
 */

/**
 * Copyright 2024 dotX Automation s.r.l.
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

#include <Eigen/Dense>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

using sensor_msgs::msg::Image;
using std_msgs::msg::Float64MultiArray;

namespace hl_tsa_cpp
{

struct HorizonState
{
  double y = 0.0;
  double theta_deg = 0.0;
};

struct HLTSAConfig
{
  int listener_frames = 16;
  int max_iterations = 3;
  double zscore = 1.96;
  int canny_low = 20;
  int canny_high = 60;
  double area_open_fraction = 0.005;
  int hough_threshold = 15;
  double hough_theta_resolution_deg = 0.25;
  double max_abs_horizon_angle_deg = 35.0;
  double min_roi_height_fraction = 0.075;
  double error_band_fraction = 0.025;
  double min_y_sd = 0.05;
  double min_theta_sd = 0.01;
};

struct FrameResult
{
  int frame = 0;
  std::string block;
  HorizonState state;
  double error = 0.0;
  double elapsed_s = 0.0;
  int iterations = 0;
  bool absence_flag = false;
  cv::Mat roi_mask;
};

struct ExecutionTimes
{
  double setup_s = 0.0;
  double listener_s = 0.0;
  double model_fit_s = 0.0;
  double main_loop_s = 0.0;
  double total_s = 0.0;

  double fps(int frames) const
  {
    return total_s > 0.0 ? static_cast<double>(frames) / total_s : 0.0;
  }
};

struct VideoResult
{
  std::vector<HorizonState> states;
  std::vector<FrameResult> frame_results;
  ExecutionTimes timings;
};

struct VideoOptions
{
  std::string video_path;
  std::string states_csv;
  std::string frame_timing_csv;
  std::string output_video;
  int max_frames = 0;
  bool show = false;
};

/**
 * Univariate ARIMA(2,d,0) mean model with a GARCH(1,1) conditional-variance model,
 * matching MATLAB's `arima('ARLags',1:2,'D',d,'Variance',garch(1,1))` as used by
 * HL_Detect_TSA.m. Parameters are jointly estimated by maximizing the exact Gaussian
 * conditional log-likelihood (Nelder-Mead simplex, since no gradient/toolbox is
 * available in C++); see hl_tsa_algorithm.cpp for the documented numerical choices
 * (presample backcast, parameter constraints) that MATLAB does not disclose exactly.
 */
class ArimaGarchModel
{
public:
  static constexpr int kOrder = 2;

  void fit(const std::vector<double> & window, int d, double variance_floor);
  std::pair<double, double> forecast(const std::vector<double> & window) const;
  int differencing() const { return d_; }

private:
  bool has_constant_ = false;
  int d_ = 0;
  double c_ = 0.0;
  double phi1_ = 0.0;
  double phi2_ = 0.0;
  double omega_ = 0.0;
  double alpha_ = 0.0;
  double beta_ = 0.0;
  double backcast_variance_ = 0.0;
  double variance_floor_ = 0.0;
  bool fitted_ = false;
};

class HLTSADetector
{
public:
  explicit HLTSADetector(HLTSAConfig config = HLTSAConfig());

  void reset();
  FrameResult process_frame(const cv::Mat & frame);

  const HLTSAConfig & config() const { return config_; }
  const std::vector<HorizonState> & states() const { return states_; }
  const std::vector<double> & errors() const { return errors_; }
  bool model_ready() const { return model_ready_; }
  double model_fit_seconds() const { return model_fit_seconds_; }
  int processed_frames() const { return processed_frames_; }

  HorizonState hlda(const cv::Mat & frame) const;
  double get_error(const HorizonState & state, const cv::Mat & frame) const;
  std::pair<cv::Mat, std::pair<int, int>> roi_rect(
    const cv::Mat & frame,
    const HorizonState & state,
    int min_height) const;
  bool roi_tsm(
    const cv::Mat & frame,
    const HorizonState & state,
    double y_sd,
    double theta_sd,
    cv::Mat & roi,
    cv::Mat & mask,
    int & delta_height) const;

private:
  struct HoughCandidate
  {
    double y = 0.0;
    double theta_deg = 0.0;
    int rank = 0;
  };

  std::vector<HoughCandidate> hough_candidates(const cv::Mat & frame) const;
  cv::Mat area_open(const cv::Mat & edges) const;
  std::pair<cv::Mat, cv::Mat> find_indices_of_regions(
    const HorizonState & state,
    int width,
    int height,
    double delta_h) const;
  FrameResult listener_step(const cv::Mat & frame, const std::chrono::steady_clock::time_point & start);
  FrameResult main_step(const cv::Mat & frame, const std::chrono::steady_clock::time_point & start);
  FrameResult presence_detector(const cv::Mat & frame, const std::chrono::steady_clock::time_point & start);
  void fit_models();
  bool needs_control_loop(
    const HorizonState & local_state,
    double error,
    int frame_height) const;
  std::vector<double> trailing_values(bool use_theta) const;

  HLTSAConfig config_;
  std::vector<HorizonState> states_;
  std::vector<double> errors_;
  ArimaGarchModel y_model_;
  ArimaGarchModel theta_model_;
  bool model_ready_ = false;
  bool absent_ = false;
  int last_present_index_ = 0;
  int processed_frames_ = 0;
  int width_ = 0;
  int height_ = 0;
  int min_roi_height_ = 0;
  double model_fit_seconds_ = 0.0;
};

class HLTSA : public dua_node::NodeBase
{
public:
  HLTSA(const rclcpp::NodeOptions & node_options = rclcpp::NodeOptions());
  ~HLTSA();

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
  bool subscribers_best_effort_qos_;
  int64_t subscribers_depth_;
  std::string subscribers_topic_name_image_;
  std::string subscribers_transport_;
  std::string output_states_csv_;
  std::string output_frame_timing_csv_;
  int64_t listener_frames_;
  int64_t max_iterations_;
  double zscore_;
  int64_t canny_low_;
  int64_t canny_high_;
  double area_open_fraction_;
  int64_t hough_threshold_;
  double hough_theta_resolution_deg_;
  double max_abs_horizon_angle_deg_;
  double min_roi_height_fraction_;
  double error_band_fraction_;
  double min_y_sd_;
  double min_theta_sd_;

  HLTSAConfig detector_config_;
  HLTSADetector detector_;
  bool running_ = false;

  rclcpp::Publisher<Float64MultiArray>::SharedPtr state_pub_;
  std::shared_ptr<image_transport::Publisher> annotated_pub_;
  std::shared_ptr<image_transport::Subscriber> image_sub_;

  std::mutex image_queue_mutex_;
  std::condition_variable image_queue_cv_;
  std::deque<Image::ConstSharedPtr> image_queue_;
  std::thread image_processing_thread_;
  bool image_processing_stop_ = false;
  size_t dropped_images_ = 0;

  std::ofstream states_csv_;
  std::ofstream frame_timing_csv_;
};

cv::Mat draw_horizon(const cv::Mat & frame, const HorizonState & state);
cv::Mat draw_roi(const cv::Mat & frame, const cv::Mat & mask);
cv::Mat draw_result(const cv::Mat & frame, const FrameResult & result);

VideoResult process_video(const VideoOptions & options, const HLTSAConfig & config);
void save_states_csv(const std::string & path, const std::vector<HorizonState> & states);
void save_frame_timing_csv(const std::string & path, const std::vector<FrameResult> & frame_results);

} // namespace hl_tsa_cpp
