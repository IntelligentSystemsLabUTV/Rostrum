/**
 * TRiD-Horizon GPU ROS 2 node implementation.
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

#include <rclcpp_components/register_node_macro.hpp>

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <opencv2/imgproc.hpp>

namespace trid_cpp
{

TRiD::TRiD(const rclcpp::NodeOptions & node_options)
: NodeBase("trid", node_options, true)
{
  dua_init_node();

  if (model_engine_path_.empty()) {
    throw std::runtime_error("Parameter 'model.engine_path' is mandatory and was not set");
  }
  if (!std::filesystem::exists(model_engine_path_)) {
    throw std::runtime_error("TensorRT engine file not found: " + model_engine_path_);
  }

  TRiDConfig detector_config;
  detector_config.input_width = static_cast<int>(model_input_width_);
  detector_config.input_height = static_cast<int>(model_input_height_);
  detector_config.clip_length = static_cast<int>(model_clip_length_);
  detector_config.dsac.hypotheses = static_cast<int>(dsac_hypotheses_);
  detector_config.dsac.temperature = dsac_temperature_;
  detector_config.dsac.inlier_threshold = dsac_inlier_threshold_;
  detector_config.dsac.min_column_delta = dsac_min_column_delta_;
  detector_config.roi_enable = roi_enable_;
  detector_config.roi_gate_enable = roi_gate_enable_;
  detector_config.fusion.median_filter_sizes = fusion_median_filter_sizes_;
  detector_config.fusion.canny_fusion_weights = fusion_canny_fusion_weights_;
  detector_config.fusion.confidence_map_sigmas = fusion_confidence_map_sigmas_;
  detector_config.fusion.confidence_map_weights = fusion_confidence_map_weights_;
  detector_config.fusion.fused_map_final_threshold = static_cast<int>(fusion_fused_map_final_threshold_);
  detector_config.fusion.hough_threshold = static_cast<int>(fusion_hough_threshold_);
  detector_config.fusion.hough_min_line_length = static_cast<int>(fusion_hough_min_line_length_);
  detector_config.fusion.hough_max_line_gap = static_cast<int>(fusion_hough_max_line_gap_);
  detector_config.grad_score = fusion_grad_score_;
  detector_config.roi_gate.min_inside_roi_fraction = roi_gate_min_inside_roi_fraction_;
  detector_config.roi_gate.max_center_shift_half_height_frac = roi_gate_max_center_shift_half_height_frac_;
  detector_config.roi_gate.max_endpoint_shift_roi_height_frac = roi_gate_max_endpoint_shift_roi_height_frac_;
  detector_config.roi_gate.max_angle_change_deg = roi_gate_max_angle_change_deg_;
  detector_config.roi_gate.min_candidate_count = static_cast<int>(roi_gate_min_candidate_count_);
  detector_config.roi_gate.min_candidate_span_frac = roi_gate_min_candidate_span_frac_;

  engine_ = std::make_shared<TRTEngine>(
    model_engine_path_,
    static_cast<int>(model_gpu_id_),
    detector_config.clip_length,
    detector_config.input_width,
    detector_config.input_height);
  detector_ = std::make_unique<TRiDDetector>(engine_, detector_config);

  running_ = autostart_;
  write_csv_headers();
  image_processing_thread_ = std::thread(&TRiD::image_processing_loop, this);
  RCLCPP_INFO(this->get_logger(), "TRiD GPU node initialized");
}

TRiD::~TRiD()
{
  running_ = false;
  {
    std::lock_guard<std::mutex> lock(image_queue_mutex_);
    image_processing_stop_ = true;
    image_queue_.clear();
  }
  image_queue_cv_.notify_all();
  if (image_processing_thread_.joinable()) {
    image_processing_thread_.join();
  }

  if (states_csv_.is_open()) {
    states_csv_.close();
  }
  if (frame_timing_csv_.is_open()) {
    frame_timing_csv_.close();
  }
}

void TRiD::init_publishers()
{
  state_pub_ = dua_create_publisher<Float64MultiArray>("~/state");

  if (publish_annotated_) {
    annotated_pub_ = std::make_shared<image_transport::Publisher>(
      image_transport::create_publisher(
        this,
        "~/annotated",
        dua_qos::Reliable::get_image_qos().get_rmw_qos_profile()));
    RCLCPP_INFO(this->get_logger(), "[TOPIC PUB] '%s' (image_transport)",
      annotated_pub_->getTopic().c_str());
  }

  if (publish_heatmap_) {
    heatmap_pub_ = std::make_shared<image_transport::Publisher>(
      image_transport::create_publisher(
        this,
        "~/heatmap",
        dua_qos::Reliable::get_image_qos().get_rmw_qos_profile()));
    RCLCPP_INFO(this->get_logger(), "[TOPIC PUB] '%s' (image_transport)",
      heatmap_pub_->getTopic().c_str());
  }
}

void TRiD::init_subscribers()
{
  const auto qos = subscribers_best_effort_qos_ ?
    dua_qos::BestEffort::get_image_qos(static_cast<uint>(subscribers_depth_)) :
    dua_qos::Reliable::get_image_qos(static_cast<uint>(subscribers_depth_));

  image_sub_ = std::make_shared<image_transport::Subscriber>(
    image_transport::create_subscription(
      this,
      subscribers_topic_name_image_,
      std::bind(&TRiD::callback_image, this, std::placeholders::_1),
      subscribers_transport_,
      qos.get_rmw_qos_profile()));
  RCLCPP_INFO(this->get_logger(), "[TOPIC SUB] '%s' (image_transport)",
    image_sub_->getTopic().c_str());
}

void TRiD::callback_image(const Image::ConstSharedPtr & image_msg)
{
  if (!running_) {
    return;
  }

  bool dropped = false;
  size_t dropped_images = 0;
  {
    std::lock_guard<std::mutex> lock(image_queue_mutex_);
    if (image_processing_stop_) {
      return;
    }
    const size_t queue_depth = static_cast<size_t>(std::max<int64_t>(1, subscribers_depth_));
    while (image_queue_.size() >= queue_depth) {
      image_queue_.pop_front();
      dropped = true;
      dropped_images = ++dropped_images_;
    }
    image_queue_.push_back(image_msg);
  }
  image_queue_cv_.notify_one();

  if (dropped) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(),
      *this->get_clock(),
      5000,
      "TRiD image processing queue full; dropped %zu old image(s). "
      "Increase subscribers.depth or lower the input FPS if every frame must be processed.",
      dropped_images);
  }
}

void TRiD::image_processing_loop()
{
  const auto context = this->get_node_base_interface()->get_context();
  while (rclcpp::ok(context)) {
    Image::ConstSharedPtr image_msg;
    {
      std::unique_lock<std::mutex> lock(image_queue_mutex_);
      image_queue_cv_.wait(lock, [this]() {
        return image_processing_stop_ || !image_queue_.empty();
      });
      if (image_processing_stop_ && image_queue_.empty()) {
        return;
      }
      image_msg = image_queue_.front();
      image_queue_.pop_front();
    }
    process_image(image_msg);
  }
}

void TRiD::process_image(const Image::ConstSharedPtr & image_msg)
{
  cv::Mat frame;
  try {
    dua_cv_bridge::msg_to_frame(image_msg, frame);
  } catch (const std::exception & ex) {
    RCLCPP_ERROR(this->get_logger(), "Image conversion failed: %s", ex.what());
    return;
  }
  if (image_msg->encoding == "rgb8") {
    cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
  }

  ++frame_count_;

  FrameResult result;
  try {
    result = detector_->process_frame(frame);
  } catch (const std::exception & ex) {
    RCLCPP_ERROR(this->get_logger(), "TRiD detection failed: %s", ex.what());
    return;
  }

  Float64MultiArray msg;
  msg.data = {
    result.y, result.theta_deg, result.elapsed_s * 1000.0,
    result.valid ? 1.0 : 0.0, result.roi_accepted ? 1.0 : 0.0};
  state_pub_->publish(msg);

  append_state_csv(result);
  append_frame_timing_csv(result);

  if (publish_annotated_ && annotated_pub_) {
    cv::Mat annotated = draw_horizon(frame, result.y, result.theta_deg);
    auto annotated_msg = dua_cv_bridge::frame_to_msg(annotated, "bgr8");
    annotated_msg->header = image_msg->header;
    annotated_pub_->publish(annotated_msg);
  }

  if (publish_heatmap_ && heatmap_pub_ && !result.heatmap.empty()) {
    auto heatmap_msg = dua_cv_bridge::frame_to_msg(result.heatmap, "mono8");
    heatmap_msg->header = image_msg->header;
    heatmap_pub_->publish(heatmap_msg);
  }

  if (verbose_) {
    RCLCPP_INFO(
      this->get_logger(),
      "y=%.3f theta=%.3f roi_accepted=%d (%s) inference=%.3f ms elapsed=%.3f ms",
      result.y,
      result.theta_deg,
      result.roi_accepted,
      result.roi_reason.c_str(),
      result.inference_elapsed_s * 1000.0,
      result.elapsed_s * 1000.0);
  }
}

void TRiD::write_csv_headers()
{
  if (!output_states_csv_.empty()) {
    states_csv_.open(output_states_csv_);
    if (!states_csv_.is_open()) {
      RCLCPP_ERROR(this->get_logger(), "Could not open states CSV: %s", output_states_csv_.c_str());
    } else {
      states_csv_ << "frame,y,theta_deg\n";
    }
  }

  if (!output_frame_timing_csv_.empty()) {
    frame_timing_csv_.open(output_frame_timing_csv_);
    if (!frame_timing_csv_.is_open()) {
      RCLCPP_ERROR(this->get_logger(), "Could not open frame timing CSV: %s", output_frame_timing_csv_.c_str());
    } else {
      frame_timing_csv_ << "frame,elapsed_seconds,inference_seconds,y,theta_deg,roi_accepted,roi_reason\n";
    }
  }
}

void TRiD::append_state_csv(const FrameResult & result)
{
  if (!states_csv_.is_open()) {
    return;
  }
  states_csv_ << frame_count_ << ","
              << std::fixed << std::setprecision(6)
              << result.y << ","
              << result.theta_deg << "\n";
}

void TRiD::append_frame_timing_csv(const FrameResult & result)
{
  if (!frame_timing_csv_.is_open()) {
    return;
  }
  frame_timing_csv_ << frame_count_ << ","
                    << std::fixed << std::setprecision(9)
                    << result.elapsed_s << ","
                    << result.inference_elapsed_s << ","
                    << std::setprecision(6)
                    << result.y << ","
                    << result.theta_deg << ","
                    << (result.roi_accepted ? 1 : 0) << ","
                    << result.roi_reason << "\n";
}

} // namespace trid_cpp

RCLCPP_COMPONENTS_REGISTER_NODE(trid_cpp::TRiD)
