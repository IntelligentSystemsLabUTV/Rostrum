/**
 * HL-TSA CPU ROS 2 node implementation.
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

#include <hl_tsa_cpp/hl_tsa_cpp.hpp>

#include <rclcpp_components/register_node_macro.hpp>

#include <algorithm>
#include <iomanip>

namespace hl_tsa_cpp
{

HLTSA::HLTSA(const rclcpp::NodeOptions & node_options)
: NodeBase("hl_tsa", node_options, true)
{
  dua_init_node();

  detector_config_.listener_frames = static_cast<int>(listener_frames_);
  detector_config_.max_iterations = static_cast<int>(max_iterations_);
  detector_config_.zscore = zscore_;
  detector_config_.canny_low = static_cast<int>(canny_low_);
  detector_config_.canny_high = static_cast<int>(canny_high_);
  detector_config_.area_open_fraction = area_open_fraction_;
  detector_config_.hough_threshold = static_cast<int>(hough_threshold_);
  detector_config_.hough_theta_resolution_deg = hough_theta_resolution_deg_;
  detector_config_.max_abs_horizon_angle_deg = max_abs_horizon_angle_deg_;
  detector_config_.min_roi_height_fraction = min_roi_height_fraction_;
  detector_config_.error_band_fraction = error_band_fraction_;
  detector_config_.min_y_sd = min_y_sd_;
  detector_config_.min_theta_sd = min_theta_sd_;
  detector_ = HLTSADetector(detector_config_);

  running_ = autostart_;
  write_csv_headers();
  image_processing_thread_ = std::thread(&HLTSA::image_processing_loop, this);
  RCLCPP_INFO(this->get_logger(), "HL-TSA CPU node initialized");
}

HLTSA::~HLTSA()
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

void HLTSA::init_publishers()
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
}

void HLTSA::init_subscribers()
{
  const auto qos = subscribers_best_effort_qos_ ?
    dua_qos::BestEffort::get_image_qos(static_cast<uint>(subscribers_depth_)) :
    dua_qos::Reliable::get_image_qos(static_cast<uint>(subscribers_depth_));

  image_sub_ = std::make_shared<image_transport::Subscriber>(
    image_transport::create_subscription(
      this,
      subscribers_topic_name_image_,
      std::bind(&HLTSA::callback_image, this, std::placeholders::_1),
      subscribers_transport_,
      qos.get_rmw_qos_profile()));
  RCLCPP_INFO(this->get_logger(), "[TOPIC SUB] '%s' (image_transport)",
    image_sub_->getTopic().c_str());
}

void HLTSA::callback_image(const Image::ConstSharedPtr & image_msg)
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
      "HL-TSA image processing queue full; dropped %zu old image(s). "
      "Increase subscribers.depth or lower the video publisher FPS if every frame must be processed.",
      dropped_images);
  }
}

void HLTSA::image_processing_loop()
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

void HLTSA::process_image(const Image::ConstSharedPtr & image_msg)
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

  FrameResult result;
  try {
    result = detector_.process_frame(frame);
  } catch (const std::exception & ex) {
    RCLCPP_ERROR(this->get_logger(), "HL detection failed: %s", ex.what());
    return;
  }

  Float64MultiArray msg;
  msg.data = {
    result.state.y,
    result.state.theta_deg,
    result.elapsed_s * 1000.0,
    static_cast<double>(result.iterations),
    result.absence_flag ? 1.0 : 0.0};
  state_pub_->publish(msg);

  append_state_csv(result);
  append_frame_timing_csv(result);

  if (publish_annotated_ && annotated_pub_) {
    cv::Mat annotated = draw_result(frame, result);
    auto annotated_msg = dua_cv_bridge::frame_to_msg(annotated, "bgr8");
    annotated_msg->header = image_msg->header;
    annotated_pub_->publish(annotated_msg);
  }

  if (verbose_) {
    RCLCPP_INFO(
      this->get_logger(),
      "frame=%d block=%s y=%.3f theta=%.3f elapsed=%.3f ms",
      result.frame,
      result.block.c_str(),
      result.state.y,
      result.state.theta_deg,
      result.elapsed_s * 1000.0);
  }
}

void HLTSA::write_csv_headers()
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
      frame_timing_csv_ << "frame,block,elapsed_seconds,y,theta_deg,iterations,absence_flag\n";
    }
  }
}

void HLTSA::append_state_csv(const FrameResult & result)
{
  if (!states_csv_.is_open()) {
    return;
  }
  states_csv_ << result.frame << ","
              << std::fixed << std::setprecision(6)
              << result.state.y << ","
              << result.state.theta_deg << "\n";
}

void HLTSA::append_frame_timing_csv(const FrameResult & result)
{
  if (!frame_timing_csv_.is_open()) {
    return;
  }
  frame_timing_csv_ << result.frame << ","
                    << result.block << ","
                    << std::fixed << std::setprecision(9)
                    << result.elapsed_s << ","
                    << std::setprecision(6)
                    << result.state.y << ","
                    << result.state.theta_deg << ","
                    << result.iterations << ","
                    << (result.absence_flag ? 1 : 0) << "\n";
}

} // namespace hl_tsa_cpp

RCLCPP_COMPONENTS_REGISTER_NODE(hl_tsa_cpp::HLTSA)
