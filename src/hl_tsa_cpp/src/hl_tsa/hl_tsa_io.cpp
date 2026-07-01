/**
 * HL-TSA CPU video and CSV utilities.
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

#include <opencv2/highgui.hpp>

#include <filesystem>
#include <iomanip>
#include <iostream>

namespace hl_tsa_cpp
{
namespace
{

double tand_local(double degrees)
{
  return std::tan(degrees * 3.14159265358979323846 / 180.0);
}

void ensure_parent_dir(const std::string & path)
{
  if (path.empty()) {
    return;
  }
  const std::filesystem::path fs_path(path);
  const auto parent = fs_path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
}

} // namespace

cv::Mat draw_horizon(const cv::Mat & frame, const HorizonState & state)
{
  cv::Mat out = frame.clone();
  const double slope = tand_local(state.theta_deg);
  const int y_left = static_cast<int>(std::round(slope * 1.0 + state.y -
    slope * static_cast<double>(out.cols) / 2.0));
  const int y_right = static_cast<int>(std::round(slope * static_cast<double>(out.cols) + state.y -
    slope * static_cast<double>(out.cols) / 2.0));
  cv::line(out, cv::Point(0, y_left), cv::Point(out.cols - 1, y_right), cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
  return out;
}

cv::Mat draw_roi(const cv::Mat & frame, const cv::Mat & mask)
{
  if (mask.empty()) {
    return frame;
  }
  cv::Mat out = frame.clone();
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  if (!contours.empty()) {
    cv::drawContours(out, contours, -1, cv::Scalar(0, 255, 255), 3, cv::LINE_AA);
  }
  return out;
}

cv::Mat draw_result(const cv::Mat & frame, const FrameResult & result)
{
  return draw_roi(draw_horizon(frame, result.state), result.roi_mask);
}

VideoResult process_video(const VideoOptions & options, const HLTSAConfig & config)
{
  const auto total_start = std::chrono::steady_clock::now();
  VideoResult result;

  const auto setup_start = std::chrono::steady_clock::now();
  cv::VideoCapture cap(options.video_path);
  if (!cap.isOpened()) {
    throw std::runtime_error("Could not open video: " + options.video_path);
  }

  const int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  const double fps = cap.get(cv::CAP_PROP_FPS) > 0.0 ? cap.get(cv::CAP_PROP_FPS) : 25.0;
  const int frame_count = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
  int total_limit = frame_count;
  if (options.max_frames > 0) {
    total_limit = frame_count > 0 ? std::min(frame_count, options.max_frames) : options.max_frames;
  }

  cv::VideoWriter writer;
  if (!options.output_video.empty()) {
    ensure_parent_dir(options.output_video);
    const int fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
    writer.open(options.output_video, fourcc, fps, cv::Size(width, height));
    if (!writer.isOpened()) {
      throw std::runtime_error("Could not create output video: " + options.output_video);
    }
  }

  HLTSADetector detector(config);
  result.timings.setup_s = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - setup_start).count();

  cv::Mat frame;
  int processed = 0;
  double listener_s = 0.0;
  double main_loop_s = 0.0;
  while ((total_limit <= 0 || processed < total_limit) && cap.read(frame)) {
    FrameResult frame_result = detector.process_frame(frame);
    if (frame_result.block == "listener") {
      listener_s += frame_result.elapsed_s;
    } else {
      main_loop_s += frame_result.elapsed_s;
    }

    result.states.push_back(frame_result.state);
    result.frame_results.push_back(frame_result);

    if (writer.isOpened() || options.show) {
      cv::Mat annotated = draw_result(frame, frame_result);
      if (writer.isOpened()) {
        writer.write(annotated);
      }
      if (options.show) {
        cv::imshow("HL-TSA C++", annotated);
        cv::waitKey(1);
      }
    }
    processed++;
  }

  result.timings.listener_s = listener_s;
  result.timings.model_fit_s = detector.model_fit_seconds();
  result.timings.main_loop_s = main_loop_s;
  result.timings.total_s = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - total_start).count();

  if (!options.states_csv.empty()) {
    save_states_csv(options.states_csv, result.states);
  }
  if (!options.frame_timing_csv.empty()) {
    save_frame_timing_csv(options.frame_timing_csv, result.frame_results);
  }
  return result;
}

void save_states_csv(const std::string & path, const std::vector<HorizonState> & states)
{
  ensure_parent_dir(path);
  std::ofstream out(path);
  if (!out.is_open()) {
    throw std::runtime_error("Could not open states CSV: " + path);
  }
  out << "frame,y,theta_deg\n";
  out << std::fixed << std::setprecision(6);
  for (size_t i = 0; i < states.size(); ++i) {
    out << (i + 1) << "," << states[i].y << "," << states[i].theta_deg << "\n";
  }
}

void save_frame_timing_csv(const std::string & path, const std::vector<FrameResult> & frame_results)
{
  ensure_parent_dir(path);
  std::ofstream out(path);
  if (!out.is_open()) {
    throw std::runtime_error("Could not open frame timing CSV: " + path);
  }
  out << "frame,block,elapsed_seconds,y,theta_deg,iterations,absence_flag\n";
  out << std::fixed << std::setprecision(9);
  for (const auto & row : frame_results) {
    out << row.frame << ","
        << row.block << ","
        << row.elapsed_s << ","
        << std::setprecision(6) << row.state.y << ","
        << row.state.theta_deg << ","
        << row.iterations << ","
        << (row.absence_flag ? 1 : 0) << "\n"
        << std::setprecision(9);
  }
}

} // namespace hl_tsa_cpp
