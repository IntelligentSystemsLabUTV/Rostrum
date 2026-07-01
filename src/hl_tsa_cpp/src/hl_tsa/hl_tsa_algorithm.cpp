/**
 * HL-TSA CPU algorithm implementation.
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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <set>

namespace hl_tsa_cpp
{
namespace
{

constexpr double kPi = 3.14159265358979323846;

double deg2rad(double degrees)
{
  return degrees * kPi / 180.0;
}

double rad2deg(double radians)
{
  return radians * 180.0 / kPi;
}

double tand(double degrees)
{
  return std::tan(deg2rad(degrees));
}

double cosd(double degrees)
{
  const double value = std::cos(deg2rad(degrees));
  if (std::abs(value) < 1e-8) {
    return value >= 0.0 ? 1e-8 : -1e-8;
  }
  return value;
}

double elapsed_seconds(const std::chrono::steady_clock::time_point & start)
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

double vector_variance(const std::vector<double> & values)
{
  if (values.empty()) {
    return 0.0;
  }
  const double mean = std::accumulate(values.begin(), values.end(), 0.0) /
    static_cast<double>(values.size());
  double accum = 0.0;
  for (double value : values) {
    const double diff = value - mean;
    accum += diff * diff;
  }
  return accum / static_cast<double>(values.size());
}

double mean_tail_with_current(const std::vector<double> & values, double current)
{
  std::vector<double> recent;
  const size_t tail = std::min<size_t>(4, values.size());
  recent.reserve(tail + 1);
  recent.insert(recent.end(), values.end() - static_cast<std::ptrdiff_t>(tail), values.end());
  recent.push_back(current);
  return std::accumulate(recent.begin(), recent.end(), 0.0) / static_cast<double>(recent.size());
}

} // namespace

void ARModel::fit(const std::vector<double> & values, int differencing, double min_sd)
{
  differencing_ = differencing;
  min_sd_ = min_sd;
  order_ = 2;

  const std::vector<double> transformed = difference(values, differencing_);
  if (transformed.size() <= static_cast<size_t>(order_)) {
    variance_ = std::max(vector_variance(transformed), min_sd_ * min_sd_);
    coefficients_ = Eigen::Vector3d(0.0, 1.0, 0.0);
    return;
  }

  const int rows = static_cast<int>(transformed.size()) - order_;
  Eigen::MatrixXd design(rows, 3);
  Eigen::VectorXd target(rows);
  for (int i = 0; i < rows; ++i) {
    const int idx = i + order_;
    design(i, 0) = 1.0;
    design(i, 1) = transformed[idx - 1];
    design(i, 2) = transformed[idx - 2];
    target(i) = transformed[idx];
  }

  coefficients_ = design.colPivHouseholderQr().solve(target);
  const Eigen::VectorXd residual = target - design * coefficients_;
  std::vector<double> residual_values;
  residual_values.reserve(static_cast<size_t>(residual.size()));
  for (int i = 0; i < residual.size(); ++i) {
    residual_values.push_back(residual(i));
  }
  variance_ = std::max(vector_variance(residual_values), min_sd_ * min_sd_);
}

std::pair<double, double> ARModel::forecast(const std::vector<double> & values) const
{
  if (values.empty()) {
    return {0.0, variance_};
  }

  const std::vector<double> transformed = difference(values, differencing_);
  double transformed_next = 0.0;
  if (transformed.empty()) {
    transformed_next = 0.0;
  } else if (transformed.size() == 1) {
    transformed_next = transformed.back();
  } else {
    transformed_next = coefficients_.dot(Eigen::Vector3d(1.0, transformed[transformed.size() - 1],
      transformed[transformed.size() - 2]));
  }

  double forecast_value = transformed_next;
  if (differencing_ == 1) {
    forecast_value = values.back() + transformed_next;
  } else if (differencing_ >= 2) {
    if (values.size() >= 2) {
      forecast_value = 2.0 * values[values.size() - 1] - values[values.size() - 2] + transformed_next;
    } else {
      forecast_value = values.back();
    }
  }

  const double local_var = local_variance(transformed);
  return {forecast_value, std::max({variance_, local_var, min_sd_ * min_sd_})};
}

std::vector<double> ARModel::difference(const std::vector<double> & values, int order)
{
  std::vector<double> out = values;
  for (int d = 0; d < order; ++d) {
    if (out.size() < 2) {
      out.clear();
      break;
    }
    std::vector<double> diff;
    diff.reserve(out.size() - 1);
    for (size_t i = 1; i < out.size(); ++i) {
      diff.push_back(out[i] - out[i - 1]);
    }
    out = std::move(diff);
  }
  return out;
}

double ARModel::local_variance(const std::vector<double> & transformed) const
{
  if (transformed.size() < 3) {
    return min_sd_ * min_sd_;
  }
  const size_t start = transformed.size() > 12 ? transformed.size() - 12 : 0;
  return vector_variance(std::vector<double>(transformed.begin() + static_cast<std::ptrdiff_t>(start),
    transformed.end()));
}

HLTSADetector::HLTSADetector(HLTSAConfig config)
: config_(config)
{}

void HLTSADetector::reset()
{
  states_.clear();
  errors_.clear();
  model_ready_ = false;
  absent_ = false;
  last_present_index_ = 0;
  processed_frames_ = 0;
  width_ = 0;
  height_ = 0;
  min_roi_height_ = 0;
  model_fit_seconds_ = 0.0;
}

FrameResult HLTSADetector::process_frame(const cv::Mat & frame)
{
  if (frame.empty()) {
    throw std::runtime_error("HLTSADetector::process_frame: empty frame");
  }
  if (width_ == 0 || height_ == 0) {
    width_ = frame.cols;
    height_ = frame.rows;
    min_roi_height_ = static_cast<int>(std::round(height_ * config_.min_roi_height_fraction));
  }

  const auto start = std::chrono::steady_clock::now();
  FrameResult result;
  if (processed_frames_ < config_.listener_frames) {
    result = listener_step(frame, start);
    if (processed_frames_ == config_.listener_frames && !model_ready_) {
      fit_models();
    }
  } else if (absent_) {
    result = presence_detector(frame, start);
  } else {
    result = main_step(frame, start);
  }

  result.frame = processed_frames_;
  return result;
}

FrameResult HLTSADetector::listener_step(
  const cv::Mat & frame,
  const std::chrono::steady_clock::time_point & start)
{
  HorizonState state;
  if (states_.empty()) {
    state = hlda(frame);
  } else {
    auto [roi, limits] = roi_rect(frame, states_.back(), min_roi_height_);
    (void)limits;
    const HorizonState local_state = hlda(roi);
    state = {local_state.y + static_cast<double>(limits.first), local_state.theta_deg};
  }

  const double error = get_error(state, frame);
  states_.push_back(state);
  errors_.push_back(error);
  processed_frames_++;

  FrameResult result;
  result.block = "listener";
  result.state = state;
  result.error = error;
  result.elapsed_s = elapsed_seconds(start);
  return result;
}

FrameResult HLTSADetector::main_step(
  const cv::Mat & frame,
  const std::chrono::steady_clock::time_point & start)
{
  std::vector<double> y_values;
  std::vector<double> theta_values;
  const size_t history = std::min<size_t>(static_cast<size_t>(config_.listener_frames), states_.size());
  y_values.reserve(history);
  theta_values.reserve(history);
  for (size_t i = states_.size() - history; i < states_.size(); ++i) {
    y_values.push_back(states_[i].y);
    theta_values.push_back(states_[i].theta_deg);
  }

  const auto [yf, yf_var] = y_model_.forecast(y_values);
  const auto [tf, tf_var] = theta_model_.forecast(theta_values);
  const double yf_sd = std::sqrt(yf_var);
  const double tf_sd = std::sqrt(tf_var);

  cv::Mat roi;
  cv::Mat mask;
  int delta_height = 0;
  const bool roi_ok = roi_tsm(frame, {yf, tf}, yf_sd, tf_sd, roi, mask, delta_height);

  FrameResult result;
  if (roi_ok) {
    HorizonState local_state = hlda(roi);
    HorizonState state = {
      yf - static_cast<double>(delta_height) + local_state.y / cosd(tf),
      tf + local_state.theta_deg};
    double error = get_error(state, frame);

    int iterations = 0;
    int min_roi_extra = 0;
    while (iterations < config_.max_iterations && needs_control_loop(local_state, error, frame.rows)) {
      iterations++;
      auto [rect_roi, limits] = roi_rect(frame, states_.back(), min_roi_extra);
      mask = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
      mask.rowRange(limits.first, limits.second + 1).setTo(1);
      local_state = hlda(rect_roi);
      state = {local_state.y + static_cast<double>(limits.first), local_state.theta_deg};
      error = get_error(state, frame);
      min_roi_extra += min_roi_height_;
    }

    states_.push_back(state);
    errors_.push_back(error);
    processed_frames_++;

    result.block = "main";
    result.state = state;
    result.error = error;
    result.iterations = iterations;
    result.absence_flag = false;
    result.roi_mask = mask;
  } else {
    last_present_index_ = static_cast<int>(states_.size()) - 1;
    HorizonState state;
    mask = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
    if (states_.back().y > static_cast<double>(frame.rows) / 2.0) {
      state = {static_cast<double>(frame.rows - 1), 0.0};
      const int top = std::max(0, frame.rows - 2 * min_roi_height_);
      mask.rowRange(top, frame.rows).setTo(1);
    } else {
      state = {0.0, 0.0};
      const int bottom = std::min(frame.rows, 2 * min_roi_height_);
      mask.rowRange(0, bottom).setTo(1);
    }

    states_.push_back(state);
    errors_.push_back(get_error(state, frame));
    absent_ = true;
    processed_frames_++;

    result.block = "absence_detector";
    result.state = state;
    result.error = errors_.back();
    result.absence_flag = true;
    result.roi_mask = mask;
  }

  result.elapsed_s = elapsed_seconds(start);
  return result;
}

FrameResult HLTSADetector::presence_detector(
  const cv::Mat & frame,
  const std::chrono::steady_clock::time_point & start)
{
  int top = 0;
  int bottom = 0;
  if (states_.back().y > 0.0) {
    top = std::max(0, frame.rows - 2 * min_roi_height_);
    bottom = frame.rows - 1;
  } else {
    top = 0;
    bottom = std::min(frame.rows - 1, 2 * min_roi_height_);
  }

  cv::Mat roi = frame.rowRange(top, bottom + 1);
  const HorizonState local_state = hlda(roi);
  HorizonState state = {local_state.y + static_cast<double>(top), local_state.theta_deg};
  const double error = get_error(state, frame);

  const double reference_error =
    last_present_index_ >= 0 && last_present_index_ < static_cast<int>(errors_.size()) ?
    errors_[static_cast<size_t>(last_present_index_)] : errors_.back();
  const double denominator = std::abs(reference_error) > 1e-12 ? reference_error : 1.0;
  absent_ = std::abs((error - reference_error) / denominator) > 0.05;
  if (absent_) {
    state = states_.back();
  }

  cv::Mat mask = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
  mask.rowRange(top, bottom + 1).setTo(1);

  states_.push_back(state);
  errors_.push_back(error);
  processed_frames_++;

  FrameResult result;
  result.block = "presence_detector";
  result.state = state;
  result.error = error;
  result.absence_flag = absent_;
  result.roi_mask = mask;
  result.elapsed_s = elapsed_seconds(start);
  return result;
}

void HLTSADetector::fit_models()
{
  const auto start = std::chrono::steady_clock::now();
  std::vector<double> y_values;
  std::vector<double> theta_values;
  const size_t history = std::min<size_t>(static_cast<size_t>(config_.listener_frames), states_.size());
  y_values.reserve(history);
  theta_values.reserve(history);
  for (size_t i = states_.size() - history; i < states_.size(); ++i) {
    y_values.push_back(states_[i].y);
    theta_values.push_back(states_[i].theta_deg);
  }
  y_model_.fit(y_values, 2, config_.min_y_sd);
  theta_model_.fit(theta_values, 0, config_.min_theta_sd);
  model_ready_ = true;
  model_fit_seconds_ += elapsed_seconds(start);
}

bool HLTSADetector::needs_control_loop(
  const HorizonState & local_state,
  double error,
  int frame_height) const
{
  const double recent_mean = mean_tail_with_current(errors_, error);
  const double denominator = std::abs(recent_mean) > 1e-12 ? recent_mean : 1.0;
  const bool error_jump = std::abs((error - recent_mean) / denominator) > 0.15;
  const bool out_of_bounds = local_state.y > 3.0 * static_cast<double>(frame_height) || local_state.y < 0.0;
  return error_jump || out_of_bounds;
}

HorizonState HLTSADetector::hlda(const cv::Mat & frame) const
{
  const std::vector<HoughCandidate> candidates = hough_candidates(frame);
  if (candidates.empty()) {
    return {static_cast<double>(frame.rows) / 2.0, 0.0};
  }

  for (const auto & candidate : candidates) {
    if (std::abs(candidate.theta_deg) <= config_.max_abs_horizon_angle_deg) {
      return {candidate.y, candidate.theta_deg};
    }
  }
  return {candidates.front().y, candidates.front().theta_deg};
}

std::vector<HLTSADetector::HoughCandidate> HLTSADetector::hough_candidates(const cv::Mat & frame) const
{
  cv::Mat gray;
  if (frame.channels() == 1) {
    gray = frame;
  } else {
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  }

  cv::Mat edges;
  cv::Canny(gray, edges, config_.canny_low, config_.canny_high, 3, true);
  edges = area_open(edges);

  std::vector<cv::Vec2f> lines;
  cv::HoughLines(edges, lines, 1.0, deg2rad(config_.hough_theta_resolution_deg), config_.hough_threshold);

  std::vector<HoughCandidate> candidates;
  std::set<std::pair<int, int>> seen;
  candidates.reserve(lines.size());
  for (size_t i = 0; i < lines.size(); ++i) {
    const double rho = lines[i][0];
    const double theta = lines[i][1];
    if (std::abs(std::sin(theta)) < 1e-8) {
      continue;
    }
    const double y_center = (rho - (static_cast<double>(frame.cols) / 2.0) * std::cos(theta)) / std::sin(theta);
    const double angle = rad2deg(std::atan2(-std::cos(theta), std::sin(theta)));
    if (y_center < -frame.rows || y_center > 2.0 * frame.rows) {
      continue;
    }
    const std::pair<int, int> key{
      static_cast<int>(std::round(y_center)),
      static_cast<int>(std::round(angle * 4.0))};
    if (seen.find(key) != seen.end()) {
      continue;
    }
    seen.insert(key);
    candidates.push_back({y_center, angle, static_cast<int>(i)});
  }
  return candidates;
}

cv::Mat HLTSADetector::area_open(const cv::Mat & edges) const
{
  const int edge_count = cv::countNonZero(edges);
  const int min_area = static_cast<int>(std::round(static_cast<double>(edge_count) * config_.area_open_fraction));
  if (edge_count == 0 || min_area <= 1) {
    return edges;
  }

  cv::Mat labels;
  cv::Mat stats;
  cv::Mat centroids;
  const int count = cv::connectedComponentsWithStats(edges > 0, labels, stats, centroids, 8);
  cv::Mat opened = cv::Mat::zeros(edges.rows, edges.cols, CV_8UC1);
  for (int label = 1; label < count; ++label) {
    if (stats.at<int>(label, cv::CC_STAT_AREA) >= min_area) {
      opened.setTo(255, labels == label);
    }
  }
  return cv::countNonZero(opened) > 0 ? opened : edges;
}

double HLTSADetector::get_error(const HorizonState & state, const cv::Mat & frame) const
{
  cv::Mat gray;
  if (frame.channels() == 1) {
    gray = frame;
  } else {
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  }

  const double delta_h = config_.error_band_fraction * static_cast<double>(frame.rows);
  auto [sky_mask, sea_mask] = find_indices_of_regions(state, frame.cols, frame.rows, delta_h);
  const int sky_count = cv::countNonZero(sky_mask);
  const int sea_count = cv::countNonZero(sea_mask);
  if (sky_count == 0 || sea_count == 0) {
    return 0.0;
  }

  std::array<double, 255> sky_hist{};
  std::array<double, 255> sea_hist{};
  for (int y = 0; y < gray.rows; ++y) {
    const auto * gray_row = gray.ptr<unsigned char>(y);
    const auto * sky_row = sky_mask.ptr<unsigned char>(y);
    const auto * sea_row = sea_mask.ptr<unsigned char>(y);
    for (int x = 0; x < gray.cols; ++x) {
      const int value = static_cast<int>(gray_row[x]);
      if (value >= 255) {
        continue;
      }
      if (sky_row[x] != 0) {
        sky_hist[static_cast<size_t>(value)] += 1.0;
      } else if (sea_row[x] != 0) {
        sea_hist[static_cast<size_t>(value)] += 1.0;
      }
    }
  }

  double mse = 0.0;
  for (size_t i = 0; i < sky_hist.size(); ++i) {
    const double sky_prob = sky_hist[i] / static_cast<double>(sky_count);
    const double sea_prob = sea_hist[i] / static_cast<double>(sea_count);
    const double diff = sky_prob - sea_prob;
    mse += diff * diff;
  }
  mse /= static_cast<double>(sky_hist.size());
  const double err = -std::sqrt(mse);
  return std::isfinite(err) ? err : 0.0;
}

std::pair<cv::Mat, cv::Mat> HLTSADetector::find_indices_of_regions(
  const HorizonState & state,
  int width,
  int height,
  double delta_h) const
{
  cv::Mat sky = cv::Mat::zeros(height, width, CV_8UC1);
  cv::Mat sea = cv::Mat::zeros(height, width, CV_8UC1);
  const double slope = tand(state.theta_deg);
  const double c = state.y - slope * static_cast<double>(width) / 2.0;
  for (int row = 0; row < height; ++row) {
    const double y_coord = static_cast<double>(row + 1);
    auto * sky_row = sky.ptr<unsigned char>(row);
    auto * sea_row = sea.ptr<unsigned char>(row);
    for (int col = 0; col < width; ++col) {
      const double x_coord = static_cast<double>(col + 1);
      const double reference = slope * x_coord + c;
      if (y_coord < reference && y_coord > reference - delta_h) {
        sky_row[col] = 255;
      } else if (y_coord > reference && y_coord < reference + delta_h) {
        sea_row[col] = 255;
      }
    }
  }
  return {sky, sea};
}

std::pair<cv::Mat, std::pair<int, int>> HLTSADetector::roi_rect(
  const cv::Mat & frame,
  const HorizonState & state,
  int min_height) const
{
  const double slope = tand(state.theta_deg);
  double min_y = std::numeric_limits<double>::max();
  double max_y = std::numeric_limits<double>::lowest();
  for (int x = 1; x <= frame.cols; ++x) {
    const double y = std::round(slope * static_cast<double>(x) + state.y -
      slope * static_cast<double>(frame.cols) / 2.0);
    min_y = std::min(min_y, y);
    max_y = std::max(max_y, y);
  }

  int top = static_cast<int>(std::floor(min_y - static_cast<double>(min_height)));
  int bottom = static_cast<int>(std::ceil(max_y + static_cast<double>(min_height)));
  top = std::clamp(top, 0, frame.rows - 1);
  bottom = std::clamp(bottom, 0, frame.rows - 1);
  if (bottom < top) {
    std::swap(top, bottom);
  }
  return {frame.rowRange(top, bottom + 1), {top, bottom}};
}

bool HLTSADetector::roi_tsm(
  const cv::Mat & frame,
  const HorizonState & state,
  double y_sd,
  double theta_sd,
  cv::Mat & roi,
  cv::Mat & mask,
  int & delta_height) const
{
  const double slope = tand(state.theta_deg);
  const double left = std::round(slope * 1.0 + state.y - slope * static_cast<double>(frame.cols) / 2.0);
  const double right = std::round(slope * static_cast<double>(frame.cols) + state.y -
    slope * static_cast<double>(frame.cols) / 2.0);
  const double angled_slope = tand(state.theta_deg + theta_sd);
  const double angled_right = angled_slope * static_cast<double>(frame.cols) + state.y -
    angled_slope * static_cast<double>(frame.cols) / 2.0;

  delta_height = std::max(1, static_cast<int>(std::round(config_.zscore *
    (y_sd + std::abs(angled_right - right)))));

  std::vector<cv::Point> polygon{
    cv::Point(0, static_cast<int>(left) - delta_height),
    cv::Point(0, static_cast<int>(left) + delta_height),
    cv::Point(frame.cols - 1, static_cast<int>(right) + delta_height),
    cv::Point(frame.cols - 1, static_cast<int>(right) - delta_height)};
  mask = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
  cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{polygon}, 1);

  if (static_cast<double>(cv::countNonZero(mask)) / static_cast<double>(mask.total()) <= 0.0025) {
    roi.release();
    mask.release();
    delta_height = 0;
    return false;
  }

  std::vector<int> col_counts(frame.cols, 0);
  int max_count = 0;
  for (int col = 0; col < frame.cols; ++col) {
    col_counts[static_cast<size_t>(col)] = cv::countNonZero(mask.col(col));
    max_count = std::max(max_count, col_counts[static_cast<size_t>(col)]);
  }
  if (max_count <= 0) {
    roi.release();
    mask.release();
    delta_height = 0;
    return false;
  }

  std::vector<int> keep_cols;
  for (int col = 0; col < frame.cols; ++col) {
    if (col_counts[static_cast<size_t>(col)] == max_count) {
      keep_cols.push_back(col);
    }
  }
  if (keep_cols.empty()) {
    roi.release();
    mask.release();
    delta_height = 0;
    return false;
  }

  cv::Mat trimmed_mask = cv::Mat::zeros(mask.rows, mask.cols, CV_8UC1);
  for (int col : keep_cols) {
    mask.col(col).copyTo(trimmed_mask.col(col));
  }
  mask = trimmed_mask;

  roi = cv::Mat(max_count, static_cast<int>(keep_cols.size()), frame.type());
  const size_t elem_size = frame.elemSize();
  for (size_t out_col = 0; out_col < keep_cols.size(); ++out_col) {
    const int src_col = keep_cols[out_col];
    int out_row = 0;
    for (int row = 0; row < frame.rows; ++row) {
      if (mask.at<unsigned char>(row, src_col) != 0) {
        std::memcpy(
          roi.ptr(out_row) + static_cast<int>(out_col) * elem_size,
          frame.ptr(row) + src_col * elem_size,
          elem_size);
        out_row++;
      }
    }
  }
  return true;
}

} // namespace hl_tsa_cpp
