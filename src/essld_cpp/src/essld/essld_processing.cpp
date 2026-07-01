/**
 * ESSLD coarse-line extraction and dual multi-scale fusion refinement.
 *
 * Ported from tools/ESSLD/utils/processing.py. The reference module also
 * defines `_ray_cast_core_numba`, `_ray_cast_to_contour`, `_is_gourd_shape`
 * and `_find_optimal_p`, but none of them are called by
 * `get_coarse_line_from_mask`, `refine_horizon_stage3`, `demo.py` or
 * `demo_video.py` (verified by grep across the reference tool) -- they are
 * dead code in the reference implementation and are intentionally not
 * ported here.
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

#include <algorithm>
#include <cmath>

namespace essld_cpp
{

std::optional<std::pair<double, double>> robust_fit_line(
  const std::vector<cv::Point2f> & points,
  cv::DistanceTypes dist_type)
{
  if (points.size() < 2) {
    return std::nullopt;
  }
  cv::Vec4f line;
  cv::fitLine(points, line, dist_type, 0, 0.01, 0.01);
  double vx = line[0];
  const double vy = line[1];
  const double x0 = line[2];
  const double y0 = line[3];
  if (std::abs(vx) < 1e-5) {
    vx = 1e-5;
  }
  const double k = vy / vx;
  const double b = y0 - k * x0;
  return std::make_pair(k, b);
}

std::optional<std::pair<double, double>> weighted_line_fit(
  const std::vector<double> & x,
  const std::vector<double> & y,
  const std::vector<double> & weights)
{
  const size_t n = x.size();
  if (n < 2 || y.size() != n) {
    return std::nullopt;
  }
  const bool has_weights = !weights.empty();
  if (has_weights && weights.size() != n) {
    return std::nullopt;
  }

  double sw = 0.0, swx = 0.0, swy = 0.0, swxx = 0.0, swxy = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double w = has_weights ? weights[i] : 1.0;
    sw += w;
    swx += w * x[i];
    swy += w * y[i];
    swxx += w * x[i] * x[i];
    swxy += w * x[i] * y[i];
  }
  const double denom = sw * swxx - swx * swx;
  if (sw < 1e-9 || std::abs(denom) < 1e-9) {
    return std::nullopt;
  }
  const double k = (sw * swxy - swx * swy) / denom;
  const double b = (swy - k * swx) / sw;
  return std::make_pair(k, b);
}

cv::Mat non_max_suppression(const cv::Mat & magnitude, const cv::Mat & angle)
{
  CV_Assert(magnitude.type() == CV_64F && angle.type() == CV_64F);
  CV_Assert(magnitude.size() == angle.size());

  const int h = magnitude.rows;
  const int w = magnitude.cols;
  cv::Mat suppressed = cv::Mat::zeros(h, w, CV_64F);

  cv::Mat angle_deg = angle * (180.0 / CV_PI);
  for (int i = 0; i < h; ++i) {
    double * row = angle_deg.ptr<double>(i);
    for (int j = 0; j < w; ++j) {
      if (row[j] < 0.0) {
        row[j] += 180.0;
      }
    }
  }

  for (int i = 1; i < h - 1; ++i) {
    const double * a_row = angle_deg.ptr<double>(i);
    const double * m_row = magnitude.ptr<double>(i);
    const double * m_prev = magnitude.ptr<double>(i - 1);
    const double * m_next = magnitude.ptr<double>(i + 1);
    double * out_row = suppressed.ptr<double>(i);
    for (int j = 1; j < w - 1; ++j) {
      const double a = a_row[j];
      double q = 255.0, r = 255.0;
      if ((a >= 0.0 && a < 22.5) || (a >= 157.5 && a <= 180.0)) {
        q = m_row[j + 1];
        r = m_row[j - 1];
      } else if (a >= 22.5 && a < 67.5) {
        q = m_next[j - 1];
        r = m_prev[j + 1];
      } else if (a >= 67.5 && a < 112.5) {
        q = m_next[j];
        r = m_prev[j];
      } else if (a >= 112.5 && a < 157.5) {
        q = m_prev[j - 1];
        r = m_next[j + 1];
      }
      const double v = m_row[j];
      out_row[j] = (v >= q && v >= r) ? v : 0.0;
    }
  }
  return suppressed;
}

std::optional<std::pair<double, double>> get_coarse_line_from_mask(const cv::Mat & binary_mask)
{
  CV_Assert(binary_mask.type() == CV_8UC1);
  const int h = binary_mask.rows;
  const int w = binary_mask.cols;

  cv::Mat labels, stats, centroids;
  const int num_labels = cv::connectedComponentsWithStats(binary_mask, labels, stats, centroids, 8, CV_32S);
  if (num_labels < 2) {
    return std::nullopt;
  }

  int largest_label = 1;
  int max_area = 0;
  for (int i = 1; i < num_labels; ++i) {
    const int area = stats.at<int>(i, cv::CC_STAT_AREA);
    if (area > max_area) {
      max_area = area;
      largest_label = i;
    }
  }
  if (max_area < 100) {
    return std::nullopt;
  }

  cv::Mat clean_mask = cv::Mat::zeros(h, w, CV_8UC1);
  clean_mask.setTo(255, labels == largest_label);

  cv::Mat repaired_mask = cv::Mat::zeros(h, w, CV_8UC1);
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(clean_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  if (!contours.empty()) {
    const auto largest_it = std::max_element(
      contours.begin(), contours.end(),
      [](const std::vector<cv::Point> & a, const std::vector<cv::Point> & b) {
        return cv::contourArea(a) < cv::contourArea(b);
      });
    std::vector<cv::Point> hull;
    cv::convexHull(*largest_it, hull);
    cv::fillPoly(repaired_mask, std::vector<std::vector<cv::Point>>{hull}, cv::Scalar(255));
  } else {
    repaired_mask = clean_mask;
  }

  std::vector<cv::Point2f> horizon_points;
  for (int col = 0; col < w; col += 4) {
    int prev = static_cast<int>(repaired_mask.at<uchar>(0, col));
    int first_transition_row = -1;
    for (int row = 1; row < h; ++row) {
      const int cur = static_cast<int>(repaired_mask.at<uchar>(row, col));
      if (std::abs(cur - prev) > 100) {
        first_transition_row = row - 1;
        break;
      }
      prev = cur;
    }
    if (first_transition_row > 5 && first_transition_row < h - 5) {
      horizon_points.emplace_back(static_cast<float>(col), static_cast<float>(first_transition_row));
    }
  }

  if (horizon_points.size() < 10) {
    return std::nullopt;
  }

  const auto fit = robust_fit_line(horizon_points, cv::DIST_HUBER);
  if (!fit) {
    return std::nullopt;
  }
  const double k = fit->first;
  const double b = fit->second;

  std::vector<double> inlier_x, inlier_y;
  for (const auto & p : horizon_points) {
    const double pred_y = k * p.x + b;
    if (std::abs(p.y - pred_y) < 5.0) {
      inlier_x.push_back(p.x);
      inlier_y.push_back(p.y);
    }
  }

  if (inlier_x.size() > 10) {
    const auto final_fit = weighted_line_fit(inlier_x, inlier_y, {});
    if (final_fit) {
      const double k_final = final_fit->first;
      const double b_final = final_fit->second;
      const double angle = std::atan(k_final) * 180.0 / CV_PI;
      const double y_mid = k_final * (w / 2.0) + b_final;
      return std::make_pair(y_mid, angle);
    }
  }

  const double angle = std::atan(k) * 180.0 / CV_PI;
  const double y_mid = k * (w / 2.0) + b;
  return std::make_pair(y_mid, angle);
}

std::optional<std::pair<double, double>> run_dual_fusion_pipeline(
  const cv::Mat & roi_image,
  const ESSLDConfig & config)
{
  if (roi_image.empty()) {
    return std::nullopt;
  }
  if (config.median_filter_sizes.size() != config.canny_fusion_weights.size() ||
    config.confidence_map_sigmas.size() != config.confidence_map_weights.size())
  {
    throw std::runtime_error(
      "fusion.median_filter_sizes/canny_fusion_weights and "
      "fusion.confidence_map_sigmas/confidence_map_weights must have matching lengths");
  }

  const int h = roi_image.rows;
  const int w = roi_image.cols;

  cv::Mat gray;
  cv::cvtColor(roi_image, gray, cv::COLOR_BGR2GRAY);
  cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(4.0, cv::Size(8, 8));
  clahe->apply(gray, gray);

  cv::Mat edge_accum = cv::Mat::zeros(h, w, CV_32F);
  for (size_t s = 0; s < config.median_filter_sizes.size(); ++s) {
    const int ksize = static_cast<int>(config.median_filter_sizes[s]);
    const double weight = config.canny_fusion_weights[s];
    cv::Mat blurred;
    cv::medianBlur(gray, blurred, ksize);
    cv::Mat edge;
    cv::Canny(blurred, edge, 50, 150);
    cv::Mat edge_f;
    edge.convertTo(edge_f, CV_32F, weight / 255.0);
    edge_accum += edge_f;
  }
  cv::Mat edge_norm;
  cv::normalize(edge_accum, edge_norm, 0, 1, cv::NORM_MINMAX);

  const int center_y = h / 2;
  cv::Mat conf_accum = cv::Mat::zeros(h, w, CV_32F);
  for (size_t s = 0; s < config.confidence_map_sigmas.size(); ++s) {
    const double sigma = config.confidence_map_sigmas[s];
    const double weight = config.confidence_map_weights[s];
    for (int y = 0; y < h; ++y) {
      const double dist = std::abs(y - center_y);
      const double conf = std::exp(-0.5 * (dist / sigma) * (dist / sigma));
      const float * edge_row = edge_norm.ptr<float>(y);
      float * conf_row = conf_accum.ptr<float>(y);
      for (int x = 0; x < w; ++x) {
        conf_row[x] += static_cast<float>(edge_row[x] * conf * weight);
      }
    }
  }

  cv::Mat sobel_x, sobel_y;
  cv::Sobel(conf_accum, sobel_x, CV_64F, 1, 0, 5);
  cv::Sobel(conf_accum, sobel_y, CV_64F, 0, 1, 5);
  cv::Mat angle(h, w, CV_64F);
  for (int y = 0; y < h; ++y) {
    const double * sx = sobel_x.ptr<double>(y);
    const double * sy = sobel_y.ptr<double>(y);
    double * a = angle.ptr<double>(y);
    for (int x = 0; x < w; ++x) {
      a[x] = std::atan2(sy[x], sx[x]);
    }
  }

  cv::Mat conf_accum_d;
  conf_accum.convertTo(conf_accum_d, CV_64F);
  const cv::Mat nms_map = non_max_suppression(conf_accum_d, angle);

  cv::Mat binary_map(h, w, CV_8UC1);
  for (int y = 0; y < h; ++y) {
    const double * nms_row = nms_map.ptr<double>(y);
    uchar * bin_row = binary_map.ptr<uchar>(y);
    for (int x = 0; x < w; ++x) {
      bin_row[x] = (nms_row[x] * 255.0 > config.fused_map_final_threshold) ? 255 : 0;
    }
  }

  cv::Mat cc_labels, cc_stats, cc_centroids;
  const int num_labels =
    cv::connectedComponentsWithStats(binary_map, cc_labels, cc_stats, cc_centroids, 4, CV_32S);
  if (num_labels > 1) {
    cv::Mat cleaned = cv::Mat::zeros(h, w, CV_8UC1);
    for (int i = 1; i < num_labels; ++i) {
      if (cc_stats.at<int>(i, cv::CC_STAT_AREA) >= 50) {
        cleaned.setTo(255, cc_labels == i);
      }
    }
    binary_map = cleaned;
  }

  std::vector<cv::Vec4i> lines;
  cv::HoughLinesP(
    binary_map, lines, 1, CV_PI / 180.0,
    static_cast<int>(config.hough_threshold),
    static_cast<double>(config.hough_min_line_length),
    static_cast<double>(config.hough_max_line_gap));
  if (!lines.empty()) {
    cv::Mat line_mask = cv::Mat::zeros(h, w, CV_8UC1);
    for (const auto & l : lines) {
      cv::line(line_mask, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255), 1);
    }
    binary_map.setTo(0, line_mask == 0);
  }

  std::vector<double> xs, ys, weights;
  for (int y = 0; y < h; ++y) {
    const uchar * bin_row = binary_map.ptr<uchar>(y);
    const float * conf_row = conf_accum.ptr<float>(y);
    for (int x = 0; x < w; ++x) {
      if (bin_row[x] > 0) {
        xs.push_back(static_cast<double>(x));
        ys.push_back(static_cast<double>(y));
        weights.push_back(static_cast<double>(conf_row[x]));
      }
    }
  }

  if (xs.size() < 10) {
    return std::nullopt;
  }

  double weight_sum = 0.0;
  for (const double wt : weights) {
    weight_sum += wt;
  }

  std::optional<std::pair<double, double>> fit;
  if (weight_sum > 1e-6) {
    fit = weighted_line_fit(xs, ys, weights);
  } else {
    std::vector<cv::Point2f> pts;
    pts.reserve(xs.size());
    for (size_t i = 0; i < xs.size(); ++i) {
      pts.emplace_back(static_cast<float>(xs[i]), static_cast<float>(ys[i]));
    }
    fit = robust_fit_line(pts, cv::DIST_L12);
  }
  if (!fit) {
    return std::nullopt;
  }

  const double k = fit->first;
  const double b = fit->second;
  const double angle_deg = std::atan(k) * 180.0 / CV_PI;
  const double y_mid = k * (w / 2.0) + b;
  return std::make_pair(y_mid, angle_deg);
}

int calculate_dynamic_padding(double grad_score)
{
  double norm_score = (grad_score - 25.0) / (200.0 - 25.0);
  norm_score = std::clamp(norm_score, 0.0, 1.0);
  const double padding = 40.0 - norm_score * (40.0 - 20.0);
  return static_cast<int>(padding);
}

RoiResult create_roi(
  const cv::Mat & frame,
  double y_coarse,
  double angle_coarse,
  int padding)
{
  RoiResult result;
  const int w = frame.cols;

  const double angle_clamped = std::clamp(angle_coarse, -89.9, 89.9);
  const double angle_rad = angle_clamped * CV_PI / 180.0;
  const double slope = std::tan(angle_rad);
  const double intercept = y_coarse - slope * (w / 2.0);
  const double y_pad = padding / (std::abs(std::cos(angle_rad)) + 1e-6);

  const int roi_h = static_cast<int>(2 * padding);
  if (roi_h <= 0) {
    return result;
  }

  const cv::Point2f src_pts[4] = {
    cv::Point2f(0.0f, static_cast<float>(intercept - y_pad)),
    cv::Point2f(static_cast<float>(w - 1), static_cast<float>(slope * (w - 1) + intercept - y_pad)),
    cv::Point2f(static_cast<float>(w - 1), static_cast<float>(slope * (w - 1) + intercept + y_pad)),
    cv::Point2f(0.0f, static_cast<float>(intercept + y_pad))};
  const cv::Point2f dst_pts[4] = {
    cv::Point2f(0.0f, 0.0f),
    cv::Point2f(static_cast<float>(w - 1), 0.0f),
    cv::Point2f(static_cast<float>(w - 1), static_cast<float>(roi_h - 1)),
    cv::Point2f(0.0f, static_cast<float>(roi_h - 1))};

  result.m = cv::getPerspectiveTransform(src_pts, dst_pts);
  result.m_inv = cv::getPerspectiveTransform(dst_pts, src_pts);
  cv::warpPerspective(frame, result.roi, result.m, cv::Size(w, roi_h));
  result.roi_height = roi_h;
  result.width = w;
  return result;
}

std::optional<std::pair<double, double>> refine_horizon_stage3(
  const cv::Mat & frame,
  double coarse_y,
  double coarse_angle_deg,
  const ESSLDConfig & config)
{
  const int padding = calculate_dynamic_padding(config.grad_score);
  const RoiResult roi = create_roi(frame, coarse_y, coarse_angle_deg, padding);
  if (roi.roi.empty()) {
    return std::nullopt;
  }

  const auto refined_local = run_dual_fusion_pipeline(roi.roi, config);
  if (!refined_local) {
    return std::nullopt;
  }

  const double local_y = refined_local->first;
  const double local_angle = refined_local->second;
  const double final_angle = coarse_angle_deg + local_angle;

  const std::vector<cv::Point2f> pt_roi{
    cv::Point2f(static_cast<float>(roi.width / 2.0), static_cast<float>(local_y))};
  std::vector<cv::Point2f> pt_global;
  cv::perspectiveTransform(pt_roi, pt_global, roi.m_inv);
  const double final_y = pt_global[0].y;

  return std::make_pair(final_y, final_angle);
}

cv::Mat draw_horizon(const cv::Mat & frame, double y, double theta_deg)
{
  cv::Mat out = frame.clone();
  const int width = out.cols;
  const double slope = std::tan(theta_deg * CV_PI / 180.0);
  const double intercept = y - slope * (width / 2.0);
  const double y1 = std::clamp(intercept, -10000.0, 10000.0);
  const double y2 = std::clamp(slope * width + intercept, -10000.0, 10000.0);
  cv::line(
    out, cv::Point(0, static_cast<int>(y1)), cv::Point(width, static_cast<int>(y2)),
    cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
  return out;
}

} // namespace essld_cpp
