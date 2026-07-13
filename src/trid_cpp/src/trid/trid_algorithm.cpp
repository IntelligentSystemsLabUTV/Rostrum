/**
 * TRiD-Horizon line geometry, dual multi-scale ROI refinement, and ROI
 * acceptance gate.
 *
 * `robust_fit_line`, `weighted_line_fit`, `non_max_suppression`,
 * `calculate_dynamic_padding`, `create_roi`, and `run_dual_fusion_pipeline`
 * are near-identical ports of tools/ESSLD/utils/processing.py's
 * corresponding functions (also ported in essld_cpp): the "dual multi-scale
 * fusion" ROI refinement stage is shared, byte-for-byte-equivalent Python
 * code across tools/ESSLD and tools/TRiD (verified by diff of
 * utils/processing.py's `_run_dual_fusion_pipeline` against
 * tools/TRiD/inference/postprocess.py's `dual_fusion_pipeline`). What is new
 * here relative to essld_cpp is `refine_existing` operating on a coarse
 * HorizonLine (rather than a segmentation mask) and `apply_bounded_roi`, the
 * ROI acceptance gate ported from tools/TRiD/inference/roi_gate.py --
 * essld_cpp has no equivalent gate; it always accepts the refined line when
 * refine_horizon_stage3 does not throw.
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

#include <algorithm>
#include <cmath>

namespace trid_cpp
{

double HorizonLine::theta_deg() const
{
  return std::atan(slope()) * 180.0 / CV_PI;
}

HorizonLine line_from_center_angle(double y_center, double theta_deg, int width, int height)
{
  const double slope = std::tan(theta_deg * CV_PI / 180.0);
  const double intercept = y_center - slope * (static_cast<double>(width) / 2.0);
  HorizonLine line;
  line.y_left = intercept;
  line.y_right = slope * static_cast<double>(width - 1) + intercept;
  line.width = width;
  line.height = height;
  return line;
}

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
  result.corners = {src_pts[0], src_pts[1], src_pts[2], src_pts[3]};
  return result;
}

FusionResult run_dual_fusion_pipeline(const cv::Mat & roi_image, const FusionConfig & config)
{
  FusionResult result;
  if (roi_image.empty()) {
    return result;
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

  // inference/postprocess.py calls this as
  // `cv2.connectedComponentsWithStats(binary, 4, cv2.CV_32S)`, apparently
  // requesting 4-connectivity. Verified locally (OpenCV 4.11.0, both
  // opencv-python and libopencv C++): that 3-positional-argument Python
  // call does NOT bind its second argument to `connectivity` -- it silently
  // falls back to the function's 8-connectivity default regardless of the
  // value passed (confirmed on a synthetic diagonal-pixel test case where
  // true 4-connectivity and 8-connectivity give different label counts;
  // the Python call gives the 8-connectivity answer either way, while the
  // equivalent C++ call with an explicit `4` gives the true 4-connectivity
  // answer). So the reference tool's *actual* runtime behavior here is
  // 8-connectivity, not the 4 its source appears to request; connectivity
  // is hardcoded to 8 below to match that actual behavior (confirmed to
  // bring candidate_count/candidate_span much closer to the Python
  // reference on real ROI data, from ~69/68 with true 4-connectivity to
  // ~494/362 with 8-connectivity, against a Python reference of 325/383).
  // tools/ESSLD/utils/processing.py's equivalent call (also ported as
  // literal 4-connectivity in essld_cpp) likely has this same latent
  // discrepancy; not fixed there as part of this change since it was out
  // of scope, but worth checking.
  cv::Mat cc_labels, cc_stats, cc_centroids;
  const int num_labels =
    cv::connectedComponentsWithStats(binary_map, cc_labels, cc_stats, cc_centroids, 8, CV_32S);
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

  // xs/ys/weights mirror postprocess.py's `pts_yx`/`pts` candidate set: xs
  // here is the column (x) coordinate, matching what roi_gate.py's
  // `candidate_span` (named "candidate_horizontal_span" in the CSV output)
  // measures over `points[:, 1]` (pts_yx's column index).
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

  result.candidate_count = static_cast<int>(xs.size());
  if (!xs.empty()) {
    const auto [min_it, max_it] = std::minmax_element(xs.begin(), xs.end());
    result.candidate_span = *max_it - *min_it;
  }

  if (xs.size() < 10) {
    return result;
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
    return result;
  }

  const double k = fit->first;
  const double b = fit->second;
  const double angle_deg = std::atan(k) * 180.0 / CV_PI;
  const double y_mid = k * (w / 2.0) + b;
  result.line = std::make_pair(y_mid, angle_deg);
  return result;
}

RefineResult refine_existing(
  const cv::Mat & frame, const HorizonLine & coarse, double grad_score, const FusionConfig & config)
{
  RefineResult result;
  const int padding = calculate_dynamic_padding(grad_score);
  const RoiResult roi = create_roi(frame, coarse.y_center(), coarse.theta_deg(), padding);
  result.padding = padding;
  result.roi_height = roi.roi_height;
  result.roi_pts = roi.corners;
  if (roi.roi.empty()) {
    return result;
  }

  const FusionResult fusion = run_dual_fusion_pipeline(roi.roi, config);
  result.candidate_count = fusion.candidate_count;
  result.candidate_span = fusion.candidate_span;
  if (!fusion.line) {
    return result;
  }

  const double local_y = fusion.line->first;
  const double local_angle = fusion.line->second;
  const double final_angle = coarse.theta_deg() + local_angle;

  const std::vector<cv::Point2f> pt_roi{
    cv::Point2f(static_cast<float>(roi.width / 2.0), static_cast<float>(local_y))};
  std::vector<cv::Point2f> pt_global;
  cv::perspectiveTransform(pt_roi, pt_global, roi.m_inv);
  const double final_y = pt_global[0].y;

  result.refined = line_from_center_angle(final_y, final_angle, coarse.width, coarse.height);
  return result;
}

namespace
{

double line_y(const HorizonLine & line, double x)
{
  return line.slope() * x + line.intercept();
}

void roi_top_bottom(
  const std::array<cv::Point2f, 4> & roi_pts,
  const std::vector<double> & xs,
  std::vector<double> & top,
  std::vector<double> & bottom)
{
  const auto & lt = roi_pts[0];
  const auto & rt = roi_pts[1];
  const auto & rb = roi_pts[2];
  const auto & lb = roi_pts[3];
  top.resize(xs.size());
  bottom.resize(xs.size());
  const double denom = std::max(static_cast<double>(rt.x - lt.x), 1.0);
  for (size_t i = 0; i < xs.size(); ++i) {
    const double alpha = (xs[i] - lt.x) / denom;
    const double t = lt.y + alpha * (rt.y - lt.y);
    const double b = lb.y + alpha * (rb.y - lb.y);
    top[i] = std::min(t, b);
    bottom[i] = std::max(t, b);
  }
}

double inside_fraction(
  const HorizonLine & line, const std::array<cv::Point2f, 4> & roi_pts, int samples = 128)
{
  std::vector<double> xs(samples), top, bottom;
  for (int i = 0; i < samples; ++i) {
    xs[i] = (static_cast<double>(line.width - 1) * i) / std::max(samples - 1, 1);
  }
  roi_top_bottom(roi_pts, xs, top, bottom);
  int inside = 0;
  for (int i = 0; i < samples; ++i) {
    const double y = line_y(line, xs[i]);
    if (y >= top[i] && y <= bottom[i]) {
      ++inside;
    }
  }
  return static_cast<double>(inside) / static_cast<double>(samples);
}

} // namespace

RoiGateResult apply_bounded_roi(
  const cv::Mat & frame,
  const HorizonLine & coarse,
  bool enable_roi,
  bool enable_gate,
  double grad_score,
  const FusionConfig & fusion_config,
  const RoiGateConfig & gate_config)
{
  RoiGateResult result;
  result.final = coarse;

  if (!enable_roi) {
    result.reason = "roi_disabled";
    return result;
  }

  const RefineResult refine = refine_existing(frame, coarse, grad_score, fusion_config);
  result.roi_pts = refine.roi_pts;
  result.candidate_count = refine.candidate_count;
  result.candidate_span = refine.candidate_span;

  if (!refine.refined) {
    result.reason = "invalid_refined_fit";
    result.inside_roi_fraction = 0.0;
    return result;
  }

  const HorizonLine & refined = *refine.refined;
  result.existing_refined = refined;
  const double inside = inside_fraction(refined, refine.roi_pts);
  const double center_corr = std::abs(refined.y_center() - coarse.y_center());
  const double endpoint_corr = std::max(
    std::abs(refined.y_left - coarse.y_left), std::abs(refined.y_right - coarse.y_right));
  const double angle_corr = std::abs(refined.theta_deg() - coarse.theta_deg());
  result.inside_roi_fraction = inside;
  result.center_correction = center_corr;
  result.endpoint_correction = endpoint_corr;
  result.angle_correction = angle_corr;

  std::string reason = "accepted";
  if (enable_gate) {
    if (inside < gate_config.min_inside_roi_fraction) {
      reason = "refined_outside_roi";
    } else if (center_corr > gate_config.max_center_shift_half_height_frac * static_cast<double>(refine.padding)) {
      reason = "center_shift_too_large";
    } else if (endpoint_corr > gate_config.max_endpoint_shift_roi_height_frac * static_cast<double>(refine.roi_height)) {
      reason = "endpoint_shift_too_large";
    } else if (angle_corr > gate_config.max_angle_change_deg) {
      reason = "angle_change_too_large";
    } else if (refine.candidate_count < gate_config.min_candidate_count) {
      reason = "insufficient_candidates";
    } else if (refine.candidate_span < gate_config.min_candidate_span_frac * static_cast<double>(coarse.width)) {
      reason = "candidate_span_too_small";
    }
  }

  const bool accept = (reason == "accepted") || !enable_gate;
  result.final = accept ? refined : coarse;
  result.accepted = accept;
  result.reason = enable_gate ? reason : "gate_disabled";
  return result;
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

} // namespace trid_cpp
