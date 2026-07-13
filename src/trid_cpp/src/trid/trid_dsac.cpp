/**
 * TRiD-Horizon per-column heatmap decoding and differentiable-RANSAC
 * (DSAC) line fit, ported from models/heads.py's `heatmap_to_points` and
 * `DSACLineFit`. These are the parts of TRiDHorizon that TensorRT cannot
 * run (see trid_cpp.hpp's TRTEngine docstring and export_onnx.py), so they
 * execute here on the engine's raw heat_logits/confidence_logits outputs.
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
#include <numeric>

namespace trid_cpp
{

ColumnPoints heatmap_to_points(
  const float * heat_logits, const float * confidence_logits, int height, int width)
{
  ColumnPoints points;
  points.x.resize(width);
  points.y.resize(width, 0.0);
  points.confidence.resize(width);

  std::vector<double> y_grid(height);
  for (int i = 0; i < height; ++i) {
    y_grid[i] = -1.0 + (2.0 * i) / std::max(height - 1, 1);
  }

  for (int w = 0; w < width; ++w) {
    points.x[w] = -1.0 + (2.0 * w) / std::max(width - 1, 1);

    // Softmax over the height dimension of column w, then expectation
    // against y_grid -- matches models.heads.heatmap_to_points exactly
    // (torch.softmax(logits, dim=1) followed by a weighted sum).
    double max_logit = -std::numeric_limits<double>::infinity();
    for (int h = 0; h < height; ++h) {
      max_logit = std::max(max_logit, static_cast<double>(heat_logits[h * width + w]));
    }
    double sum_exp = 0.0;
    double weighted_sum = 0.0;
    for (int h = 0; h < height; ++h) {
      const double e = std::exp(static_cast<double>(heat_logits[h * width + w]) - max_logit);
      sum_exp += e;
      weighted_sum += e * y_grid[h];
    }
    points.y[w] = weighted_sum / std::max(sum_exp, 1e-12);

    points.confidence[w] = 1.0 / (1.0 + std::exp(-static_cast<double>(confidence_logits[w])));
  }

  return points;
}

std::pair<double, double> dsac_line_fit(
  const ColumnPoints & points, const DSACConfig & config, std::mt19937 & rng)
{
  const int width = static_cast<int>(points.x.size());
  const double temperature = std::max(config.temperature, 1e-6);

  std::vector<double> probs(points.confidence.size());
  double confidence_sum = 0.0;
  for (size_t i = 0; i < probs.size(); ++i) {
    probs[i] = std::max(points.confidence[i], 1e-6);
    confidence_sum += points.confidence[i];
  }
  // torch.multinomial(probs, hypotheses, replacement=True) samples
  // `hypotheses` iid columns from Categorical(probs); std::discrete_
  // distribution implements the same semantics from an unnormalized weight
  // vector. The RNG stream differs from PyTorch's (see trid_cpp.hpp's
  // dsac_line_fit docstring and README.md), so results are not bit-exact,
  // but statistically equivalent -- this differentiable-RANSAC-style head
  // consensus-averages many samples specifically so no single sample's
  // identity matters.
  std::discrete_distribution<int> column_dist(probs.begin(), probs.end());

  const int hypotheses = config.hypotheses;
  std::vector<int> idx1(hypotheses), idx2(hypotheses);
  for (int h = 0; h < hypotheses; ++h) {
    idx1[h] = column_dist(rng);
    idx2[h] = column_dist(rng);
  }

  const int offset = std::max(2, width / 4);
  std::vector<double> m(hypotheses), b(hypotheses);
  for (int h = 0; h < hypotheses; ++h) {
    double x1 = points.x[idx1[h]], y1 = points.y[idx1[h]];
    double x2 = points.x[idx2[h]], y2 = points.y[idx2[h]];
    if (std::abs(x2 - x1) < config.min_column_delta) {
      idx2[h] = (idx1[h] + offset) % width;
      x2 = points.x[idx2[h]];
      y2 = points.y[idx2[h]];
    }
    double dx = x2 - x1;
    const double denom = (dx < 0.0) ? std::min(dx, -1e-6) : std::max(dx, 1e-6);
    m[h] = (y2 - y1) / denom;
    b[h] = y1 - m[h] * x1;
  }

  std::vector<double> scores(hypotheses);
  for (int h = 0; h < hypotheses; ++h) {
    double weighted_inliers = 0.0;
    for (int w = 0; w < width; ++w) {
      const double residual = std::abs(points.y[w] - (m[h] * points.x[w] + b[h]));
      const double soft_inlier = 1.0 / (1.0 + std::exp(-(config.inlier_threshold - residual) / temperature));
      weighted_inliers += soft_inlier * points.confidence[w];
    }
    const double consensus = weighted_inliers / std::max(confidence_sum, 1e-6);
    const double conf_score = points.confidence[idx1[h]] * points.confidence[idx2[h]];
    const double prior = 0.1 * std::exp(-0.5 * (m[h] / 0.75) * (m[h] / 0.75));
    scores[h] = consensus + 0.25 * conf_score + prior;
  }

  const double max_score = *std::max_element(scores.begin(), scores.end());
  std::vector<double> hyp_probs(hypotheses);
  double hyp_sum = 0.0;
  for (int h = 0; h < hypotheses; ++h) {
    hyp_probs[h] = std::exp((scores[h] - max_score) / temperature);
    hyp_sum += hyp_probs[h];
  }
  double m_final = 0.0, b_final = 0.0;
  for (int h = 0; h < hypotheses; ++h) {
    const double weight = hyp_probs[h] / hyp_sum;
    m_final += weight * m[h];
    b_final += weight * b[h];
  }

  return {m_final, b_final};
}

} // namespace trid_cpp
