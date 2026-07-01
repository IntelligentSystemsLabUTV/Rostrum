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
#include <functional>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>

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

std::vector<double> difference(const std::vector<double> & values, int order)
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

// Inverts the D-th order differencing for a 1-step-ahead forecast, i.e. recovers a
// forecast of y[T+1] given a forecast of w[T+1] = Delta^D y[T+1] and the most recent
// raw (undifferenced) observations. Only D in {0,1,2} is needed by HL_Detect_TSA.m.
double undifference_forecast(const std::vector<double> & window, int d, double w_forecast)
{
  switch (d) {
    case 0:
      return w_forecast;
    case 1:
      return w_forecast + window.back();
    case 2:
      return w_forecast + 2.0 * window[window.size() - 1] - window[window.size() - 2];
    default:
      throw std::invalid_argument("undifference_forecast: unsupported differencing order");
  }
}

double sigmoid(double x)
{
  return 1.0 / (1.0 + std::exp(-x));
}

double logit(double p)
{
  p = std::clamp(p, 1e-6, 1.0 - 1e-6);
  return std::log(p / (1.0 - p));
}

// Caps alpha+beta strictly below 1 so the unconditional/backcast variance
// omega/(1-alpha-beta) stays finite and well-conditioned during optimization.
constexpr double kMaxGarchPersistence = 0.999;

// Maps an unconstrained optimization vector theta to the model's natural parameters.
// AR(2) coefficients are produced via a Durbin-Levinson step on tanh-mapped partial
// autocorrelations (r1, r2), which guarantees the causal/stationary AR(2) triangle
// for any real (u1, u2) -- the standard reparametrization used to keep ARMA fits
// stationary during unconstrained optimization (Monahan, 1984). GARCH persistence
// (alpha+beta) and its alpha/beta split are mapped through sigmoids so omega > 0,
// alpha >= 0, beta >= 0 and alpha+beta < 1 hold unconditionally.
void decode_params(
  const std::vector<double> & theta, bool has_constant,
  double & c, double & phi1, double & phi2, double & omega, double & alpha, double & beta)
{
  size_t idx = 0;
  c = has_constant ? theta[idx++] : 0.0;
  const double r1 = std::tanh(theta[idx++]);
  const double r2 = std::tanh(theta[idx++]);
  phi2 = r2;
  phi1 = r1 * (1.0 - r2);
  omega = std::exp(theta[idx++]);
  const double persistence = kMaxGarchPersistence * sigmoid(theta[idx++]);
  const double alpha_fraction = sigmoid(theta[idx++]);
  alpha = persistence * alpha_fraction;
  beta = persistence * (1.0 - alpha_fraction);
}

std::vector<double> encode_initial(
  bool has_constant, double c0, double phi1_0, double phi2_0,
  double omega0, double alpha0, double beta0)
{
  std::vector<double> theta;
  if (has_constant) {
    theta.push_back(c0);
  }
  const double r2 = std::clamp(phi2_0, -0.95, 0.95);
  const double r1 = std::clamp(std::abs(1.0 - r2) > 1e-6 ? phi1_0 / (1.0 - r2) : 0.0, -0.95, 0.95);
  theta.push_back(std::atanh(r1));
  theta.push_back(std::atanh(r2));
  theta.push_back(std::log(std::max(omega0, 1e-10)));
  const double persistence0 = std::clamp((alpha0 + beta0) / kMaxGarchPersistence, 1e-3, 1.0 - 1e-3);
  theta.push_back(logit(persistence0));
  const double fraction0 = std::clamp(alpha0 / std::max(alpha0 + beta0, 1e-10), 1e-3, 1.0 - 1e-3);
  theta.push_back(logit(fraction0));
  return theta;
}

struct GarchPathResult
{
  double log_likelihood = -std::numeric_limits<double>::infinity();
  double last_eps = 0.0;
  double last_sigma2 = 0.0;
  bool valid = false;
};

// Runs the ARMA(2) mean recursion and the GARCH(1,1) conditional-variance recursion
// over the (already differenced) series w, and accumulates the exact Gaussian
// conditional log-likelihood. The presample innovation and variance are seeded with
// a FIXED backcast_variance (the sample mean of squared conditional-least-squares
// residuals, computed once before optimization). MATLAB's Econometrics Toolbox
// documents (mathworks.com/help/econ/presample-data-for-conditional-variance-
// estimation.html) that presample variances/innovations default to "the sample mean
// of squared response series" but does not disclose the exact numeric procedure, so
// this is a standard, fixed-backcast choice rather than a verified bit-exact match.
// Critically, the backcast must stay fixed across optimization iterations (not
// recomputed from the trial omega/alpha/beta), matching the documented "presample
// default" semantics.
GarchPathResult evaluate_path(
  const std::vector<double> & w, bool has_constant,
  double c, double phi1, double phi2, double omega, double alpha, double beta,
  double backcast_variance)
{
  GarchPathResult result;
  if (w.size() <= static_cast<size_t>(ArimaGarchModel::kOrder)) {
    return result;
  }

  double prev_eps2 = backcast_variance;
  double prev_sigma2 = backcast_variance;
  double log_lik = 0.0;
  constexpr double kLog2Pi = 1.8378770664093453;

  for (size_t t = static_cast<size_t>(ArimaGarchModel::kOrder); t < w.size(); ++t) {
    const double mean = (has_constant ? c : 0.0) + phi1 * w[t - 1] + phi2 * w[t - 2];
    const double eps = w[t] - mean;
    const double sigma2 = std::max(omega + alpha * prev_eps2 + beta * prev_sigma2, 1e-12);
    log_lik += -0.5 * kLog2Pi - 0.5 * std::log(sigma2) - 0.5 * (eps * eps) / sigma2;
    prev_eps2 = eps * eps;
    prev_sigma2 = sigma2;
    result.last_eps = eps;
    result.last_sigma2 = sigma2;
  }
  result.log_likelihood = log_lik;
  result.valid = true;
  return result;
}

double negative_log_likelihood(
  const std::vector<double> & w, bool has_constant, double backcast_variance,
  const std::vector<double> & theta)
{
  double c = 0.0;
  double phi1 = 0.0;
  double phi2 = 0.0;
  double omega = 0.0;
  double alpha = 0.0;
  double beta = 0.0;
  decode_params(theta, has_constant, c, phi1, phi2, omega, alpha, beta);
  const GarchPathResult result = evaluate_path(w, has_constant, c, phi1, phi2, omega, alpha, beta, backcast_variance);
  if (!result.valid || !std::isfinite(result.log_likelihood)) {
    return 1e12;
  }
  return -result.log_likelihood;
}

// Dependency-free Nelder-Mead simplex minimizer (Nelder & Mead, 1965). Used in place
// of MATLAB's `estimate` (fmincon-based MLE) since no nonlinear-optimization or
// econometrics toolbox is available in C++; the objective/constraints are exact,
// the optimizer implementation is not the same one MATLAB uses internally.
std::vector<double> nelder_mead(
  const std::function<double(const std::vector<double> &)> & objective,
  std::vector<double> x0, int max_iterations, double tolerance)
{
  const size_t n = x0.size();
  if (n == 0) {
    return x0;
  }
  constexpr double kAlpha = 1.0;
  constexpr double kGamma = 2.0;
  constexpr double kRho = 0.5;
  constexpr double kSigma = 0.5;

  std::vector<std::vector<double>> simplex(n + 1, x0);
  for (size_t i = 0; i < n; ++i) {
    const double step = std::abs(x0[i]) > 1e-8 ? 0.1 * x0[i] : 0.1;
    simplex[i + 1][i] += step;
  }

  std::vector<double> scores(n + 1);
  for (size_t i = 0; i <= n; ++i) {
    scores[i] = objective(simplex[i]);
  }

  for (int iter = 0; iter < max_iterations; ++iter) {
    std::vector<size_t> order(n + 1);
    for (size_t i = 0; i <= n; ++i) {
      order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&scores](size_t a, size_t b) { return scores[a] < scores[b]; });

    std::vector<std::vector<double>> sorted_simplex(n + 1);
    std::vector<double> sorted_scores(n + 1);
    for (size_t i = 0; i <= n; ++i) {
      sorted_simplex[i] = simplex[order[i]];
      sorted_scores[i] = scores[order[i]];
    }
    simplex = sorted_simplex;
    scores = sorted_scores;

    if (std::abs(scores[n] - scores[0]) < tolerance) {
      break;
    }

    std::vector<double> centroid(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        centroid[j] += simplex[i][j];
      }
    }
    for (size_t j = 0; j < n; ++j) {
      centroid[j] /= static_cast<double>(n);
    }

    std::vector<double> reflected(n);
    for (size_t j = 0; j < n; ++j) {
      reflected[j] = centroid[j] + kAlpha * (centroid[j] - simplex[n][j]);
    }
    const double reflected_score = objective(reflected);

    if (reflected_score < scores[0]) {
      std::vector<double> expanded(n);
      for (size_t j = 0; j < n; ++j) {
        expanded[j] = centroid[j] + kGamma * (reflected[j] - centroid[j]);
      }
      const double expanded_score = objective(expanded);
      if (expanded_score < reflected_score) {
        simplex[n] = expanded;
        scores[n] = expanded_score;
      } else {
        simplex[n] = reflected;
        scores[n] = reflected_score;
      }
      continue;
    }

    if (reflected_score < scores[n - 1]) {
      simplex[n] = reflected;
      scores[n] = reflected_score;
      continue;
    }

    std::vector<double> contracted(n);
    for (size_t j = 0; j < n; ++j) {
      contracted[j] = centroid[j] + kRho * (simplex[n][j] - centroid[j]);
    }
    const double contracted_score = objective(contracted);
    if (contracted_score < scores[n]) {
      simplex[n] = contracted;
      scores[n] = contracted_score;
      continue;
    }

    for (size_t i = 1; i <= n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        simplex[i][j] = simplex[0][j] + kSigma * (simplex[i][j] - simplex[0][j]);
      }
      scores[i] = objective(simplex[i]);
    }
  }

  size_t best = 0;
  for (size_t i = 1; i <= n; ++i) {
    if (scores[i] < scores[best]) {
      best = i;
    }
  }
  return simplex[best];
}

} // namespace

void ArimaGarchModel::fit(const std::vector<double> & window, int d, double variance_floor)
{
  d_ = d;
  has_constant_ = (d == 0);
  variance_floor_ = variance_floor;
  fitted_ = false;

  const std::vector<double> w = difference(window, d_);
  if (w.size() <= static_cast<size_t>(kOrder) + 1) {
    // Not enough data for a well-posed joint MLE fit: fall back to a flat model
    // whose forecast variance is the (floored) sample variance of the series.
    c_ = w.empty() ? 0.0 : std::accumulate(w.begin(), w.end(), 0.0) / static_cast<double>(w.size());
    phi1_ = 0.0;
    phi2_ = 0.0;
    omega_ = std::max(w.size() > 1 ? vector_variance(w) : 0.0, variance_floor_ * variance_floor_);
    alpha_ = 0.0;
    beta_ = 0.0;
    backcast_variance_ = omega_;
    fitted_ = true;
    return;
  }

  // Conditional least squares gives a well-behaved starting point for the AR(2)
  // mean coefficients before the joint ARMA+GARCH log-likelihood is optimized.
  const int rows = static_cast<int>(w.size()) - kOrder;
  Eigen::MatrixXd design(rows, has_constant_ ? 3 : 2);
  Eigen::VectorXd target(rows);
  for (int i = 0; i < rows; ++i) {
    const int t = i + kOrder;
    int col = 0;
    if (has_constant_) {
      design(i, col++) = 1.0;
    }
    design(i, col++) = w[static_cast<size_t>(t) - 1];
    design(i, col++) = w[static_cast<size_t>(t) - 2];
    target(i) = w[static_cast<size_t>(t)];
  }
  const Eigen::VectorXd ols = design.colPivHouseholderQr().solve(target);
  const double c0 = has_constant_ ? ols(0) : 0.0;
  const double phi1_0 = ols(has_constant_ ? 1 : 0);
  const double phi2_0 = ols(has_constant_ ? 2 : 1);

  const Eigen::VectorXd residual = target - design * ols;
  std::vector<double> residual_values(residual.data(), residual.data() + residual.size());
  const double residual_var = std::max(vector_variance(residual_values), variance_floor_ * variance_floor_);

  // Fixed presample backcast for the GARCH recursion: the sample mean of squared
  // conditional-least-squares residuals, matching MATLAB's documented default
  // ("sample mean of squared response series") for presample variances/innovations.
  // This is computed once, before optimization, and held fixed across iterations
  // and across every subsequent forecast() call for this fitted model.
  double residual_sum_sq = 0.0;
  for (const double value : residual_values) {
    residual_sum_sq += value * value;
  }
  backcast_variance_ = std::max(
    residual_sum_sq / static_cast<double>(residual_values.size()),
    variance_floor_ * variance_floor_);

  // Standard GARCH(1,1) starting point (moderate persistence, ARCH-dominated).
  constexpr double kAlpha0 = 0.1;
  constexpr double kBeta0 = 0.8;
  const double omega0 = residual_var * (1.0 - kAlpha0 - kBeta0);

  const std::vector<double> theta0 = encode_initial(has_constant_, c0, phi1_0, phi2_0, omega0, kAlpha0, kBeta0);
  const double backcast = backcast_variance_;
  const auto objective = [&w, this, backcast](const std::vector<double> & theta) {
    return negative_log_likelihood(w, has_constant_, backcast, theta);
  };
  const std::vector<double> theta_opt = nelder_mead(objective, theta0, 4000, 1e-9);

  decode_params(theta_opt, has_constant_, c_, phi1_, phi2_, omega_, alpha_, beta_);
  fitted_ = true;
}

std::pair<double, double> ArimaGarchModel::forecast(const std::vector<double> & window) const
{
  if (!fitted_ || window.size() <= static_cast<size_t>(kOrder)) {
    const double fallback = window.empty() ? 0.0 : window.back();
    return {fallback, variance_floor_ * variance_floor_};
  }

  const std::vector<double> w = difference(window, d_);
  const GarchPathResult path =
    evaluate_path(w, has_constant_, c_, phi1_, phi2_, omega_, alpha_, beta_, backcast_variance_);

  double w_forecast = 0.0;
  double variance_forecast = 0.0;
  if (path.valid) {
    w_forecast = (has_constant_ ? c_ : 0.0) + phi1_ * w[w.size() - 1] + phi2_ * w[w.size() - 2];
    variance_forecast = omega_ + alpha_ * (path.last_eps * path.last_eps) + beta_ * path.last_sigma2;
  } else {
    w_forecast = w.empty() ? 0.0 : w.back();
    variance_forecast = backcast_variance_;
  }

  const double mean_forecast = undifference_forecast(window, d_, w_forecast);
  const double floored_variance = std::max(variance_forecast, variance_floor_ * variance_floor_);
  return {mean_forecast, floored_variance};
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
  const std::vector<double> y_values = trailing_values(false);
  const std::vector<double> theta_values = trailing_values(true);

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
  const std::vector<double> y_values = trailing_values(false);
  const std::vector<double> theta_values = trailing_values(true);
  y_model_.fit(y_values, 2, config_.min_y_sd);
  theta_model_.fit(theta_values, 0, config_.min_theta_sd);
  model_ready_ = true;
  model_fit_seconds_ += elapsed_seconds(start);
}

std::vector<double> HLTSADetector::trailing_values(bool use_theta) const
{
  // Matches MATLAB's HL_est(end-N+Mdl.P+1:end,:) slice: both the initial estimate()
  // call and every subsequent forecast() call use the trailing N-P samples, where
  // N = listener_frames and Mdl.P is the *compound* AR polynomial degree p+D (not
  // just the AR lag order): P=4 for the y model (ARLags 1:2, D=2), P=2 for theta
  // (ARLags 1:2, D=0).
  const int differencing_order = use_theta ? 0 : 2;
  const int p = ArimaGarchModel::kOrder + differencing_order;
  const size_t window_len = std::min<size_t>(
    static_cast<size_t>(std::max(config_.listener_frames - p, 1)),
    states_.size());
  std::vector<double> values;
  values.reserve(window_len);
  for (size_t i = states_.size() - window_len; i < states_.size(); ++i) {
    values.push_back(use_theta ? states_[i].theta_deg : states_[i].y);
  }
  return values;
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
