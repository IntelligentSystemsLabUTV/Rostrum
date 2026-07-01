#!/usr/bin/env python3
"""Horizon-line detection using time-series-assisted ROI selection.

This is a Python port of ``HL_Detect_TSA.m``.  It keeps the same high-level
blocks:

* listener block for the first N frames,
* a time-series forecast for the next horizon state,
* parallelogram and rectangular ROIs,
* Canny + line-transform horizon detection,
* an optional absence/presence detector.

The MATLAB implementation uses Econometrics Toolbox ARIMA/GARCH objects and a
Radon transform.  To keep this script lightweight and runnable in the local
Python environment, the forecast is implemented as small AR(2) least-squares
models and the line detector uses OpenCV's Hough transform.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


State = np.ndarray


@dataclass
class FrameExecutionTime:
    frame: int
    block: str
    elapsed_s: float
    y: float
    theta_deg: float
    iterations: int = 0
    absence_flag: int = 0


@dataclass(frozen=True)
class HLTSAConfig:
    """Configuration values matching the MATLAB algorithm where practical."""

    listener_frames: int = 60
    max_iterations: int = 3
    zscore: float = 1.96
    canny_low: int = 50
    canny_high: int = 150
    area_open_fraction: float = 0.005
    hough_threshold: int = 20
    hough_theta_resolution_deg: float = 0.25
    max_abs_horizon_angle_deg: float = 35.0
    min_roi_height_fraction: float = 0.075
    error_band_fraction: float = 0.025
    min_y_sd: float = 2.0
    min_theta_sd: float = 0.25


@dataclass
class ExecutionTimes:
    setup_s: float = 0.0
    listener_s: float = 0.0
    model_fit_s: float = 0.0
    main_loop_s: float = 0.0
    total_s: float = 0.0
    frame_times: list[FrameExecutionTime] = field(default_factory=list)

    def fps(self, frames: int) -> float:
        return frames / self.total_s if self.total_s > 0 else 0.0


@dataclass
class _ARModel:
    coefficients: np.ndarray
    variance: float
    order: int
    differencing: int
    min_sd: float

    @classmethod
    def fit(cls, values: Iterable[float], differencing: int, min_sd: float) -> "_ARModel":
        values = np.asarray(list(values), dtype=float)
        transformed = _difference(values, differencing)
        order = 2

        if transformed.size <= order:
            variance = float(np.var(transformed)) if transformed.size else min_sd * min_sd
            return cls(np.array([0.0, 1.0, 0.0]), max(variance, min_sd * min_sd), order, differencing, min_sd)

        rows = []
        targets = []
        for idx in range(order, transformed.size):
            rows.append([1.0, transformed[idx - 1], transformed[idx - 2]])
            targets.append(transformed[idx])

        design = np.asarray(rows, dtype=float)
        target = np.asarray(targets, dtype=float)
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        residual = target - design @ coefficients
        variance = float(np.var(residual)) if residual.size else min_sd * min_sd
        variance = max(variance, min_sd * min_sd)
        return cls(coefficients, variance, order, differencing, min_sd)

    def forecast(self, values: Iterable[float]) -> tuple[float, float]:
        values = np.asarray(list(values), dtype=float)
        if values.size == 0:
            return 0.0, self.variance

        transformed = _difference(values, self.differencing)
        if transformed.size == 0:
            transformed_next = 0.0
        elif transformed.size == 1:
            transformed_next = float(transformed[-1])
        else:
            transformed_next = float(np.array([1.0, transformed[-1], transformed[-2]]) @ self.coefficients)

        if self.differencing == 0:
            forecast = transformed_next
        elif self.differencing == 1:
            forecast = float(values[-1] + transformed_next)
        elif values.size >= 2:
            forecast = float(2.0 * values[-1] - values[-2] + transformed_next)
        else:
            forecast = float(values[-1])

        local_variance = self._local_variance(transformed)
        return forecast, max(self.variance, local_variance, self.min_sd * self.min_sd)

    def _local_variance(self, transformed: np.ndarray) -> float:
        if transformed.size < 3:
            return self.min_sd * self.min_sd
        tail = transformed[-min(12, transformed.size) :]
        return float(np.var(tail))


def _difference(values: np.ndarray, order: int) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    for _ in range(order):
        out = np.diff(out)
    return out


def hl_detect_tsa(
    video_path: str | Path,
    listener_frames: int | None = None,
    *,
    max_frames: int | None = None,
    config: HLTSAConfig | None = None,
    output_video: str | Path | None = None,
    show: bool = False,
    return_timings: bool = False,
) -> np.ndarray | tuple[np.ndarray, ExecutionTimes]:
    """Detect horizon-line states for a video.

    Args:
        video_path: Input video path.
        listener_frames: Number of frames used by the listener block.  If
            omitted, ``config.listener_frames`` is used.
        max_frames: Optional cap for quick tests.
        config: Algorithm configuration.
        output_video: Optional path for an annotated output video.
        show: Display annotated frames with OpenCV while processing.

    Returns:
        An ``(num_processed_frames, 2)`` array.  Column 0 is the horizon
        vertical position at the image center, and column 1 is the angle in
        degrees.
    """

    total_start = time.perf_counter()
    timings = ExecutionTimes()

    setup_start = time.perf_counter()
    config = config or HLTSAConfig()
    if listener_frames is not None:
        config = HLTSAConfig(**{**config.__dict__, "listener_frames": listener_frames})

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_limit = frame_count if frame_count > 0 else None
    if max_frames is not None:
        total_limit = min(max_frames, total_limit) if total_limit is not None else max_frames

    writer = _make_video_writer(output_video, fps, width, height) if output_video else None
    timings.setup_s = time.perf_counter() - setup_start

    try:
        listener_start = time.perf_counter()
        states, errors, last_frame, processed = _listener_block(
            cap,
            config,
            width,
            height,
            total_limit,
            writer,
            show,
            timings.frame_times,
        )
        timings.listener_s = time.perf_counter() - listener_start
        if processed == 0:
            raise RuntimeError(f"No frames could be read from video: {video_path}")

        model_start = time.perf_counter()
        y_model = _ARModel.fit(
            [state[0] for state in states[-config.listener_frames :]],
            differencing=2,
            min_sd=config.min_y_sd,
        )
        theta_model = _ARModel.fit(
            [state[1] for state in states[-config.listener_frames :]],
            differencing=0,
            min_sd=config.min_theta_sd,
        )
        timings.model_fit_s = time.perf_counter() - model_start

        absent = False
        last_present_index = len(states) - 1
        min_roi_height = int(round(height * config.min_roi_height_fraction))
        blank_mask = np.zeros((height, width), dtype=np.uint8)

        main_loop_start = time.perf_counter()
        while total_limit is None or processed < total_limit:
            ok, frame = cap.read()
            if not ok:
                break

            frame_index = processed + 1
            frame_start = time.perf_counter()
            if absent:
                state, error, absent, mask = _presence_detector(
                    frame,
                    states,
                    errors,
                    min_roi_height,
                    blank_mask,
                    last_present_index,
                    config,
                )
                states.append(state)
                errors.append(error)
                _emit_frame(frame, state, mask, writer, show)
                timings.frame_times.append(
                    FrameExecutionTime(
                        frame=frame_index,
                        block="presence_detector",
                        elapsed_s=time.perf_counter() - frame_start,
                        y=float(state[0]),
                        theta_deg=float(state[1]),
                        absence_flag=int(absent),
                    )
                )
                processed += 1
                last_frame = frame
                continue

            y_values = [state[0] for state in states[-config.listener_frames :]]
            theta_values = [state[1] for state in states[-config.listener_frames :]]
            y_forecast, y_var = y_model.forecast(y_values)
            theta_forecast, theta_var = theta_model.forecast(theta_values)
            y_sd = math.sqrt(y_var)
            theta_sd = math.sqrt(theta_var)

            roi, absent, mask, delta_height = roi_tsm(
                frame,
                np.array([y_forecast, theta_forecast], dtype=float),
                y_sd,
                theta_sd,
                config.zscore,
            )

            if not absent:
                local_state = hlda(roi, config)
                state = np.array(
                    [
                        y_forecast - delta_height + local_state[0] / _cosd(theta_forecast),
                        theta_forecast + local_state[1],
                    ],
                    dtype=float,
                )
                error = get_error(state, frame, config)

                iterations = 0
                min_roi_extra = 0
                while iterations < config.max_iterations and _needs_control_loop(
                    local_state,
                    state,
                    error,
                    errors,
                    height,
                ):
                    iterations += 1
                    roi, (top, bottom) = roi_rect(frame, states[-1], min_roi_extra)
                    mask = blank_mask.copy()
                    mask[top : bottom + 1, :] = 1
                    local_state = hlda(roi, config)
                    state = np.array([local_state[0] + top, local_state[1]], dtype=float)
                    error = get_error(state, frame, config)
                    min_roi_extra += min_roi_height

                states.append(state)
                errors.append(error)
                _emit_frame(frame, state, mask, writer, show)
                timings.frame_times.append(
                    FrameExecutionTime(
                        frame=frame_index,
                        block="main",
                        elapsed_s=time.perf_counter() - frame_start,
                        y=float(state[0]),
                        theta_deg=float(state[1]),
                        iterations=iterations,
                        absence_flag=0,
                    )
                )
            else:
                last_present_index = len(states) - 1
                if states[-1][0] > height / 2.0:
                    state = np.array([height - 1.0, 0.0], dtype=float)
                    mask = blank_mask.copy()
                    mask[max(0, height - 2 * min_roi_height) : height, :] = 1
                else:
                    state = np.array([0.0, 0.0], dtype=float)
                    mask = blank_mask.copy()
                    mask[: min(height, 2 * min_roi_height), :] = 1
                states.append(state)
                errors.append(get_error(state, frame, config))
                _emit_frame(frame, state, mask, writer, show)
                timings.frame_times.append(
                    FrameExecutionTime(
                        frame=frame_index,
                        block="absence_detector",
                        elapsed_s=time.perf_counter() - frame_start,
                        y=float(state[0]),
                        theta_deg=float(state[1]),
                        absence_flag=1,
                    )
                )

            processed += 1
            last_frame = frame

        timings.main_loop_s = time.perf_counter() - main_loop_start
        timings.total_s = time.perf_counter() - total_start
        result = np.asarray(states, dtype=float)
        if return_timings:
            return result, timings
        return result
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()


def _listener_block(
    cap: cv2.VideoCapture,
    config: HLTSAConfig,
    width: int,
    height: int,
    total_limit: int | None,
    writer: cv2.VideoWriter | None,
    show: bool,
    frame_times: list[FrameExecutionTime],
) -> tuple[list[State], list[float], np.ndarray | None, int]:
    states: list[State] = []
    errors: list[float] = []
    min_roi_height = int(round(height * config.min_roi_height_fraction))

    ok, frame = cap.read()
    if not ok:
        return states, errors, None, 0

    frame_start = time.perf_counter()
    state = hlda(frame, config)
    states.append(state)
    errors.append(get_error(state, frame, config))
    _emit_frame(frame, state, None, writer, show)
    frame_times.append(
        FrameExecutionTime(
            frame=1,
            block="listener",
            elapsed_s=time.perf_counter() - frame_start,
            y=float(state[0]),
            theta_deg=float(state[1]),
        )
    )
    processed = 1
    last_frame = frame

    listener_limit = config.listener_frames
    if total_limit is not None:
        listener_limit = min(listener_limit, total_limit)

    while processed < listener_limit:
        ok, frame = cap.read()
        if not ok:
            break
        frame_start = time.perf_counter()
        roi, (top, _bottom) = roi_rect(frame, states[-1], min_roi_height)
        local_state = hlda(roi, config)
        state = np.array([local_state[0] + top, local_state[1]], dtype=float)
        states.append(state)
        errors.append(get_error(state, frame, config))
        _emit_frame(frame, state, None, writer, show)
        frame_times.append(
            FrameExecutionTime(
                frame=processed + 1,
                block="listener",
                elapsed_s=time.perf_counter() - frame_start,
                y=float(state[0]),
                theta_deg=float(state[1]),
            )
        )
        processed += 1
        last_frame = frame

    return states, errors, last_frame, processed


def hlda(frame: np.ndarray, config: HLTSAConfig | None = None) -> np.ndarray:
    """Detect a horizon line in a full frame or ROI."""

    config = config or HLTSAConfig()
    height, width = frame.shape[:2]
    candidates = _hough_candidates(frame, config)
    if not candidates:
        return np.array([height / 2.0, 0.0], dtype=float)

    plausible = [candidate for candidate in candidates if abs(candidate[1]) <= config.max_abs_horizon_angle_deg]
    if not plausible:
        plausible = candidates

    y_center, angle, _rank = plausible[0]
    return np.array([y_center, angle], dtype=float)


def _hough_candidates(frame: np.ndarray, config: HLTSAConfig) -> list[tuple[float, float, int]]:
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    candidates: list[tuple[float, float, int]] = []
    theta_step = math.radians(config.hough_theta_resolution_deg)
    seen: set[tuple[int, int]] = set()
    edges = cv2.Canny(gray, config.canny_low, config.canny_high, L2gradient=True)
    edges = _area_open(edges, config.area_open_fraction)
    lines = cv2.HoughLines(edges, 1.0, theta_step, config.hough_threshold)
    if lines is None:
        return candidates

    for rank, (rho, theta) in enumerate(lines[:, 0, :]):
        if abs(math.sin(theta)) < 1e-8:
            continue
        y_center = (float(rho) - (width / 2.0) * math.cos(theta)) / math.sin(theta)
        angle = math.degrees(math.atan2(-math.cos(theta), math.sin(theta)))
        if y_center < -height or y_center > 2.0 * height:
            continue
        key = (int(round(y_center)), int(round(angle * 4.0)))
        if key in seen:
            continue
        seen.add(key)
        candidates.append((y_center, angle, rank))

    return candidates


def _area_open(edges: np.ndarray, fraction: float) -> np.ndarray:
    edge_count = int(np.count_nonzero(edges))
    min_area = int(round(edge_count * fraction))
    if edge_count == 0 or min_area <= 1:
        return edges

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats((edges > 0).astype(np.uint8), 8)
    opened = np.zeros_like(edges)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            opened[labels == label] = 255

    return opened if np.any(opened) else edges


def get_error(state: State, frame: np.ndarray, config: HLTSAConfig | None = None) -> float:
    """Compute the sky/sea histogram error used by the control loop."""

    config = config or HLTSAConfig()
    height, width = frame.shape[:2]
    delta_h = config.error_band_fraction * height
    y_coords, x_coords = np.mgrid[1 : height + 1, 1 : width + 1]
    sky_mask, sea_mask = find_indices_of_regions(state, width, x_coords, y_coords, delta_h)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not np.any(sky_mask) or not np.any(sea_mask):
        return 0.0

    hist_sky = _probability_histogram(gray[sky_mask])
    hist_sea = _probability_histogram(gray[sea_mask])
    err = -float(np.sqrt(np.mean((hist_sky - hist_sea) ** 2)))
    return 0.0 if not np.isfinite(err) else err


def find_indices_of_regions(
    state: State,
    width: int,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    delta_h: float,
) -> tuple[np.ndarray, np.ndarray]:
    y_intercept = float(state[0])
    theta = float(state[1])
    slope = _tand(theta)
    c = y_intercept - slope * width / 2.0
    reference = slope * x_coords + c
    lower = reference - delta_h
    upper = reference + delta_h
    sky = (y_coords < reference) & (y_coords > lower)
    sea = (y_coords > reference) & (y_coords < upper)
    return sky, sea


def _probability_histogram(values: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=np.arange(256))
    total = counts.sum()
    if total == 0:
        return np.zeros(255, dtype=float)
    return counts.astype(float) / float(total)


def roi_rect(frame: np.ndarray, state: State, min_height: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build the rectangular ROI around the previous horizon estimate."""

    height, width = frame.shape[:2]
    xs = np.arange(1, width + 1, dtype=float)
    ys = np.rint(_tand(float(state[1])) * xs + float(state[0]) - _tand(float(state[1])) * width / 2.0)
    top = int(np.floor(np.min(ys) - min_height))
    bottom = int(np.ceil(np.max(ys) + min_height))
    top = max(0, min(height - 1, top))
    bottom = max(0, min(height - 1, bottom))
    if bottom < top:
        top, bottom = bottom, top
    return frame[top : bottom + 1, :, :], (top, bottom)


def roi_tsm(
    frame: np.ndarray,
    state: State,
    y_sd: float,
    theta_sd: float,
    zscore: float,
) -> tuple[np.ndarray | None, bool, np.ndarray | None, int]:
    """Build the parallelogram ROI from a forecasted state and uncertainty."""

    height, width = frame.shape[:2]
    y_forecast = float(state[0])
    theta_forecast = float(state[1])
    slope = _tand(theta_forecast)

    horizon = np.rint(slope * np.array([1.0, float(width)]) + y_forecast - slope * width / 2.0)
    angled_slope = _tand(theta_forecast + theta_sd)
    angled_right = angled_slope * width + y_forecast - angled_slope * width / 2.0
    delta_height = max(1, int(round(zscore * (y_sd + abs(angled_right - horizon[1])))))

    rows = np.array(
        [
            horizon[0] - delta_height,
            horizon[0] + delta_height,
            horizon[1] + delta_height,
            horizon[1] - delta_height,
        ],
        dtype=np.int32,
    )
    cols = np.array([0, 0, width - 1, width - 1], dtype=np.int32)
    polygon = np.stack([cols, rows], axis=1)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    if float(np.count_nonzero(mask)) / float(mask.size) <= 0.0025:
        return None, True, None, 0

    col_counts = mask.sum(axis=0)
    max_count = int(col_counts.max(initial=0))
    if max_count <= 0:
        return None, True, None, 0

    keep_cols = np.flatnonzero(col_counts == max_count)
    if keep_cols.size == 0:
        return None, True, None, 0

    trimmed_mask = np.zeros_like(mask)
    trimmed_mask[:, keep_cols] = mask[:, keep_cols]
    roi = np.empty((max_count, keep_cols.size, frame.shape[2]), dtype=frame.dtype)
    for out_col, source_col in enumerate(keep_cols):
        roi[:, out_col, :] = frame[trimmed_mask[:, source_col].astype(bool), source_col, :]

    return roi, False, trimmed_mask, delta_height


def _presence_detector(
    frame: np.ndarray,
    states: list[State],
    errors: list[float],
    min_roi_height: int,
    blank_mask: np.ndarray,
    last_present_index: int,
    config: HLTSAConfig,
) -> tuple[State, float, bool, np.ndarray]:
    height = frame.shape[0]
    if states[-1][0] > height / 2.0:
        top = max(0, height - 2 * min_roi_height)
        bottom = height - 1
    else:
        top = 0
        bottom = min(height - 1, 2 * min_roi_height)

    roi = frame[top : bottom + 1, :, :]
    local_state = hlda(roi, config)
    state = np.array([local_state[0] + top, local_state[1]], dtype=float)
    error = get_error(state, frame, config)

    reference_error = errors[last_present_index] if 0 <= last_present_index < len(errors) else errors[-1]
    denominator = reference_error if abs(reference_error) > 1e-12 else 1.0
    still_absent = abs((error - reference_error) / denominator) > 0.05
    if still_absent:
        state = states[-1].copy()

    mask = blank_mask.copy()
    mask[top : bottom + 1, :] = 1
    return state, error, still_absent, mask


def _needs_control_loop(
    local_state: State,
    state: State,
    error: float,
    previous_errors: list[float],
    frame_height: int,
) -> bool:
    recent = previous_errors[-4:] + [error]
    recent_mean = float(np.mean(recent)) if recent else 0.0
    denominator = recent_mean if abs(recent_mean) > 1e-12 else 1.0
    error_jump = abs((error - recent_mean) / denominator) > 0.15
    out_of_bounds = local_state[0] > 3.0 * frame_height or local_state[0] < 0.0
    return bool(error_jump or out_of_bounds)


def draw_horizon(frame: np.ndarray, state: State, color: tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
    out = frame.copy()
    height, width = out.shape[:2]
    slope = _tand(float(state[1]))
    y_left = int(round(slope * 1.0 + float(state[0]) - slope * width / 2.0))
    y_right = int(round(slope * width + float(state[0]) - slope * width / 2.0))
    cv2.line(out, (0, y_left), (width - 1, y_right), color, 3, cv2.LINE_AA)
    return out


def draw_roi(frame: np.ndarray, mask: np.ndarray | None, color: tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    if mask is None:
        return frame
    out = frame.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(out, contours, -1, color, 3, cv2.LINE_AA)
    return out


def _emit_frame(
    frame: np.ndarray,
    state: State,
    mask: np.ndarray | None,
    writer: cv2.VideoWriter | None,
    show: bool,
) -> None:
    if writer is None and not show:
        return
    annotated = draw_roi(draw_horizon(frame, state), mask)
    if writer is not None:
        writer.write(annotated)
    if show:
        cv2.imshow("HL-TSA", annotated)
        cv2.waitKey(1)


def _make_video_writer(path: str | Path | None, fps: float, width: int, height: int) -> cv2.VideoWriter | None:
    if path is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if suffix == ".mp4" else "XVID"))
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {path}")
    return writer


def load_ground_truth_states(mat_path: str | Path, width: int) -> np.ndarray:
    """Load HorizonGT MAT files into ``[y_at_center, theta_deg]`` states."""

    from scipy.io import loadmat

    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if "structXML" not in mat:
        raise KeyError(f"{mat_path} does not contain 'structXML'")

    states = []
    for item in np.ravel(mat["structXML"]):
        nx = float(item.Nx)
        ny = float(item.Ny)
        x0 = float(item.X)
        y0 = float(item.Y)
        slope = -nx / ny if abs(ny) > 1e-12 else 0.0
        theta = math.degrees(math.atan(slope))
        y_center = y0 + slope * (width / 2.0 - x0)
        states.append([y_center, theta])
    return np.asarray(states, dtype=float)


def save_states_csv(path: str | Path, states: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "y", "theta_deg"])
        for idx, (y_value, theta) in enumerate(states, start=1):
            writer.writerow([idx, f"{y_value:.6f}", f"{theta:.6f}"])


def save_timing_csv(path: str | Path, timings: ExecutionTimes, frames: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "seconds"])
        writer.writerow(["setup", f"{timings.setup_s:.6f}"])
        writer.writerow(["listener", f"{timings.listener_s:.6f}"])
        writer.writerow(["model_fit", f"{timings.model_fit_s:.6f}"])
        writer.writerow(["main_loop", f"{timings.main_loop_s:.6f}"])
        writer.writerow(["total", f"{timings.total_s:.6f}"])
        writer.writerow(["frames", frames])
        writer.writerow(["fps", f"{timings.fps(frames):.6f}"])


def save_frame_timing_csv(path: str | Path, frame_times: list[FrameExecutionTime]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "block", "elapsed_seconds", "y", "theta_deg", "iterations", "absence_flag"])
        for row in frame_times:
            writer.writerow(
                [
                    row.frame,
                    row.block,
                    f"{row.elapsed_s:.9f}",
                    f"{row.y:.6f}",
                    f"{row.theta_deg:.6f}",
                    row.iterations,
                    row.absence_flag,
                ]
            )


def _video_size(video_path: str | Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()


def _tand(degrees: float) -> float:
    return math.tan(math.radians(degrees))


def _cosd(degrees: float) -> float:
    value = math.cos(math.radians(degrees))
    if abs(value) < 1e-8:
        return 1e-8 if value >= 0 else -1e-8
    return value


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect maritime horizon lines using a TSA-style dynamic ROI.")
    parser.add_argument("video", type=Path, help="Input video file.")
    parser.add_argument("-N", "--listener-frames", type=int, default=60, help="Listener block frame count.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame cap for quick tests.")
    parser.add_argument("--ground-truth", type=Path, default=None, help="Optional HorizonGT .mat file for evaluation.")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path for states.")
    parser.add_argument("--timing-csv", type=Path, default=None, help="Optional CSV output path for execution times.")
    parser.add_argument(
        "--frame-timing-csv",
        type=Path,
        default=None,
        help="Optional CSV output path for per-frame HL detection execution times.",
    )
    parser.add_argument("--output-video", type=Path, default=None, help="Optional annotated output video path.")
    parser.add_argument("--show", action="store_true", help="Display annotated frames while processing.")
    parser.add_argument("--canny-low", type=int, default=50, help="OpenCV Canny low threshold.")
    parser.add_argument("--canny-high", type=int, default=150, help="OpenCV Canny high threshold.")
    parser.add_argument("--hough-threshold", type=int, default=20, help="OpenCV Hough vote threshold.")
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    config = HLTSAConfig(
        listener_frames=args.listener_frames,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        hough_threshold=args.hough_threshold,
    )

    states, timings = hl_detect_tsa(
        args.video,
        config=config,
        max_frames=args.max_frames,
        output_video=args.output_video,
        show=args.show,
        return_timings=True,
    )
    print(f"Processed {len(states)} frames")
    print(f"Last state: y={states[-1, 0]:.3f}, theta={states[-1, 1]:.3f} deg")
    print("Execution times:")
    print(f"  setup:      {timings.setup_s:.3f} s")
    print(f"  listener:   {timings.listener_s:.3f} s")
    print(f"  model fit:  {timings.model_fit_s:.3f} s")
    print(f"  main loop:  {timings.main_loop_s:.3f} s")
    print(f"  total:      {timings.total_s:.3f} s ({timings.fps(len(states)):.2f} frames/s)")

    if args.csv:
        save_states_csv(args.csv, states)
        print(f"Wrote states CSV: {args.csv}")

    if args.timing_csv:
        save_timing_csv(args.timing_csv, timings, len(states))
        print(f"Wrote timing CSV: {args.timing_csv}")

    if args.frame_timing_csv:
        save_frame_timing_csv(args.frame_timing_csv, timings.frame_times)
        print(f"Wrote per-frame timing CSV: {args.frame_timing_csv}")

    if args.ground_truth:
        width, _height = _video_size(args.video)
        ground_truth = load_ground_truth_states(args.ground_truth, width)
        count = min(len(states), len(ground_truth))
        if count:
            diff = states[:count] - ground_truth[:count]
            mae = np.mean(np.abs(diff), axis=0)
            rmse = np.sqrt(np.mean(diff**2, axis=0))
            print(f"Compared {count} frames against ground truth")
            print(f"MAE: y={mae[0]:.3f} px, theta={mae[1]:.3f} deg")
            print(f"RMSE: y={rmse[0]:.3f} px, theta={rmse[1]:.3f} deg")

    if args.output_video:
        print(f"Wrote annotated video: {args.output_video}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
