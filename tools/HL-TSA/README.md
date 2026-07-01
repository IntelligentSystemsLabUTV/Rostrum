# HL-TSA Python Port

Python implementation of `tools/HL-Detection-using-TSA/HL_Detect_TSA.m`.

The main entry point is `hl_detect_tsa.py`. It returns an `N x 2` state matrix
where column 1 is the vertical horizon position at the image center in pixels,
and column 2 is the horizon angle in degrees.

## Usage

Place the Buoy videos and HorizonGT MAT files under `logs/Buoy/` before running
the Python port. The MAT files are used as horizon-line ground truth when passed
with `--ground-truth`.

```bash
python tools/HL-TSA/hl_detect_tsa.py \
  logs/Buoy/buoyGT_2_5_3_4.avi \
  --listener-frames 16 \
  --canny-low 20 \
  --canny-high 60 \
  --hough-threshold 15 \
  --ground-truth logs/Buoy/buoyGT_2_5_3_4HorizonGT.mat \
  --csv logs/Buoy/buoyGT_2_5_3_4_states.csv \
  --timing-csv logs/Buoy/buoyGT_2_5_3_4_timing.csv
```

Add `--output-video path/to/output.avi` to save frames annotated with the
detected horizon in red and the ROI in yellow. Add `--max-frames N` for quick
smoke tests. Each run prints setup, listener-block, model-fit, main-loop, and
total execution time. Use `--frame-timing-csv` to save per-frame HL detection
runtime rows.

## Notes

The MATLAB version depends on the Econometrics Toolbox for ARIMA/GARCH models
and on the Image Processing Toolbox for Radon-based line detection. This port
uses a small local AR(2) forecaster and OpenCV Hough lines so it can run with
the Python packages already available in this workspace.
