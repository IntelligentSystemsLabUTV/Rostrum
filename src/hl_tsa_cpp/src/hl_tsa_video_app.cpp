/**
 * HL-TSA CPU offline video application.
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

#include <cstdlib>
#include <iostream>
#include <string>

namespace
{

void print_usage(const char * program)
{
  std::cout
    << "Usage: " << program << " --video PATH [options]\n\n"
    << "Options:\n"
    << "  --listener-frames N\n"
    << "  --max-frames N\n"
    << "  --canny-low N\n"
    << "  --canny-high N\n"
    << "  --hough-threshold N\n"
    << "  --states-csv PATH\n"
    << "  --frame-timing-csv PATH\n"
    << "  --output-video PATH\n"
    << "  --show\n";
}

bool read_arg(int argc, char ** argv, int & i, std::string & value)
{
  if (i + 1 >= argc) {
    return false;
  }
  value = argv[++i];
  return true;
}

} // namespace

int main(int argc, char ** argv)
{
  hl_tsa_cpp::HLTSAConfig config;
  hl_tsa_cpp::VideoOptions options;

  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    std::string value;
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return EXIT_SUCCESS;
    } else if (arg == "--video" && read_arg(argc, argv, i, value)) {
      options.video_path = value;
    } else if (arg == "--listener-frames" && read_arg(argc, argv, i, value)) {
      config.listener_frames = std::stoi(value);
    } else if (arg == "--max-frames" && read_arg(argc, argv, i, value)) {
      options.max_frames = std::stoi(value);
    } else if (arg == "--canny-low" && read_arg(argc, argv, i, value)) {
      config.canny_low = std::stoi(value);
    } else if (arg == "--canny-high" && read_arg(argc, argv, i, value)) {
      config.canny_high = std::stoi(value);
    } else if (arg == "--hough-threshold" && read_arg(argc, argv, i, value)) {
      config.hough_threshold = std::stoi(value);
    } else if (arg == "--states-csv" && read_arg(argc, argv, i, value)) {
      options.states_csv = value;
    } else if (arg == "--frame-timing-csv" && read_arg(argc, argv, i, value)) {
      options.frame_timing_csv = value;
    } else if (arg == "--output-video" && read_arg(argc, argv, i, value)) {
      options.output_video = value;
    } else if (arg == "--show") {
      options.show = true;
    } else {
      std::cerr << "Unknown or incomplete argument: " << arg << "\n";
      print_usage(argv[0]);
      return EXIT_FAILURE;
    }
  }

  if (options.video_path.empty()) {
    std::cerr << "--video is required\n";
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  try {
    const hl_tsa_cpp::VideoResult result = hl_tsa_cpp::process_video(options, config);
    std::cout << "Processed " << result.states.size() << " frames\n";
    if (!result.states.empty()) {
      const auto & state = result.states.back();
      std::cout << "Last state: y=" << state.y << ", theta=" << state.theta_deg << " deg\n";
    }
    std::cout << "Execution times:\n"
              << "  setup:      " << result.timings.setup_s << " s\n"
              << "  listener:   " << result.timings.listener_s << " s\n"
              << "  model fit:  " << result.timings.model_fit_s << " s\n"
              << "  main loop:  " << result.timings.main_loop_s << " s\n"
              << "  total:      " << result.timings.total_s << " s ("
              << result.timings.fps(static_cast<int>(result.states.size())) << " frames/s)\n";
    if (!options.states_csv.empty()) {
      std::cout << "Wrote states CSV: " << options.states_csv << "\n";
    }
    if (!options.frame_timing_csv.empty()) {
      std::cout << "Wrote per-frame timing CSV: " << options.frame_timing_csv << "\n";
    }
    if (!options.output_video.empty()) {
      std::cout << "Wrote annotated video: " << options.output_video << "\n";
    }
  } catch (const std::exception & ex) {
    std::cerr << "HL-TSA failed: " << ex.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
