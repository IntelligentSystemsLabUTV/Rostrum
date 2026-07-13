/**
 * TRiD-Horizon TensorRT engine wrapper.
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

#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace trid_cpp
{

namespace
{

void cuda_check(cudaError_t status, const std::string & what)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(what + ": " + cudaGetErrorString(status));
  }
}

std::vector<char> read_file(const std::string & path)
{
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open TensorRT engine file: " + path);
  }
  const std::streamsize size = file.tellg();
  if (size <= 0) {
    throw std::runtime_error("TensorRT engine file is empty: " + path);
  }
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(static_cast<size_t>(size));
  if (!file.read(buffer.data(), size)) {
    throw std::runtime_error("Failed to read TensorRT engine file: " + path);
  }
  return buffer;
}

std::string dims_to_string(const nvinfer1::Dims & dims)
{
  std::ostringstream oss;
  oss << "(";
  for (int32_t i = 0; i < dims.nbDims; ++i) {
    oss << dims.d[i];
    if (i + 1 < dims.nbDims) {
      oss << ",";
    }
  }
  oss << ")";
  return oss.str();
}

} // namespace

TRTEngine::TRTEngine(
  const std::string & engine_path,
  int gpu_id,
  int clip_length,
  int expected_width,
  int expected_height)
{
  cuda_check(cudaSetDevice(gpu_id), "cudaSetDevice failed");

  const std::vector<char> engine_bytes = read_file(engine_path);

  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  if (!runtime_) {
    throw std::runtime_error("Failed to create TensorRT runtime");
  }

  engine_.reset(runtime_->deserializeCudaEngine(engine_bytes.data(), engine_bytes.size()));
  if (!engine_) {
    throw std::runtime_error("Failed to deserialize TensorRT engine: " + engine_path);
  }

  context_.reset(engine_->createExecutionContext());
  if (!context_) {
    throw std::runtime_error("Failed to create TensorRT execution context");
  }

  // Enumerate I/O tensors and validate the expected TRiD-Horizon export layout:
  // exactly one FLOAT input "input" (1,clip_length,3,H,W), and two FLOAT
  // outputs "heat_logits" (1,1,H,W) and "confidence_logits" (1,1,W).
  int input_count = 0;
  int output_count = 0;
  nvinfer1::Dims input_dims{};
  nvinfer1::Dims heat_dims{};
  nvinfer1::Dims conf_dims{};
  bool have_heat = false;
  bool have_conf = false;

  const int32_t n_tensors = engine_->getNbIOTensors();
  for (int32_t i = 0; i < n_tensors; ++i) {
    const char * name = engine_->getIOTensorName(i);
    const nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(name);
    const nvinfer1::DataType dtype = engine_->getTensorDataType(name);

    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      if (dtype != nvinfer1::DataType::kFLOAT) {
        throw std::runtime_error(
          "TensorRT engine input tensor '" + std::string(name) + "' is not FLOAT32");
      }
      input_name_ = name;
      input_dims = engine_->getTensorShape(name);
      ++input_count;
    } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
      if (dtype != nvinfer1::DataType::kFLOAT) {
        throw std::runtime_error(
          "TensorRT engine output tensor '" + std::string(name) + "' is not FLOAT32");
      }
      const std::string tensor_name(name);
      if (tensor_name == "heat_logits") {
        heat_name_ = name;
        heat_dims = engine_->getTensorShape(name);
        have_heat = true;
      } else if (tensor_name == "confidence_logits") {
        conf_name_ = name;
        conf_dims = engine_->getTensorShape(name);
        have_conf = true;
      }
      ++output_count;
    }
  }

  if (input_count != 1 || output_count != 2 || !have_heat || !have_conf) {
    throw std::runtime_error(
      "TensorRT engine must have exactly one input and two outputs named "
      "'heat_logits'/'confidence_logits', found " + std::to_string(input_count) +
      " input(s) and " + std::to_string(output_count) + " output(s): " + engine_path);
  }

  const bool input_shape_ok = input_dims.nbDims == 5 && input_dims.d[0] == 1 &&
    input_dims.d[1] == clip_length && input_dims.d[2] == 3 &&
    input_dims.d[3] == expected_height && input_dims.d[4] == expected_width;
  const bool heat_shape_ok = heat_dims.nbDims == 4 && heat_dims.d[0] == 1 &&
    heat_dims.d[1] == 1 && heat_dims.d[2] == expected_height && heat_dims.d[3] == expected_width;
  const bool conf_shape_ok = conf_dims.nbDims == 3 && conf_dims.d[0] == 1 &&
    conf_dims.d[1] == 1 && conf_dims.d[2] == expected_width;

  if (!input_shape_ok || !heat_shape_ok || !conf_shape_ok) {
    throw std::runtime_error(
      "TensorRT engine '" + engine_path + "' shape mismatch: input " +
      dims_to_string(input_dims) + ", heat_logits " + dims_to_string(heat_dims) +
      ", confidence_logits " + dims_to_string(conf_dims) + "; expected input (1," +
      std::to_string(clip_length) + ",3," + std::to_string(expected_height) + "," +
      std::to_string(expected_width) + "), heat_logits (1,1," +
      std::to_string(expected_height) + "," + std::to_string(expected_width) +
      "), confidence_logits (1,1," + std::to_string(expected_width) +
      ") (model.clip_length / model.input_height / model.input_width parameters)");
  }

  clip_length_ = clip_length;
  input_height_ = expected_height;
  input_width_ = expected_width;
  input_elems_ = static_cast<size_t>(clip_length_) * 3ULL * input_height_ * input_width_;
  heat_elems_ = 1ULL * input_height_ * input_width_;
  conf_elems_ = 1ULL * input_width_;

  cuda_check(cudaStreamCreate(&stream_), "cudaStreamCreate failed");
  cuda_check(
    cudaMallocHost(reinterpret_cast<void **>(&host_input_), input_elems_ * sizeof(float)),
    "cudaMallocHost (input) failed");
  cuda_check(
    cudaMallocHost(reinterpret_cast<void **>(&host_heat_), heat_elems_ * sizeof(float)),
    "cudaMallocHost (heat) failed");
  cuda_check(
    cudaMallocHost(reinterpret_cast<void **>(&host_conf_), conf_elems_ * sizeof(float)),
    "cudaMallocHost (confidence) failed");
  cuda_check(cudaMalloc(&device_input_, input_elems_ * sizeof(float)), "cudaMalloc (input) failed");
  cuda_check(cudaMalloc(&device_heat_, heat_elems_ * sizeof(float)), "cudaMalloc (heat) failed");
  cuda_check(cudaMalloc(&device_conf_, conf_elems_ * sizeof(float)), "cudaMalloc (confidence) failed");

  if (!context_->setTensorAddress(input_name_.c_str(), device_input_) ||
    !context_->setTensorAddress(heat_name_.c_str(), device_heat_) ||
    !context_->setTensorAddress(conf_name_.c_str(), device_conf_))
  {
    throw std::runtime_error("Failed to bind TensorRT tensor addresses");
  }
}

TRTEngine::~TRTEngine()
{
  if (stream_) {
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
  }
  if (device_input_) {
    cudaFree(device_input_);
  }
  if (device_heat_) {
    cudaFree(device_heat_);
  }
  if (device_conf_) {
    cudaFree(device_conf_);
  }
  if (host_input_) {
    cudaFreeHost(host_input_);
  }
  if (host_heat_) {
    cudaFreeHost(host_heat_);
  }
  if (host_conf_) {
    cudaFreeHost(host_conf_);
  }
  // context_/engine_/runtime_ are released by unique_ptr's default deleter:
  // TensorRT 10.x interfaces have public, default virtual destructors.
}

void TRTEngine::infer(const float * input_clip, float * heat_logits, float * confidence_logits)
{
  std::memcpy(host_input_, input_clip, input_elems_ * sizeof(float));

  cuda_check(
    cudaMemcpyAsync(
      device_input_, host_input_, input_elems_ * sizeof(float),
      cudaMemcpyHostToDevice, stream_),
    "cudaMemcpyAsync H2D failed");

  if (!context_->enqueueV3(stream_)) {
    throw std::runtime_error("TensorRT enqueueV3 failed");
  }

  cuda_check(
    cudaMemcpyAsync(
      host_heat_, device_heat_, heat_elems_ * sizeof(float),
      cudaMemcpyDeviceToHost, stream_),
    "cudaMemcpyAsync D2H (heat) failed");
  cuda_check(
    cudaMemcpyAsync(
      host_conf_, device_conf_, conf_elems_ * sizeof(float),
      cudaMemcpyDeviceToHost, stream_),
    "cudaMemcpyAsync D2H (confidence) failed");

  cuda_check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize failed");

  std::memcpy(heat_logits, host_heat_, heat_elems_ * sizeof(float));
  std::memcpy(confidence_logits, host_conf_, conf_elems_ * sizeof(float));
}

} // namespace trid_cpp
