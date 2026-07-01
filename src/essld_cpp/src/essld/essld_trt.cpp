/**
 * ESSLD TensorRT engine wrapper.
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

#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace essld_cpp
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

  // Enumerate I/O tensors and validate the expected DCEUNet layout: exactly
  // one FLOAT input (1,3,H,W) and one FLOAT output (1,1,H,W).
  int input_count = 0;
  int output_count = 0;
  nvinfer1::Dims input_dims{};
  nvinfer1::Dims output_dims{};

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
      output_name_ = name;
      output_dims = engine_->getTensorShape(name);
      ++output_count;
    }
  }

  if (input_count != 1 || output_count != 1) {
    throw std::runtime_error(
      "TensorRT engine must have exactly one input and one output tensor, found " +
      std::to_string(input_count) + " input(s) and " + std::to_string(output_count) +
      " output(s): " + engine_path);
  }

  const bool input_shape_ok = input_dims.nbDims == 4 && input_dims.d[0] == 1 &&
    input_dims.d[1] == 3 && input_dims.d[2] == expected_height && input_dims.d[3] == expected_width;
  const bool output_shape_ok = output_dims.nbDims == 4 && output_dims.d[0] == 1 &&
    output_dims.d[1] == 1 && output_dims.d[2] == expected_height && output_dims.d[3] == expected_width;

  if (!input_shape_ok || !output_shape_ok) {
    throw std::runtime_error(
      "TensorRT engine '" + engine_path + "' shape mismatch: input " +
      dims_to_string(input_dims) + ", output " + dims_to_string(output_dims) +
      "; expected input (1,3," + std::to_string(expected_height) + "," +
      std::to_string(expected_width) + ") and output (1,1," +
      std::to_string(expected_height) + "," + std::to_string(expected_width) +
      ") (model.input_height / model.input_width parameters)");
  }

  input_height_ = expected_height;
  input_width_ = expected_width;
  input_elems_ = 3ULL * input_height_ * input_width_;
  output_elems_ = 1ULL * input_height_ * input_width_;

  cuda_check(cudaStreamCreate(&stream_), "cudaStreamCreate failed");
  cuda_check(
    cudaMallocHost(reinterpret_cast<void **>(&host_input_), input_elems_ * sizeof(float)),
    "cudaMallocHost (input) failed");
  cuda_check(
    cudaMallocHost(reinterpret_cast<void **>(&host_output_), output_elems_ * sizeof(float)),
    "cudaMallocHost (output) failed");
  cuda_check(cudaMalloc(&device_input_, input_elems_ * sizeof(float)), "cudaMalloc (input) failed");
  cuda_check(cudaMalloc(&device_output_, output_elems_ * sizeof(float)), "cudaMalloc (output) failed");

  if (!context_->setTensorAddress(input_name_.c_str(), device_input_) ||
    !context_->setTensorAddress(output_name_.c_str(), device_output_))
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
  if (device_output_) {
    cudaFree(device_output_);
  }
  if (host_input_) {
    cudaFreeHost(host_input_);
  }
  if (host_output_) {
    cudaFreeHost(host_output_);
  }
  // context_/engine_/runtime_ are released by unique_ptr's default deleter:
  // TensorRT 10.x interfaces have public, default virtual destructors.
}

void TRTEngine::infer(const float * input_chw, float * output_logits)
{
  std::memcpy(host_input_, input_chw, input_elems_ * sizeof(float));

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
      host_output_, device_output_, output_elems_ * sizeof(float),
      cudaMemcpyDeviceToHost, stream_),
    "cudaMemcpyAsync D2H failed");

  cuda_check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize failed");

  std::memcpy(output_logits, host_output_, output_elems_ * sizeof(float));
}

} // namespace essld_cpp
