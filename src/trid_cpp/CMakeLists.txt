cmake_minimum_required(VERSION 3.8)
project(trid_cpp)

set(CMAKE_BUILD_TYPE "RelWithDebInfo")

cmake_policy(SET CMP0146 OLD)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)

# If we're on a Jetson device with L4T, some paths and options must be configured to find stubs
# that the Nvidia container runtime will then mount when an executable is loaded
# NOTES
# - This node uses only the plain TensorRT/CUDA runtime API (no DeepStream), same as essld_cpp.
if(EXISTS "/etc/nv_tegra_release")
  file(STRINGS "/etc/nv_tegra_release" FILE_CONTENTS LIMIT_COUNT 1)
  string(FIND "${FILE_CONTENTS}" "# R36" MATCH_INDEX_R36)
  if(${MATCH_INDEX_R36} GREATER -1)
    message(STATUS "Jetson R36 platform detected, configuring check for CUDA stubs")
    list(APPEND CMAKE_LIBRARY_PATH "/usr/local/cuda/lib64/stubs")
    list(APPEND CMAKE_CXX_FLAGS "-Wl,--allow-shlib-undefined")
  endif()
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(dua_app_management REQUIRED)
find_package(dua_cv_bridge REQUIRED)
find_package(dua_node_cpp REQUIRED)
find_package(dua_qos_cpp REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)
find_package(params_manager_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Threads REQUIRED)

# Generate parameters source code
generate_init_parameters(
  YAML_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/trid/params.yaml"
  OUT_FILE "init_parameters.cpp")

# Configure TRiD node source files
file(GLOB TRID_NODE_SRCS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "src/trid/*.cpp")
list(APPEND TRID_NODE_SRCS init_parameters.cpp)

# TRiD node/component library
add_library(trid_cpp SHARED ${TRID_NODE_SRCS})
target_compile_definitions(trid_cpp PRIVATE COMPOSITION_BUILDING_DLL)
target_compile_features(trid_cpp PUBLIC c_std_99 cxx_std_17)
target_include_directories(trid_cpp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(trid_cpp
  ${OpenCV_LIBS}
  CUDA::cudart
  nvinfer
  Threads::Threads)
ament_target_dependencies(trid_cpp
  params_manager_cpp
  dua_cv_bridge
  dua_node_cpp
  dua_qos_cpp
  image_transport
  rclcpp
  rclcpp_components
  sensor_msgs
  std_msgs)
rclcpp_components_register_nodes(trid_cpp "trid_cpp::TRiD")

# TRiD ROS app
add_executable(trid_cpp_app
  src/trid_app.cpp)
target_include_directories(trid_cpp_app PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(trid_cpp_app PUBLIC c_std_99 cxx_std_17)
target_link_libraries(trid_cpp_app trid_cpp)
ament_target_dependencies(trid_cpp_app
  dua_app_management
  rclcpp)

# Install TRiD component
install(TARGETS trid_cpp
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

# Install app
install(TARGETS
  trid_cpp_app
  DESTINATION lib/${PROJECT_NAME})

# Install headers
install(DIRECTORY include/
  DESTINATION include)

# Install config and launch files
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME})
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_include_directories(include)
ament_export_libraries(trid_cpp)
ament_export_dependencies(
  dua_cv_bridge
  dua_node_cpp
  dua_qos_cpp
  image_transport
  rclcpp
  sensor_msgs
  std_msgs)

ament_package()
