cmake_minimum_required(VERSION 3.8)
project(hl_tsa_cpp)

set(CMAKE_BUILD_TYPE "RelWithDebInfo")

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

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(dua_app_management REQUIRED)
find_package(dua_cv_bridge REQUIRED)
find_package(dua_node_cpp REQUIRED)
find_package(dua_qos_cpp REQUIRED)
find_package(Eigen3 3.4 REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
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
  YAML_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/hl_tsa/params.yaml"
  OUT_FILE "init_parameters.cpp")

# Configure HL-TSA node source files
file(GLOB HL_TSA_NODE_SRCS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "src/hl_tsa/*.cpp")
list(APPEND HL_TSA_NODE_SRCS init_parameters.cpp)

# HL-TSA node/component library
add_library(hl_tsa_cpp SHARED ${HL_TSA_NODE_SRCS})
target_compile_definitions(hl_tsa_cpp PRIVATE COMPOSITION_BUILDING_DLL)
target_compile_features(hl_tsa_cpp PUBLIC c_std_99 cxx_std_17)
target_include_directories(hl_tsa_cpp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(hl_tsa_cpp
  ${OpenCV_LIBS}
  Eigen3::Eigen
  Threads::Threads)
ament_target_dependencies(hl_tsa_cpp
  params_manager_cpp
  dua_cv_bridge
  dua_node_cpp
  dua_qos_cpp
  image_transport
  rclcpp
  rclcpp_components
  sensor_msgs
  std_msgs)
rclcpp_components_register_nodes(hl_tsa_cpp "hl_tsa_cpp::HLTSA")

# HL-TSA ROS app
add_executable(hl_tsa_cpp_app
  src/hl_tsa_app.cpp)
target_include_directories(hl_tsa_cpp_app PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(hl_tsa_cpp_app PUBLIC c_std_99 cxx_std_17)
target_link_libraries(hl_tsa_cpp_app hl_tsa_cpp)
ament_target_dependencies(hl_tsa_cpp_app
  dua_app_management
  rclcpp)

# HL-TSA offline video app
add_executable(hl_tsa_video_app
  src/hl_tsa_video_app.cpp)
target_include_directories(hl_tsa_video_app PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(hl_tsa_video_app PUBLIC c_std_99 cxx_std_17)
target_link_libraries(hl_tsa_video_app hl_tsa_cpp)

# Install HL-TSA component
install(TARGETS hl_tsa_cpp
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

# Install apps
install(TARGETS
  hl_tsa_cpp_app
  hl_tsa_video_app
  DESTINATION lib/${PROJECT_NAME})

# Install Python video publisher utility
install(PROGRAMS
  scripts/video_image_publisher
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
ament_export_libraries(hl_tsa_cpp)
ament_export_dependencies(
  dua_cv_bridge
  dua_node_cpp
  dua_qos_cpp
  image_transport
  rclcpp
  sensor_msgs
  std_msgs)

ament_package()
