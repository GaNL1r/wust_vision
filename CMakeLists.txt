cmake_minimum_required(VERSION 3.14)
project(wust_vision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE "Release")
message(STATUS "--------------------CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}--------------------")

add_compile_options(-O3 -w)

option(BUILD_WITH_TRT      "Enable TensorRT backend"   ON)
option(BUILD_WITH_OPENVINO "Enable OpenVINO backend"   ON)
option(BUILD_WITH_NCNN     "Enable NCNN backend"       ON)
option(BUILD_WITH_ORT      "Enable ORT backend"        ON)

include_directories(${PROJECT_SOURCE_DIR})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

# ------------------- Dependencies -------------------
find_package(OpenCV      REQUIRED)
find_package(Eigen3      REQUIRED)
find_package(fmt         REQUIRED)
find_package(yaml-cpp    REQUIRED)
find_package(G2O         REQUIRED)
find_package(Ceres       REQUIRED)
find_package(HikSDK      REQUIRED)
find_package(TBB         REQUIRED)

if(BUILD_WITH_OPENVINO)
    find_package(OpenVINO REQUIRED COMPONENTS Runtime ONNX)
endif()

if(BUILD_WITH_TRT)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/home/hy/TensorRT-10.6.0.26")
    find_package(TensorRT REQUIRED)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
    set(CMAKE_CUDA_ARCHITECTURES 86)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    add_subdirectory(${CMAKE_SOURCE_DIR}/cuda_infer ${CMAKE_BINARY_DIR}/cuda_infer_build)
endif()

if(BUILD_WITH_NCNN)
    find_package(ncnn REQUIRED)
endif()

if(BUILD_WITH_ORT)
    set(ort_root_path "/home/hy/onnxruntime-linux-x64-gpu-1.22.0")
    find_package(Ort REQUIRED)
endif()

find_package(wust_vl REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})

# ------------------- 公共依赖 target -------------------
add_library(project_min_deps INTERFACE)

target_include_directories(project_min_deps INTERFACE
    ${WUST_VL_INCLUDE_DIRS}
    ${HikSDK_INCLUDE_DIR}
)

target_link_libraries(project_min_deps INTERFACE
    ${WUST_VL_LIBS}
    ${HikSDK_LIB}
    yaml-cpp
    fmt::fmt
    ${OpenCV_LIBS}
)

# ------------------- Subdirectories -------------------
add_subdirectory(tasks/auto_aim)
add_subdirectory(tasks/auto_buff)
add_subdirectory(tasks)

# ------------------- Executables -------------------
add_executable(standard src/standard.cpp)
target_link_libraries(standard
    project_min_deps
    $<TARGET_OBJECTS:auto_aim>
    ${WUST_AUTO_AIM_LIBS}
    $<TARGET_OBJECTS:auto_buff>
    ${WUST_AUTO_BUFF_LIBS}
    tasks_common
)

add_executable(calcapture calibration/capture.cpp)
target_link_libraries(calcapture project_min_deps)

add_executable(calcamera calibration/calcamera.cpp)
target_link_libraries(calcamera project_min_deps)

add_executable(eye_hand calibration/eye_hand.cpp)
target_link_libraries(eye_hand project_min_deps)


# ------------------- ROS2 -------------------
find_package(ament_cmake QUIET)
find_package(rclcpp QUIET)
find_package(geometry_msgs QUIET)
find_package(std_msgs QUIET)
set(ROS2_CORE_FOUND FALSE)
set(ROS2_FULL_FOUND FALSE)

if(ament_cmake_FOUND AND rclcpp_FOUND)
  set(ROS2_CORE_FOUND TRUE)
endif()

if(ament_cmake_FOUND AND rclcpp_FOUND  AND geometry_msgs_FOUND )
  set(ROS2_FULL_FOUND TRUE)
endif()

if(ROS2_CORE_FOUND)
  message(STATUS "ROS2 core environment found, adding ros2 subdirectory.")
  add_subdirectory(ros2)
endif()

if(ROS2_FULL_FOUND)
  message(STATUS "ROS2 full environment found, compiling test_ros2 and sentry.")

  add_executable(test_ros2 test/test_ros2.cpp)
  target_link_libraries(test_ros2
    project_min_deps
    toros2
  )
  ament_target_dependencies(test_ros2 rclcpp std_msgs geometry_msgs)

  add_executable(sentry src/sentry.cpp)
  target_link_libraries(sentry
    project_min_deps
    $<TARGET_OBJECTS:auto_aim>
    ${WUST_AUTO_AIM_LIBS}
    $<TARGET_OBJECTS:auto_buff>
    ${WUST_AUTO_BUFF_LIBS}
    tasks_common
    toros2
  )
  ament_target_dependencies(sentry rclcpp std_msgs geometry_msgs)
  add_executable(control test/control.cpp)
  target_link_libraries(control project_min_deps toros2)
  ament_target_dependencies(control rclcpp std_msgs)
else()
  message(WARNING "ROS2 dependencies incomplete, skipping ...")
endif()
