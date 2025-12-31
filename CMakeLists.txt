cmake_minimum_required(VERSION 3.14)
project(wust_vision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_UNITY_BUILD ON)
set(CMAKE_UNITY_BUILD_BATCH_SIZE 16) 
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_BUILD_TYPE "Release")
message(STATUS "--------------------CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}--------------------")


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
find_package(Ceres       REQUIRED)
find_package(HikSDK      REQUIRED)
find_package(wust_vl REQUIRED)

if(BUILD_WITH_OPENVINO)
    find_package(OpenVINO REQUIRED COMPONENTS Runtime ONNX)
    list(APPEND BUILD_DEFINITIONS USE_OPENVINO)
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
    set(CMAKE_CUDA_COMPILER_LAUNCHER ccache)
    add_subdirectory(${CMAKE_SOURCE_DIR}/cuda_infer ${CMAKE_BINARY_DIR}/cuda_infer_build)
    list(APPEND BUILD_DEFINITIONS USE_TRT)
endif()

if(BUILD_WITH_NCNN)
    find_package(ncnn REQUIRED)
    list(APPEND BUILD_DEFINITIONS USE_NCNN)
endif()

if(BUILD_WITH_ORT)
    set(ort_root_path "/home/hy/onnxruntime-linux-x64-gpu-1.22.0")
    find_package(Ort REQUIRED)
    list(APPEND BUILD_DEFINITIONS USE_ORT)
endif()

add_library(project_options INTERFACE)
target_compile_definitions(project_options INTERFACE ${BUILD_DEFINITIONS})

include_directories(${EIGEN3_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${Ceres_INCLUDE_DIRS})

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
    utils
)

# ------------------- Subdirectories -------------------
add_subdirectory(tasks/auto_aim)
add_subdirectory(tasks/auto_buff)
add_subdirectory(tasks/auto_sniper)
add_subdirectory(tasks/auto_guidance)
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
add_executable(dart src/dart.cpp)
target_link_libraries(dart
    project_min_deps
    yaml-cpp
    $<TARGET_OBJECTS:auto_guidance>
    ${WUST_auto_guidance_LIBS}
    
)

add_executable(test_usbcamera test/test_usbcamera.cpp)
target_link_libraries(test_usbcamera project_min_deps)
# ------------------- ROS2 -------------------
find_package(ament_cmake QUIET)
find_package(rclcpp QUIET)
find_package(geometry_msgs QUIET)
find_package(std_msgs QUIET)
find_package(tf2_ros QUIET)
find_package(wust_interfaces QUIET)
find_package(sensor_msgs QUIET)

set(ROS2_CORE_FOUND FALSE)
set(ROS2_FULL_FOUND FALSE)

if(ament_cmake_FOUND AND rclcpp_FOUND)
  set(ROS2_CORE_FOUND TRUE)
endif()

if(ament_cmake_FOUND AND rclcpp_FOUND  AND geometry_msgs_FOUND AND wust_interfaces_FOUND AND tf2_ros_FOUND)
  set(ROS2_FULL_FOUND TRUE)
  
endif()

if(ROS2_CORE_FOUND)
  message(STATUS "ROS2 core environment found, adding ros2 subdirectory.")
  add_subdirectory(ros2)
endif()

if(ROS2_FULL_FOUND)
  message(STATUS "ROS2 full environment found, compiling test_ros2 and sentry.")
  include_directories(${geometry_msgs_INCLUDE_DIRS} ${sensor_msgs_INCLUDE_DIRS} ${rclcpp_INCLUDE_DIRS} ${rcl_interfaces_INCLUDE_DIRS} ${tf2_ros_INCLUDE_DIRS} ${wust_interfaces_INCLUDE_DIRS} )

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
  ament_target_dependencies(sentry rclcpp std_msgs geometry_msgs wust_interfaces tf2_ros)
  add_executable(hero src/hero.cpp)
  target_link_libraries(hero
    project_min_deps
    $<TARGET_OBJECTS:auto_aim>
    ${WUST_AUTO_AIM_LIBS}
    $<TARGET_OBJECTS:auto_buff>
    ${WUST_AUTO_BUFF_LIBS}
    tasks_common
    toros2
    auto_sniper
  )
  ament_target_dependencies(hero rclcpp std_msgs geometry_msgs wust_interfaces tf2_ros)
  add_executable(sim src/sim.cpp)
  target_link_libraries(sim
    project_min_deps
    $<TARGET_OBJECTS:auto_aim>
    ${WUST_AUTO_AIM_LIBS}
    $<TARGET_OBJECTS:auto_buff>
    ${WUST_AUTO_BUFF_LIBS}
    tasks_common
    toros2
  )
  ament_target_dependencies(sim rclcpp std_msgs geometry_msgs  tf2_ros sensor_msgs )
  add_executable(control test/control.cpp)
  target_link_libraries(control project_min_deps toros2)
  ament_target_dependencies(control rclcpp std_msgs)
else()
  message(WARNING "ROS2 dependencies incomplete, skipping ...")
endif()
