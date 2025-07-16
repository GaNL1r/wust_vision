cmake_minimum_required(VERSION 3.14)
project(wust_vision LANGUAGES CXX)

# C++ 标准设置
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE "Release")
# 构建选项
option(BUILD_WITH_TRT      "Enable TensorRT backend"   ON)
option(BUILD_WITH_OPENVINO "Enable OpenVINO backend"   ON)
option(BUILD_WITH_NCNN     "Enable NCNN backend"       ON)

message(STATUS "Options: TRT=${BUILD_WITH_TRT}  OpenVINO=${BUILD_WITH_OPENVINO}  NCNN=${BUILD_WITH_NCNN}")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake_modules)
# ----------------------------------------------------------------------------
# 公共依赖
# ----------------------------------------------------------------------------
find_package(OpenCV      REQUIRED)
find_package(Python3     REQUIRED COMPONENTS Interpreter Development)
find_package(Eigen3      REQUIRED)
find_package(fmt         REQUIRED)
find_package(yaml-cpp    REQUIRED)
find_package(G2O         REQUIRED)
find_package(OpenGL      REQUIRED)
find_package(Ceres       REQUIRED)
find_package(HikSDK      REQUIRED)

if (BUILD_WITH_OPENVINO)
  find_package(OpenVINO REQUIRED COMPONENTS Runtime ONNX)
endif()
if (BUILD_WITH_TRT)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/home/hy/TensorRT-8.5.2.2")
  find_package(TensorRT REQUIRED)
endif()
if (BUILD_WITH_NCNN)
  find_package(ncnn REQUIRED)
endif()

# ----------------------------------------------------------------------------
# 源文件列表
# ----------------------------------------------------------------------------
set(COMMON_SOURCES
  src/main.cpp
  src/wust_vision.cpp
  src/omni.cpp
  src/common/debug/tools.cpp
  src/common/gobal.cpp
  src/common/debug/toolsgobal.cpp
  src/common/debug/matplottools.cpp
  src/common/calculation.cpp
  src/control/armor_solver.cpp
  src/control/manual_compensator.cpp
  src/control/trajectory_compensator.cpp
  src/control/rune_solver.cpp
  src/detect/armor_detect/armor_pose_estimator.cpp
  src/detect/ba_solver.cpp
  src/detect/graph_optimizer.cpp
  src/detect/armor_detect/light_corner_corrector.cpp
  src/detect/mono_measure_tool.cpp
  src/detect/pnp_solver.cpp
  src/detect/armor_detect/armor_detector_opencv.cpp
  src/detect/armor_detect/armor_detector_opencv_wrapper.cpp
  src/detect/armor_detect/armor_detect_common.cpp
  src/detect/armor_detect/number_classifier.cpp
  src/driver/crc8_crc16.cpp
  src/driver/hik.cpp
  src/driver/tools/labeler.cpp
  src/driver/tools/recorder.cpp
  src/driver/serial.cpp
  src/driver/tools/video_player.cpp
  src/tracker/math/curve_fitter.cpp
  src/tracker/math/extended_kalman_filter.cpp
  src/tracker/math/error_state_extended_kalman_filter.cpp
  src/tracker/one_ca_tracker.cpp
  src/tracker/one_tracker.cpp
  src/tracker/tracker.cpp
  src/tracker/tracker_manager.cpp
  src/tracker/math/adaptive_extended_kalman_filter.cpp
  src/tracker/math/unscented_kalman_filter.cpp
  src/common/camera_calibrator.cpp
  
)

set(OPENVINO_SOURCES
  src/detect/armor_detect/armor_detector_openvino.cpp
  src/detect/armor_detect/armor_detector_openvino_wrapper.cpp
  src/detect/rune_detect/rune_detector_openvino.cpp
  src/detect/rune_detect/rune_detector_openvino_wrapper.cpp
)

set(TRT_SOURCES
  src/detect/armor_detect/armor_detector_trt.cpp
  src/detect/armor_detect/armor_detector_trt_wrapper.cpp
  src/detect/rune_detect/rune_detector_trt.cpp
  src/detect/rune_detect/rune_detector_trt_wrapper.cpp
)

set(NCNN_SOURCES
  src/detect/armor_detect/armor_detector_ncnn.cpp
  src/detect/armor_detect/armor_detector_ncnn_wrapper.cpp
  src/detect/rune_detect/rune_detector_ncnn.cpp
  src/detect/rune_detect/rune_detector_ncnn_wrapper.cpp
)

# ----------------------------------------------------------------------------
# 函数：创建 wust_vision 可执行目标
# ----------------------------------------------------------------------------
function(add_wust_target suffix srcs defs libs)
  set(target_name "${PROJECT_NAME}_${suffix}")
  add_executable(${target_name} ${COMMON_SOURCES} ${srcs})

  target_include_directories(${target_name} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${G2O_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${HikSDK_INCLUDE_DIR}
    # 可选 include
    $<$<BOOL:${BUILD_WITH_OPENVINO}>:${OpenVINO_INCLUDE_DIRS}>
    $<$<BOOL:${BUILD_WITH_TRT}>:${TensorRT_INCLUDE_DIR}>
    # 如果该目标是 openvino 并启用了 NCNN
    $<$<AND:$<BOOL:${BUILD_WITH_NCNN}>,$<STREQUAL:${suffix},openvino>>:${ncnn_INCLUDE_DIRS}>
    # 如果该目标是 trt 并启用了 NCNN
    $<$<AND:$<BOOL:${BUILD_WITH_NCNN}>,$<STREQUAL:${suffix},trt>>:${ncnn_INCLUDE_DIRS}>
    # 如果该目标是 ncnn
    $<$<AND:$<BOOL:${BUILD_WITH_NCNN}>,$<STREQUAL:${suffix},ncnn>>:${ncnn_INCLUDE_DIRS}>
  )

  target_compile_options(${target_name} PRIVATE -w)
  target_compile_definitions(${target_name} PRIVATE ${defs})
  target_link_libraries(${target_name} PRIVATE
    ${OpenCV_LIBS}
    fmt::fmt
    yaml-cpp
    ${Python3_LIBRARIES}
    g2o_core g2o_stuff g2o_solver_csparse g2o_types_sba g2o_types_slam3d g2o_solver_dense
    ${OPENGL_gl_LIBRARY}
    ${CERES_LIBRARIES}
    ${HikSDK_LIB}
    GL
    tbb
    ${libs}
  )
endfunction()

# ----------------------------------------------------------------------------
# OpenVINO 变体（可选 NCNN）
# ----------------------------------------------------------------------------
if (BUILD_WITH_OPENVINO)
  set(openvino_defs USE_OPENVINO)
  set(openvino_libs openvino::runtime;openvino::frontend::onnx)
  set(openvino_srcs ${OPENVINO_SOURCES})
  if (BUILD_WITH_NCNN)
    list(APPEND openvino_defs USE_NCNN)
    list(APPEND openvino_libs ncnn)
    list(APPEND openvino_srcs ${NCNN_SOURCES})
  endif()
  string(JOIN ";" openvino_defs_str ${openvino_defs})
  string(JOIN ";" openvino_libs_str ${openvino_libs})
  add_wust_target(openvino "${openvino_srcs}" "${openvino_defs_str}" "${openvino_libs_str}")
endif()

# ----------------------------------------------------------------------------
# TensorRT 变体（可选 NCNN）
# ----------------------------------------------------------------------------
if (BUILD_WITH_TRT)
  set(trt_defs USE_TRT)
  set(trt_libs TensorRT::TensorRT;CUDA::cudart;/home/hy/TensorRT-8.5.2.2/lib/libnvonnxparser.so)
  set(trt_srcs ${TRT_SOURCES})
  if (BUILD_WITH_NCNN)
    list(APPEND trt_defs USE_NCNN)
    list(APPEND trt_libs ncnn)
    list(APPEND trt_srcs ${NCNN_SOURCES})
  endif()
  string(JOIN ";" trt_defs_str ${trt_defs})
  string(JOIN ";" trt_libs_str ${trt_libs})
  add_wust_target(trt "${trt_srcs}" "${trt_defs_str}" "${trt_libs_str}")
endif()

# ----------------------------------------------------------------------------
# 纯 NCNN 版本
# ----------------------------------------------------------------------------
if (BUILD_WITH_NCNN AND NOT BUILD_WITH_OPENVINO AND NOT BUILD_WITH_TRT)
  add_wust_target(ncnn "${NCNN_SOURCES}" "USE_NCNN_ONLY" "ncnn")
endif()
add_executable(camera_calibrator
  src/camera_calibrator_main.cpp
  src/driver/hik.cpp
  src/common/camera_calibrator.cpp
  src/common/gobal.cpp
  src/driver/tools/recorder.cpp )
target_include_directories(
  camera_calibrator PRIVATE
  ${HikSDK_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${Python3_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(
  camera_calibrator PRIVATE
  ${HikSDK_LIB}
  ${OpenCV_LIBS}
  ${Python3_LIBRARIES}
  yaml-cpp
)