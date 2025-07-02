cmake_minimum_required(VERSION 3.14)
project(wust_vision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# === 构建选项 ===
option(BUILD_WITH_TRT "Build TensorRT version" ON)
option(BUILD_WITH_OPENVINO "Build OpenVINO version" ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
message(STATUS "BUILD_WITH_TRT: ${BUILD_WITH_TRT}")
message(STATUS "BUILD_WITH_OPENVINO: ${BUILD_WITH_OPENVINO}")




set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake_modules)
set(EXTERNAL_INCLUDE_DIRS ${G2O_INCLUDE_DIR})
set(EXTERNAL_LIBS ${G2O_LIBRARIES})

# === 公共依赖 ===
find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED COMPONENTS Runtime ONNX)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(fmt REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(G2O REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Ceres REQUIRED)

# === TensorRT 库路径（手动指定）===
set(TENSORRT_INCLUDE_DIR /home/hy/TensorRT-8.5.2.2/include)
set(TENSORRT_LIB_DIR /home/hy/TensorRT-8.5.2.2/lib)
set(TENSORRT_LIBS
  ${TENSORRT_LIB_DIR}/libnvinfer.so
  ${TENSORRT_LIB_DIR}/libnvinfer_plugin.so
  ${TENSORRT_LIB_DIR}/libnvonnxparser.so
  ${TENSORRT_LIB_DIR}/libnvparsers.so
)

# === 公共源文件 ===
set(COMMON_SOURCES
  src/common/tools.cpp
  src/common/gobal.cpp
  src/common/toolsgobal.cpp
  src/common/matplottools.cpp
  src/common/calculation.cpp
  src/control/armor_solver.cpp
  src/control/manual_compensator.cpp
  src/control/trajectory_compensator.cpp
  src/control/rune_solver.cpp
  src/detect/armor_pose_estimator.cpp
  src/detect/ba_solver.cpp
  src/detect/graph_optimizer.cpp
  src/detect/light_corner_corrector.cpp
  src/detect/mono_measure_tool.cpp
  src/detect/pnp_solver.cpp
  src/driver/crc8_crc16.cpp
  src/driver/hik.cpp
  src/driver/labeler.cpp
  src/driver/recorder.cpp
  src/driver/serial.cpp
  src/driver/video_player.cpp
  src/tracker/curve_fitter.cpp
  src/tracker/extended_kalman_filter.cpp
  src/tracker/one_ca_tracker.cpp
  src/tracker/one_tracker.cpp
  src/tracker/one_ypd_tracker.cpp
  src/tracker/tracker.cpp
  src/tracker/tracker_manager.cpp
  src/tracker/ypd_tracker.cpp
  src/detect/armor_detector_opencv.cpp
  src/detect/armor_detector_opencv_wrapper.cpp
)

# === include 路径 ===
include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/hikSDK/include
  ${OpenCV_INCLUDE_DIRS}
  ${OpenVINO_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${Python3_INCLUDE_DIRS}
  ${G2O_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  /home/hy/TensorRT-8.5.2.2/include
  /home/hy/wust_vision/thirdparty/matplotlibcpp
  /usr/local/cuda-11.8/include
  
)

# === link 路径 ===
link_directories(
  ${TENSORRT_LIB_DIR}
  /usr/local/cuda-11.8/lib64
)

# === OpenVINO 版本 ===
if(BUILD_WITH_OPENVINO)
  add_executable(${PROJECT_NAME}_openvino
    src/main.cpp
    src/wust_vision.cpp
    src/detect/armor_detector_openvino.cpp
    src/detect/armor_detector_openvino_wrapper.cpp
    src/detect/rune_detector_openvino.cpp
    src/detect/rune_detector_openvino_wrapper.cpp
    ${COMMON_SOURCES}
  )
  target_compile_definitions(${PROJECT_NAME}_openvino PRIVATE USE_OPENVINO)

  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    target_link_directories(${PROJECT_NAME}_openvino PUBLIC hikSDK/lib/amd64)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    target_link_directories(${PROJECT_NAME}_openvino PUBLIC hikSDK/lib/arm64)
  endif()

  target_link_libraries(${PROJECT_NAME}_openvino
    ${OpenCV_LIBS}
    openvino::runtime
    openvino::frontend::onnx
    ${OpenVINO_LIBRARIES}
    ${OpenVINO_RUNTIME_LIBRARIES}
    ${OpenVINO_ONNX_LIBRARIES}
    ${TENSORRT_LIBS}  # 若某些模块间仍有依赖
    /usr/local/cuda-11.8/lib64/libcudart.so
    fmt::fmt
    yaml-cpp
    ${Python3_LIBRARIES}
    g2o_core g2o_stuff g2o_solver_csparse g2o_types_sba g2o_types_slam3d g2o_solver_dense
    ${OPENGL_gl_LIBRARY}
    ${CERES_LIBRARIES}
    FormatConversion MediaProcess MvCameraControl MVRender MvUsb3vTL
    GL
    tbb
  )
endif()

# === TensorRT 版本 ===
if(BUILD_WITH_TRT)
  add_executable(${PROJECT_NAME}_trt
    src/main.cpp
    src/wust_vision.cpp
    src/detect/armor_detector_trt.cpp
    src/detect/armor_detector_trt_wrapper.cpp
    src/detect/rune_detector_trt.cpp
    src/detect/rune_detector_trt_wrapper.cpp
    ${COMMON_SOURCES}
  )
  target_compile_definitions(${PROJECT_NAME}_trt PRIVATE USE_TRT)

  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    target_link_directories(${PROJECT_NAME}_trt PUBLIC hikSDK/lib/amd64)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    target_link_directories(${PROJECT_NAME}_trt PUBLIC hikSDK/lib/arm64)
  endif()

  target_link_libraries(${PROJECT_NAME}_trt
    ${OpenCV_LIBS}
    openvino::runtime
    openvino::frontend::onnx
    ${OpenVINO_LIBRARIES}
    ${OpenVINO_RUNTIME_LIBRARIES}
    ${OpenVINO_ONNX_LIBRARIES}
    ${TENSORRT_LIBS}
    /usr/local/cuda-11.8/lib64/libcudart.so
    fmt::fmt
    yaml-cpp
    ${Python3_LIBRARIES}
    g2o_core g2o_stuff g2o_solver_csparse g2o_types_sba g2o_types_slam3d g2o_solver_dense
    ${OPENGL_gl_LIBRARY}
    ${CERES_LIBRARIES}
    FormatConversion MediaProcess MvCameraControl MVRender MvUsb3vTL
    GL
    tbb
  )
endif()
