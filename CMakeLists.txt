cmake_minimum_required(VERSION 3.14)
project(wust_vision LANGUAGES CXX)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/3rdparty)
# C++ 标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE "Release")

option(BUILD_WITH_RUNE     "Enable RUNE"               ON)
option(BUILD_WITH_TRT      "Enable TensorRT backend"   ON)
option(BUILD_WITH_OPENVINO "Enable OpenVINO backend"   ON)
option(BUILD_WITH_NCNN     "Enable NCNN backend"       ON)
option(BUILD_WITH_ORT      "Enable ORT backend"        ON)

message(STATUS "Options: RUNE=${BUILD_WITH_RUNE}  TRT=${BUILD_WITH_TRT}  OpenVINO=${BUILD_WITH_OPENVINO}  NCNN=${BUILD_WITH_NCNN}  ORT=${BUILD_WITH_ORT}")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake_modules)

# 公共依赖
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/usr/local/opencv4.11.0")

find_package(OpenCV 4.11 REQUIRED)

find_package(Python3     REQUIRED COMPONENTS Interpreter Development)
find_package(Eigen3      REQUIRED)
find_package(fmt         REQUIRED)
find_package(yaml-cpp    REQUIRED)
find_package(G2O         REQUIRED)
find_package(OpenGL      REQUIRED)
find_package(Ceres       REQUIRED)
find_package(HikSDK      REQUIRED)

if(BUILD_WITH_OPENVINO)
  find_package(OpenVINO REQUIRED COMPONENTS Runtime ONNX)
endif()
if(BUILD_WITH_TRT)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/home/hy/TensorRT-8.5.2.2")
  find_package(TensorRT REQUIRED)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
  set(CMAKE_CUDA_ARCHITECTURES 86)  
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  add_subdirectory(cuda_infer)
  include_directories("cuda_infer")
endif()
if(BUILD_WITH_NCNN)
  find_package(ncnn REQUIRED)
endif()
if(BUILD_WITH_ORT)
  set(ort_root_path "/home/hy/onnxruntime-linux-x64-gpu-1.22.0")
  find_package(Ort REQUIRED)
endif()

add_subdirectory(KalmanHyLib)

# 公共源码
set(COMMON_SOURCES
  src/main.cpp
  src/wust_vision.cpp
  src/omni.cpp
  src/common/debug/tools.cpp
  src/common/gobal.cpp
  src/common/debug/toolsgobal.cpp
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
  src/detect/armor_detect/opencv/armor_detector_opencv.cpp
  src/detect/armor_detect/opencv/armor_detector_opencv_wrapper.cpp
  src/detect/armor_detect/armor_detect_common.cpp
  src/detect/armor_detect/number_classifier.cpp
  src/driver/crc8_crc16.cpp
  src/driver/hik.cpp
  src/driver/tools/labeler.cpp
  src/driver/tools/recorder.cpp
  src/driver/serial.cpp
  src/driver/tools/video_player.cpp
  src/tracker/curve_fitter.cpp
  src/tracker/one_ca_tracker.cpp
  src/tracker/one_tracker.cpp
  src/tracker/tracker.cpp
  src/tracker/tracker_manager.cpp
  src/common/camera_calibrator.cpp
  src/detect/armor_detect/armor_infer.cpp
  src/detect/rune_detect/rune_infer.cpp
)

# 各后端专属源码，启用则加入
set(OPENVINO_SOURCES
  src/detect/armor_detect/openvino/armor_detector_openvino.cpp
  src/detect/armor_detect/openvino/armor_detector_openvino_wrapper.cpp
  src/detect/rune_detect/openvino/rune_detector_openvino.cpp
  src/detect/rune_detect/openvino/rune_detector_openvino_wrapper.cpp
)
set(TRT_SOURCES
  src/detect/armor_detect/tensorrt/armor_detector_trt.cpp
  src/detect/armor_detect/tensorrt/armor_detector_trt_wrapper.cpp
  src/detect/rune_detect/tensorrt/rune_detector_trt.cpp
  src/detect/rune_detect/tensorrt/rune_detector_trt_wrapper.cpp

)
set(NCNN_SOURCES
  src/detect/armor_detect/ncnn/armor_detector_ncnn.cpp
  src/detect/armor_detect/ncnn/armor_detector_ncnn_wrapper.cpp
  src/detect/rune_detect/ncnn/rune_detector_ncnn.cpp
  src/detect/rune_detect/ncnn/rune_detector_ncnn_wrapper.cpp
)
set(ORT_SOURCES
  src/detect/armor_detect/onnxruntime/armor_detector_ort.cpp
  src/detect/armor_detect/onnxruntime/armor_detector_ort_wrapper.cpp
  src/detect/rune_detect/onnxruntime/rune_detector_ort.cpp
  src/detect/rune_detect/onnxruntime/rune_detector_ort_wrapper.cpp
)

# 汇总所有启用的后端源码
set(BACKEND_SOURCES "")
if(BUILD_WITH_OPENVINO)
  list(APPEND BACKEND_SOURCES ${OPENVINO_SOURCES})
endif()
if(BUILD_WITH_TRT)
  list(APPEND BACKEND_SOURCES ${TRT_SOURCES})
endif()
if(BUILD_WITH_NCNN)
  list(APPEND BACKEND_SOURCES ${NCNN_SOURCES})
endif()
if(BUILD_WITH_ORT)
  list(APPEND BACKEND_SOURCES ${ORT_SOURCES})
endif()

# 定义宏列表
set(BUILD_DEFINITIONS "")
if(BUILD_WITH_RUNE)
  list(APPEND BUILD_DEFINITIONS USE_RUNE)
endif()
if(BUILD_WITH_OPENVINO)
  list(APPEND BUILD_DEFINITIONS USE_OPENVINO)
endif()
if(BUILD_WITH_TRT)
  list(APPEND BUILD_DEFINITIONS USE_TRT)
endif()
if(BUILD_WITH_NCNN)
  list(APPEND BUILD_DEFINITIONS USE_NCNN)
endif()
if(BUILD_WITH_ORT)
  list(APPEND BUILD_DEFINITIONS USE_ORT)
endif()

# 汇总链接库
set(BUILD_LIBS
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
  KalmanHyLib
)
if(BUILD_WITH_OPENVINO)
  list(APPEND BUILD_LIBS openvino::runtime openvino::frontend::onnx)
endif()
if(BUILD_WITH_TRT)
  list(APPEND BUILD_LIBS TensorRT::TensorRT CUDA::cudart /home/hy/TensorRT-8.5.2.2/lib/libnvonnxparser.so cuda_infer)
endif()
if(BUILD_WITH_NCNN)
  list(APPEND BUILD_LIBS ncnn)
endif()
if(BUILD_WITH_ORT)
  list(APPEND BUILD_LIBS ${Ort_LIB})
endif()

# 创建唯一可执行文件
add_executable(${PROJECT_NAME}
  ${COMMON_SOURCES}
  ${BACKEND_SOURCES}
)

target_include_directories(${PROJECT_NAME} PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
  ${Python3_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${G2O_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${HikSDK_INCLUDE_DIR}
  $<$<BOOL:${BUILD_WITH_OPENVINO}>:${OpenVINO_INCLUDE_DIRS}>
  $<$<BOOL:${BUILD_WITH_TRT}>:${TensorRT_INCLUDE_DIR}>
  $<$<BOOL:${BUILD_WITH_NCNN}>:${ncnn_INCLUDE_DIRS}>
  $<$<BOOL:${BUILD_WITH_ORT}>:${Ort_INCLUDE_DIR}>
)

target_compile_options(${PROJECT_NAME} PRIVATE -w)
target_compile_definitions(${PROJECT_NAME} PRIVATE ${BUILD_DEFINITIONS})

target_link_libraries(${PROJECT_NAME} PRIVATE ${BUILD_LIBS})

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