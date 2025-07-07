cmake_minimum_required(VERSION 3.14)
project(wust_vision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# === 构建选项 ===
option(BUILD_WITH_TRT "Build TensorRT version" ON)
option(BUILD_WITH_OPENVINO "Build OpenVINO version" ON)
option(BUILD_WITH_NCNN "Build NCNN version" ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
message(STATUS "BUILD_WITH_TRT: ${BUILD_WITH_TRT}")
message(STATUS "BUILD_WITH_OPENVINO: ${BUILD_WITH_OPENVINO}")
message(STATUS "BUILD_WITH_NCNN: ${BUILD_WITH_NCNN}")




set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake_modules)
set(EXTERNAL_INCLUDE_DIRS ${G2O_INCLUDE_DIR})
set(EXTERNAL_LIBS ${G2O_LIBRARIES})


find_package(OpenCV REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(G2O REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Ceres REQUIRED)

if(BUILD_WITH_OPENVINO)
find_package(OpenVINO REQUIRED COMPONENTS Runtime ONNX)
endif()
if(BUILD_WITH_TRT)
find_package(CUDAToolkit REQUIRED)
endif()
if(BUILD_WITH_NCNN)
find_package(ncnn REQUIRED) 
endif()

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
  src/main.cpp
  src/wust_vision.cpp
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
  src/detect/armor_detect/armor_detector_opencv.cpp
  src/detect/armor_detect/armor_detector_opencv_wrapper.cpp
  src/detect/armor_detect/armor_detect_common.cpp
  src/detect/armor_detect/number_classifier.cpp
  
)
set(NCNN_SOURCES
  src/detect/armor_detect/armor_detector_ncnn.cpp
  src/detect/armor_detect/armor_detector_ncnn_wrapper.cpp
  src/detect/rune_detect/rune_detector_ncnn.cpp
  src/detect/rune_detect/rune_detector_ncnn_wrapper.cpp
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
  /usr/local/cuda-11.8/include
  ${ncnn_INCLUDE_DIRS}
  
)

# === link 路径 ===
link_directories(
  ${TENSORRT_LIB_DIR}
  /usr/local/cuda-11.8/lib64
)

# === OpenVINO 版本 ===
if(BUILD_WITH_OPENVINO)
  set(SOURCES
    ${COMMON_SOURCES}
  )
  list(APPEND SOURCES ${OPENVINO_SOURCES})

  if(BUILD_WITH_NCNN)
    list(APPEND SOURCES ${NCNN_SOURCES})
  endif()
  add_executable(${PROJECT_NAME}_openvino ${SOURCES})
  if(BUILD_WITH_NCNN)
    target_compile_definitions(${PROJECT_NAME}_openvino PRIVATE USE_OPENVINO USE_NCNN)
  else()
    target_compile_definitions(${PROJECT_NAME}_openvino PRIVATE USE_OPENVINO)
  endif()

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
  if(BUILD_WITH_NCNN)
    target_link_libraries(${PROJECT_NAME}_openvino
    ncnn
  )
  endif()
endif()

# === TensorRT 版本 ===
if(BUILD_WITH_TRT)
  set(SOURCES
    ${COMMON_SOURCES}
  )
  list(APPEND SOURCES ${TRT_SOURCES})

  if(BUILD_WITH_NCNN)
    list(APPEND SOURCES ${NCNN_SOURCES})
  endif()
  add_executable(${PROJECT_NAME}_trt ${SOURCES})
  
  if(BUILD_WITH_NCNN)
    target_compile_definitions(${PROJECT_NAME}_trt PRIVATE USE_TRT USE_NCNN)
  else()
    target_compile_definitions(${PROJECT_NAME}_trt PRIVATE USE_TRT)
  endif()

  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    target_link_directories(${PROJECT_NAME}_trt PUBLIC hikSDK/lib/amd64)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    target_link_directories(${PROJECT_NAME}_trt PUBLIC hikSDK/lib/arm64)
  endif()

  target_link_libraries(${PROJECT_NAME}_trt
    ${OpenCV_LIBS}
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
  if(BUILD_WITH_NCNN)
    target_link_libraries(${PROJECT_NAME}_trt
    ncnn
  )
endif()
endif()
# === NCNN only 版本 ===
if(BUILD_WITH_NCNN)
  set(SOURCES
    ${COMMON_SOURCES}
  )
  list(APPEND SOURCES ${NCNN_SOURCES})
  add_executable(${PROJECT_NAME}_ncnn ${SOURCES})
  target_compile_definitions(${PROJECT_NAME}_ncnn PRIVATE USE_NCNN_ONLY USE_NCNN)


  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    target_link_directories(${PROJECT_NAME}_ncnn PUBLIC hikSDK/lib/amd64)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    target_link_directories(${PROJECT_NAME}_ncnn PUBLIC hikSDK/lib/arm64)
  endif()

  target_link_libraries(${PROJECT_NAME}_ncnn
    ${OpenCV_LIBS}
    fmt::fmt
    yaml-cpp
    ${Python3_LIBRARIES}
    g2o_core g2o_stuff g2o_solver_csparse g2o_types_sba g2o_types_slam3d g2o_solver_dense
    ${OPENGL_gl_LIBRARY}
    ${CERES_LIBRARIES}
    FormatConversion MediaProcess MvCameraControl MVRender MvUsb3vTL
    GL
    tbb 
    ncnn
  )
  
endif()