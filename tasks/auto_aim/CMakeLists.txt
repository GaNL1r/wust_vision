cmake_minimum_required(VERSION 3.14)
project(auto_aim LANGUAGES CXX CUDA)


find_package(OpenCV 4.11 REQUIRED)
find_package(Eigen3      REQUIRED)
find_package(fmt         REQUIRED)
find_package(yaml-cpp    REQUIRED)
find_package(G2O         REQUIRED)
find_package(OpenGL      REQUIRED)


set(BUILD_DEFINITIONS "")

if(BUILD_WITH_OPENVINO)
  list(APPEND BUILD_DEFINITIONS USE_OPENVINO)
  list(APPEND BUILD_LIBS openvino::runtime openvino::frontend::onnx)
endif()
if(BUILD_WITH_TRT)
  list(APPEND BUILD_DEFINITIONS USE_TRT)
  list(APPEND BUILD_LIBS TensorRT::TensorRT CUDA::cudart /home/hy/TensorRT-8.5.2.2/lib/libnvonnxparser.so cuda_infer)
endif()
if(BUILD_WITH_NCNN)
  list(APPEND BUILD_DEFINITIONS USE_NCNN)
  list(APPEND BUILD_LIBS ncnn)
endif()
if(BUILD_WITH_ORT)
  list(APPEND BUILD_DEFINITIONS USE_ORT)
  list(APPEND BUILD_LIBS ${Ort_LIB})
endif()

# ================== 创建目标库 ==================
add_library(auto_aim OBJECT)

target_compile_options(auto_aim PRIVATE -w)
target_compile_definitions(auto_aim PRIVATE ${BUILD_DEFINITIONS})
target_include_directories(auto_aim PRIVATE
    include
    $<$<BOOL:${BUILD_WITH_OPENVINO}>:${OpenVINO_INCLUDE_DIRS}>
    $<$<BOOL:${BUILD_WITH_TRT}>:${TensorRT_INCLUDE_DIR}>
    $<$<BOOL:${BUILD_WITH_NCNN}>:${ncnn_INCLUDE_DIRS}>
    $<$<BOOL:${BUILD_WITH_ORT}>:${Ort_INCLUDE_DIR}>
)
# ================== 公共源文件 ==================
target_sources(auto_aim PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/auto_aim.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/aimer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shooter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/armor_tracker/one_ca_tracker.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/armor_tracker/one_tracker.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/armor_tracker/tracker_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/armor_tracker/tracker.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/armor_tracker/trackerv2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/armor_tracker/trackerv3.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/armor_tracker/target.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/armor_optimize/ba_solver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/armor_optimize/graph_optimizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/armor_detect/opencv/armor_detector_opencv_wrapper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/armor_detect/opencv/armor_detector_opencv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/armor_detect/armor_detect_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/armor_detect/armor_infer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/armor_detect/armor_pose_estimator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/armor_detect/light_corner_corrector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/armor_detect/number_classifier.cpp
)

# ================== 后端源码 ==================
if(BUILD_WITH_OPENVINO)
    target_sources(auto_aim PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/armor_detect/openvino/armor_detector_openvino.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/armor_detect/openvino/armor_detector_openvino_wrapper.cpp
    )
endif()

if(BUILD_WITH_TRT)
    target_sources(auto_aim PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/armor_detect/tensorrt/armor_detector_trt.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/armor_detect/tensorrt/armor_detector_trt_wrapper.cpp
    )
endif()

if(BUILD_WITH_NCNN)
    target_sources(auto_aim PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/armor_detect/ncnn/armor_detector_ncnn.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/armor_detect/ncnn/armor_detector_ncnn_wrapper.cpp
    )
endif()

if(BUILD_WITH_ORT)
    target_sources(auto_aim PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/armor_detect/onnxruntime/armor_detector_ort.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/armor_detect/onnxruntime/armor_detector_ort_wrapper.cpp
    )
endif()
set(WUST_AUTO_AIM_LIBS ${BUILD_LIBS} PARENT_SCOPE)