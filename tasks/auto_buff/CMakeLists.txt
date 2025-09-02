cmake_minimum_required(VERSION 3.14)
project(auto_buff LANGUAGES CXX)


find_package(OpenCV  REQUIRED)
find_package(Eigen3      REQUIRED)
find_package(fmt         REQUIRED)
find_package(yaml-cpp    REQUIRED)

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
add_library(auto_buff OBJECT)

target_compile_options(auto_buff PRIVATE -w)
target_compile_definitions(auto_buff PRIVATE ${BUILD_DEFINITIONS})
target_include_directories(auto_buff PRIVATE
    include
    $<$<BOOL:${BUILD_WITH_OPENVINO}>:${OpenVINO_INCLUDE_DIRS}>
    $<$<BOOL:${BUILD_WITH_TRT}>:${TensorRT_INCLUDE_DIR}>
    $<$<BOOL:${BUILD_WITH_NCNN}>:${ncnn_INCLUDE_DIRS}>
    $<$<BOOL:${BUILD_WITH_ORT}>:${Ort_INCLUDE_DIR}>
)
# ================== 公共源文件 ==================
target_sources(auto_buff PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/auto_buff.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rune_detect/rune_infer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rune_solver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/curve_fitter.cpp
)

# ================== 后端源码 ==================
if(BUILD_WITH_OPENVINO)
    target_sources(auto_buff PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/rune_detect/openvino/rune_detector_openvino.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/rune_detect/openvino/rune_detector_openvino_wrapper.cpp
    )
endif()

if(BUILD_WITH_TRT)
    target_sources(auto_buff PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/rune_detect/tensorrt/rune_detector_trt.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/rune_detect/tensorrt/rune_detector_trt_wrapper.cpp
    )
endif()

if(BUILD_WITH_NCNN)
    target_sources(auto_buff PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/rune_detect/ncnn/rune_detector_ncnn.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/rune_detect/ncnn/rune_detector_ncnn_wrapper.cpp
    )
endif()

if(BUILD_WITH_ORT)
    target_sources(auto_buff PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/rune_detect/onnxruntime/rune_detector_ort.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/rune_detect/onnxruntime/rune_detector_ort_wrapper.cpp
    )
endif()
set(WUST_AUTO_BUFF_LIBS ${BUILD_LIBS} PARENT_SCOPE)