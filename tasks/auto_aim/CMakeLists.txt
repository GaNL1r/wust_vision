cmake_minimum_required(VERSION 3.14)
project(auto_aim LANGUAGES CXX )

if(BUILD_WITH_OPENVINO)
  list(APPEND BUILD_LIBS openvino::runtime openvino::frontend::onnx)
endif()
if(BUILD_WITH_TRT)
  list(APPEND BUILD_LIBS TensorRT::TensorRT TensorRT::nvonnxparser  CUDA::cudart  cuda_infer)
endif()
if(BUILD_WITH_NCNN)
  list(APPEND BUILD_LIBS ncnn)
endif()
if(BUILD_WITH_ORT)
  list(APPEND BUILD_LIBS ${Ort_LIB})
endif()
add_subdirectory(armor_control/tinympc)
list(APPEND BUILD_LIBS tinympcstatic   ${CERES_LIBRARIES} ${WUST_VL_LIBS} )

add_library(auto_aim OBJECT)

target_compile_options(auto_aim PRIVATE -w)
target_compile_definitions(auto_aim PRIVATE ${BUILD_DEFINITIONS})
target_link_libraries(auto_aim PRIVATE
    project_options
)
target_include_directories(auto_aim PRIVATE
    include
    $<$<BOOL:${BUILD_WITH_OPENVINO}>:${OpenVINO_INCLUDE_DIRS}>
    $<$<BOOL:${BUILD_WITH_TRT}>:${TensorRT_INCLUDE_DIR}>
    $<$<BOOL:${BUILD_WITH_NCNN}>:${ncnn_INCLUDE_DIRS}>
    $<$<BOOL:${BUILD_WITH_ORT}>:${Ort_INCLUDE_DIR}>
)

target_sources(auto_aim PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/auto_aim.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/type.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/armor_control/very_aimer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/armor_control/very_aimer_opt.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/armor_control/very_aimer_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/armor_tracker/tracker_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/armor_tracker/trackerv3.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/armor_tracker/target.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/armor_optimize/ba_solver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/armor_detect/opencv/armor_detector_opencv_wrapper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/armor_detect/opencv/armor_detector_opencv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/armor_detect/armor_detect_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/armor_detect/armor_infer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/armor_detect/armor_pose_estimator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/armor_detect/light_corner_corrector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/armor_detect/number_classifier/number_classifier.cpp
)
target_include_directories(auto_aim PUBLIC
    ${WUST_VL_INCLUDE_DIRS}
    ${HikSDK_INCLUDE_DIR}
)

if(BUILD_WITH_OPENVINO)
    target_sources(auto_aim PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/armor_detect/openvino/armor_detector_openvino.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/armor_detect/openvino/armor_detector_openvino_wrapper.cpp
    )
endif()

if(BUILD_WITH_TRT)
    target_sources(auto_aim PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/armor_detect/tensorrt/armor_detector_tensorrt.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/armor_detect/tensorrt/armor_detector_tensorrt_wrapper.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/armor_detect/number_classifier/number_classifier_trt.cpp
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
        ${CMAKE_CURRENT_SOURCE_DIR}/armor_detect/onnxruntime/armor_detector_onnxruntime.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/armor_detect/onnxruntime/armor_detector_onnxruntime_wrapper.cpp
    )
endif()
set(WUST_AUTO_AIM_LIBS ${BUILD_LIBS} PARENT_SCOPE)