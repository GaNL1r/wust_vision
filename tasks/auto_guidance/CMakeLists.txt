cmake_minimum_required(VERSION 3.14)
project(auto_guidance LANGUAGES CXX )

if(BUILD_WITH_OPENVINO)
  list(APPEND BUILD_DEFINITIONS USE_OPENVINO)
  list(APPEND BUILD_LIBS openvino::runtime openvino::frontend::onnx)
endif()


list(APPEND BUILD_LIBS  yaml-cpp)

add_library(auto_guidance OBJECT)
if(BUILD_WITH_OPENVINO)
    target_sources(auto_guidance PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/guidance_detector/openvino/guidance_detector_openvino.cpp
    )
endif()
target_compile_options(auto_guidance PRIVATE -w)
target_compile_definitions(auto_guidance PRIVATE ${BUILD_DEFINITIONS})
target_include_directories(auto_guidance PRIVATE
    include
    $<$<BOOL:${BUILD_WITH_OPENVINO}>:${OpenVINO_INCLUDE_DIRS}>
)

target_sources(auto_guidance PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/auto_guidance.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/guidance_detector/green_light_infer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/guidance_detector/opencv/guidance_detector_opencv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/guidance_tracker/guidance_tracker.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/guidance_tracker/guidance_target.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/debug.cpp
)





set(WUST_auto_guidance_LIBS ${BUILD_LIBS} PARENT_SCOPE)