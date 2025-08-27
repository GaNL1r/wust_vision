cmake_minimum_required(VERSION 3.14)
project(wust_vision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE "Release")
add_compile_options(-O3 -w)
option(BUILD_WITH_TRT      "Enable TensorRT backend"   ON)
option(BUILD_WITH_OPENVINO "Enable OpenVINO backend"   ON)
option(BUILD_WITH_NCNN     "Enable NCNN backend"       ON)
option(BUILD_WITH_ORT      "Enable ORT backend"        ON)
include_directories(${PROJECT_SOURCE_DIR})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
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
find_package(TBB         REQUIRED)
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

    add_subdirectory(${CMAKE_SOURCE_DIR}/cuda_infer ${CMAKE_BINARY_DIR}/cuda_infer_build)
    include_directories("cuda_infer")
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


add_subdirectory(tasks/auto_aim)
add_subdirectory(tasks/auto_buff)
add_subdirectory(tasks)
add_executable(test src/auto_aim.cpp)

target_include_directories(test PRIVATE ${WUST_VL_INCLUDE_DIRS} ${HikSDK_INCLUDE_DIR})
target_link_libraries(test
    $<TARGET_OBJECTS:auto_aim>
    ${WUST_AUTO_AIM_LIBS}
    $<TARGET_OBJECTS:auto_buff>
    ${WUST_AUTO_BUFF_LIBS}
    tasks_common
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    ${WUST_VL_LIBS}
    yaml-cpp
    TBB::tbb
    ${G2O_LIBRARIES} g2o_solver_dense
    ${HikSDK_LIB}
    fmt::fmt
)