cmake_minimum_required(VERSION 3.14)
cmake_policy(SET CMP0072 NEW)
project(wust_vision LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_BUILD_TYPE "Release")
message(STATUS "--------------------CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}--------------------")


option(BUILD_WITH_TRT      "Enable TensorRT backend"   ON)
option(BUILD_WITH_OPENVINO "Enable OpenVINO backend"   ON)
option(BUILD_WITH_NCNN     "Enable NCNN backend"       OFF)
option(BUILD_WITH_ORT      "Enable ORT backend"        ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)


file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS tasks/*.cpp)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Ceres REQUIRED)
find_package(HikSDK REQUIRED)
find_package(wust_vl REQUIRED)

set(BUILD_DEFINITIONS "")
set(BUILD_LIBS "")


if(BUILD_WITH_OPENVINO)
    find_package(OpenVINO REQUIRED COMPONENTS Runtime ONNX)
    list(APPEND BUILD_DEFINITIONS USE_OPENVINO)
    list(APPEND BUILD_LIBS openvino::runtime openvino::frontend::onnx)
endif()

if(BUILD_WITH_TRT)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/home/hy/TensorRT-10.6.0.26")
    find_package(TensorRT REQUIRED)
    include_directories(${TensorRT_INCLUDE_DIR})
    include_directories(/usr/local/cuda/include)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
    set(CMAKE_CUDA_ARCHITECTURES 86)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_COMPILER_LAUNCHER ccache)
    add_subdirectory(${CMAKE_SOURCE_DIR}/cuda_infer ${CMAKE_BINARY_DIR}/cuda_infer_build)
    list(APPEND BUILD_DEFINITIONS USE_TRT)
    list(APPEND BUILD_LIBS TensorRT::TensorRT TensorRT::nvonnxparser CUDA::cudart cuda_infer)
endif()

if(BUILD_WITH_NCNN)
    find_package(ncnn REQUIRED)
    list(APPEND BUILD_DEFINITIONS USE_NCNN)
    list(APPEND BUILD_LIBS ncnn)
endif()

if(BUILD_WITH_ORT)
    set(ort_root_path "/home/hy/onnxruntime-linux-x64-gpu-1.22.0")
    find_package(Ort REQUIRED)
    include_directories(${Ort_INCLUDE_DIR})
    list(APPEND BUILD_DEFINITIONS USE_ORT)
    list(APPEND BUILD_LIBS ${Ort_LIB})
endif()


add_library(${PROJECT_NAME} STATIC ${SOURCES})
target_compile_definitions(${PROJECT_NAME} PUBLIC ${BUILD_DEFINITIONS})
target_include_directories(${PROJECT_NAME} PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}
    wust_vl::wust_vl
    HikSDK::HikSDK
    yaml-cpp
    fmt::fmt
    Eigen3::Eigen
    Ceres::ceres
    ${OpenCV_LIBS}
    ${BUILD_LIBS}
)
set(ROS2_PACKAGES
    ament_cmake
    rosidl_typesupport_cpp
    rclcpp
    geometry_msgs
    tf2_ros
    tf2_geometry_msgs
    sensor_msgs
    visualization_msgs
    sentry_interfaces
    nav_msgs
)



function(add_common_executable exe_name src_file)
    add_executable(${exe_name} ${src_file})
    target_link_libraries(${exe_name}  ${PROJECT_NAME})
endfunction()
add_common_executable(standard src/standard.cpp)
# add_common_executable(dart src/dart.cpp)
add_common_executable(test_usbcamera test/test_usbcamera.cpp)


foreach(pkg IN LISTS ROS2_PACKAGES)
    find_package(${pkg} QUIET)
endforeach()


set(ROS2_FULL_FOUND TRUE)
foreach(pkg IN LISTS ROS2_PACKAGES)
    if(NOT ${pkg}_FOUND)
        set(ROS2_FULL_FOUND FALSE)
        message(WARNING "ROS2 package ${pkg} not found")
    endif()
endforeach()

if(ROS2_FULL_FOUND)
    message(STATUS "ROS2 full environment found, compiling ROS2 targets ...")
    list(APPEND BUILD_DEFINITIONS USE_ROS2)

    set(ROS2_INCLUDES)
    foreach(pkg IN LISTS ROS2_PACKAGES)
        if(TARGET ${pkg}::${pkg})
            list(APPEND ROS2_INCLUDES ${${pkg}_INCLUDE_DIRS})
        endif()
    endforeach()
    include_directories(${ROS2_INCLUDES})
    find_package(Open3D REQUIRED)

    target_link_libraries(${PROJECT_NAME} 
        Open3D::Open3D 
    )
    ament_target_dependencies(${PROJECT_NAME}  ${ROS2_PACKAGES})
    target_compile_definitions(${PROJECT_NAME}  PUBLIC ${BUILD_DEFINITIONS})
    macro(add_ros2_executable exe_name src_file)
        add_executable(${exe_name} ${src_file})
        target_link_libraries(${exe_name} ${PROJECT_NAME} Open3D::Open3D )
    endmacro()


    add_ros2_executable(sentry src/sentry.cpp)
    add_ros2_executable(hero src/hero.cpp)
    # add_ros2_executable(sim src/sim.cpp)
    add_ros2_executable(nav test/nav.cpp)
else()
    message(WARNING "ROS2 dependencies incomplete, skipping ROS2 targets ...")
endif()

