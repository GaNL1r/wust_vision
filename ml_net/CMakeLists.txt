cmake_minimum_required(VERSION 3.14)
project(ml_net LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE "Release")

# -------------------
# 可选后端开关
# -------------------
option(ML_NET_USE_OPENVINO "Enable OpenVINO backend" ON)
option(ML_NET_USE_TRT      "Enable TensorRT backend" ON)
option(ML_NET_USE_ORT      "Enable ONNX Runtime backend" ON)
option(ML_NET_USE_NCNN     "Enable NCNN backend" ON)

# -------------------
# 公共 include
# -------------------
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# -------------------
# 公共源码
# -------------------
set(ML_NET_SOURCES
    # 公共源码，如果有的话，可以放这里
)

# -------------------
# 后端源码，根据选项添加
# -------------------
if(ML_NET_USE_OPENVINO)
    list(APPEND ML_NET_SOURCES
        src/openvino/openvino_net.cpp
    )
endif()

if(ML_NET_USE_TRT)
    list(APPEND ML_NET_SOURCES
        src/tensorrt/tensorrt_net.cpp
    )
endif()

if(ML_NET_USE_ORT)
    list(APPEND ML_NET_SOURCES
        src/onnxruntime/onnxruntime_net.cpp
    )
endif()

if(ML_NET_USE_NCNN)
    list(APPEND ML_NET_SOURCES
        src/ncnn/ncnn_net.cpp
    )
endif()

# -------------------
# 创建静态库
# -------------------
add_library(ml_net STATIC ${ML_NET_SOURCES})

# 暴露 include
target_include_directories(ml_net PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# -------------------
# 后端依赖
# -------------------
if(ML_NET_USE_OPENVINO)
    find_package(OpenVINO REQUIRED COMPONENTS Runtime ONNX)
    target_link_libraries(ml_net PUBLIC openvino::runtime openvino::frontend::onnx)
    target_compile_definitions(ml_net PUBLIC USE_OPENVINO)
endif()

if(ML_NET_USE_TRT)
    find_package(CUDAToolkit REQUIRED)
    # 手动设置 TensorRT 路径或通过 find_package(TensorRT)
    target_include_directories(ml_net PUBLIC ${TensorRT_INCLUDE_DIR})
    target_link_libraries(ml_net PUBLIC TensorRT::TensorRT CUDA::cudart)
    target_compile_definitions(ml_net PUBLIC USE_TRT)
endif()

if(ML_NET_USE_ORT)
    find_package(Ort REQUIRED)
    target_include_directories(ml_net PUBLIC ${Ort_INCLUDE_DIR})
    target_link_libraries(ml_net PUBLIC ${Ort_LIB})
    target_compile_definitions(ml_net PUBLIC USE_ORT)
endif()

if(ML_NET_USE_NCNN)
    find_package(ncnn REQUIRED)
    target_include_directories(ml_net PUBLIC ${ncnn_INCLUDE_DIRS})
    target_link_libraries(ml_net PUBLIC ncnn)
    target_compile_definitions(ml_net PUBLIC USE_NCNN)
endif()

# -------------------
# 安装
# -------------------
install(TARGETS ml_net
    EXPORT ml_netTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/ DESTINATION include)

install(EXPORT ml_netTargets
    FILE ml_netTargets.cmake
    NAMESPACE ml_net::
    DESTINATION lib/cmake/ml_net
)
