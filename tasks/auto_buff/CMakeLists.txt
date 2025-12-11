cmake_minimum_required(VERSION 3.14)
project(auto_buff LANGUAGES CXX)
set(BUILD_DEFINITIONS "")
list(APPEND BUILD_LIBS  TBB::tbb  ${CERES_LIBRARIES} ${WUST_VL_LIBS})
add_library(auto_buff OBJECT)
target_compile_options(auto_buff PRIVATE -w)
target_compile_definitions(auto_buff PRIVATE ${BUILD_DEFINITIONS})
target_sources(auto_buff PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/auto_buff.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rune_control/aimer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rune_optimize/ba_solver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rune_tracker/rune_tracker.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rune_tracker/rune_target.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rune_detector/rune_detector.cpp
)
target_include_directories(auto_buff PUBLIC
    ${WUST_VL_INCLUDE_DIRS}
    ${HikSDK_INCLUDE_DIR}
)

set(WUST_AUTO_BUFF_LIBS ${BUILD_LIBS} PARENT_SCOPE)
