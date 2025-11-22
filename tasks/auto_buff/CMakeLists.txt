cmake_minimum_required(VERSION 3.14)
project(auto_buff LANGUAGES CXX)
set(BUILD_DEFINITIONS "")
list(APPEND BUILD_LIBS  TBB::tbb ${G2O_LIBRARIES} g2o_solver_dense ${CERES_LIBRARIES})
add_library(auto_buff OBJECT)
target_compile_options(auto_buff PRIVATE -w)
target_compile_definitions(auto_buff PRIVATE ${BUILD_DEFINITIONS})
target_sources(auto_buff PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/auto_buff.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rune_control/aimer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rune_optimize/graph_optimizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rune_optimize/ba_solver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rune_tracker/rune_tracker.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rune_tracker/rune_target.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rune_detector/rune_detector.cpp
)


set(WUST_AUTO_BUFF_LIBS ${BUILD_LIBS} PARENT_SCOPE)
