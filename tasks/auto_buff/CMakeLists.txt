cmake_minimum_required(VERSION 3.14)
project(auto_buff LANGUAGES CXX)
set(BUILD_DEFINITIONS "")
list(APPEND BUILD_LIBS  TBB::tbb ${G2O_LIBRARIES} g2o_solver_dense ${CERES_LIBRARIES})
add_subdirectory(rune_detector/rm_vision_core)
VisCore_add_module(auto_buff
DEPENDS rune_detector
)
list(APPEND BUILD_LIBS  VisCore_rune_detector
    VisCore_core
    VisCore_detector
    VisCore_dataio
    VisCore_rune_center
    VisCore_rune_target
    VisCore_rune_fan
    VisCore_rune_combo
    VisCore_rune_tracker
    VisCore_rune_group
)
target_compile_options(VisCore_auto_buff PRIVATE -w)
target_compile_definitions(VisCore_auto_buff PRIVATE ${BUILD_DEFINITIONS})
target_sources(VisCore_auto_buff PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/auto_buff.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rune_control/aimer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rune_optimize/graph_optimizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rune_optimize/ba_solver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rune_tracker/rune_tracker.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rune_tracker/rune_target.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rune_detector/scut_robot_detector.cpp
    
)


set(WUST_AUTO_BUFF_LIBS ${BUILD_LIBS} PARENT_SCOPE)
