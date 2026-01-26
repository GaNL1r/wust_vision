add_library(tasks_common
    ${CMAKE_CURRENT_SOURCE_DIR}/debug.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/type_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/vision_base.cpp
)
target_include_directories(tasks_common PRIVATE ${WUST_VL_INCLUDE_DIRS} ${HikSDK_INCLUDE_DIR})
target_link_libraries(tasks_common PRIVATE
    project_options pch_core
)
add_library(utils
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
)
target_link_libraries(utils PRIVATE
    project_options pch_core
)
target_include_directories(utils PRIVATE ${WUST_VL_INCLUDE_DIRS} ${HikSDK_INCLUDE_DIR})