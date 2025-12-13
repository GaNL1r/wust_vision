add_library(tasks_common
    ${CMAKE_CURRENT_SOURCE_DIR}/debug.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/type_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/vision_base.cpp
)
target_include_directories(tasks_common PRIVATE ${WUST_VL_INCLUDE_DIRS} ${HikSDK_INCLUDE_DIR})

add_library(utils
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
)
target_include_directories(utils PRIVATE ${WUST_VL_INCLUDE_DIRS} ${HikSDK_INCLUDE_DIR})