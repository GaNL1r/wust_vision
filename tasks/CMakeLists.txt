add_library(tasks_common
    ${CMAKE_CURRENT_SOURCE_DIR}/debug.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mono_measure_tool.cpp
)

target_link_libraries(auto_aim PRIVATE tasks_common)
target_link_libraries(auto_buff PRIVATE tasks_common)
