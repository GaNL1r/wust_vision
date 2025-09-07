find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)

add_library(toros2 STATIC)

target_sources(toros2 PRIVATE 
  ${CMAKE_CURRENT_SOURCE_DIR}/ros2.cpp
)

# 关键：用 ament_target_dependencies 来引入依赖
ament_target_dependencies(toros2
  rclcpp
)

# 如果是库，还要导出依赖，防止下游链接失败
ament_export_dependencies(rclcpp)
ament_export_targets(toros2)
