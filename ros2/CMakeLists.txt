find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

add_library(toros2 STATIC)

target_sources(toros2 PRIVATE 
  ${CMAKE_CURRENT_SOURCE_DIR}/ros2.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ros2_pub_node.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ros2_sub_node.cpp
)

# 关键：用 ament_target_dependencies 来引入依赖
ament_target_dependencies(toros2
  rclcpp
  std_msgs
  geometry_msgs
)

# 如果是库，还要导出依赖，防止下游链接失败
ament_export_dependencies(rclcpp std_msgs geometry_msgs)
ament_export_targets(toros2)
