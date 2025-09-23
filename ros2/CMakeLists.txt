find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)

add_library(toros2 STATIC)

target_sources(toros2 PRIVATE 
  ${CMAKE_CURRENT_SOURCE_DIR}/ros2.cpp
)


ament_target_dependencies(toros2
  rclcpp
)

ament_export_dependencies(rclcpp)
ament_export_targets(toros2)
