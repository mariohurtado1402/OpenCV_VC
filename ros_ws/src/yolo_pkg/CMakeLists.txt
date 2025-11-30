cmake_minimum_required(VERSION 3.10)
project(yolo_pkg)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ayuda a CMake a encontrar la instalación local de ncnn si no viene en el ambiente
# Forzamos la ruta local de ncnn (ajusta si instalas en otro lado)
set(ncnn_DIR "/home/mario/ncnn/build/install/lib/cmake/ncnn"
    CACHE PATH "Path to ncnnConfig.cmake" FORCE)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(ncnn REQUIRED)
find_package(ament_cmake_python REQUIRED)

add_executable(yolo_main src/main.cpp)

target_link_libraries(yolo_main OpenMP::OpenMP_CXX ncnn)

ament_target_dependencies(yolo_main
  rclcpp
  std_msgs
  OpenCV
)

install(TARGETS yolo_main
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY models/
  DESTINATION share/${PROJECT_NAME}/models
  FILES_MATCHING PATTERN "*.bin" PATTERN "*.param"
)

install(DIRECTORY runs/
  DESTINATION share/${PROJECT_NAME}/runs
)

ament_package()
