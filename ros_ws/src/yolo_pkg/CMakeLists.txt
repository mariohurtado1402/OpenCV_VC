cmake_minimum_required(VERSION 3.10)
project(yolo_pkg)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(ncnn REQUIRED)

add_executable(yolo_main src/main.cpp)

target_link_libraries(yolo_main
  PRIVATE
  ${OpenCV_LIBS}
  OpenMP::OpenMP_CXX
  ncnn
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
