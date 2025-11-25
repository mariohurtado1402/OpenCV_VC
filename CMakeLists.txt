cmake_minimum_required(VERSION 3.10)
project(yolo_ncnn_rpi)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Where NCNN CMake config is
set(ncnn_DIR "/home/mario/ncnn/build/install/lib/cmake/ncnn")

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(ncnn REQUIRED)

add_executable(main main.cpp)

target_link_libraries(main
    PRIVATE
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
    ncnn
)

