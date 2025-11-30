cmake_minimum_required(VERSION 3.10)
project(drive_cpp_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CURL REQUIRED)
find_package(nlohmann_json REQUIRED)

add_executable(drive_cpp_test main.cpp)

target_link_libraries(drive_cpp_test
  PRIVATE
  CURL::libcurl
  nlohmann_json::nlohmann_json
)
