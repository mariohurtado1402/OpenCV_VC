cmake_minimum_required(VERSION 3.10)
project(metrics_service)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(nlohmann_json REQUIRED)

add_executable(metrics_service main.cpp)
target_include_directories(metrics_service PRIVATE third_party)
target_link_libraries(metrics_service PRIVATE nlohmann_json::nlohmann_json)
