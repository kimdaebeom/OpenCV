cmake_minimum_required(VERSION 2.8)
project(OpenCV_test)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(OpenCV_test main.cpp)
target_link_libraries(OpenCV_test ${OpenCV_LIBS})
