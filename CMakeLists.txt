cmake_minimum_required(VERSION 3.26)

project(TinyInfer)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tinyinfer)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

add_library(ti STATIC tinyinfer/net/net.cpp tinyinfer/net/graph.cpp)
add_executable(mnist_test tinyinfer/net/mnist.cpp)
target_link_libraries(mnist_test PRIVATE ti)
