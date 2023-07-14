cmake_minimum_required(VERSION 3.20)

project(TinyInfer)

if (TI_ENABLE_SANITIZER)
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif()

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tinyinfer)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

file(GLOB_RECURSE srcs
    ${PROJECT_SOURCE_DIR}/tinyinfer/common/*.h
    ${PROJECT_SOURCE_DIR}/tinyinfer/common/*.cpp
    ${PROJECT_SOURCE_DIR}/tinyinfer/layer/*.cpp
    ${PROJECT_SOURCE_DIR}/tinyinfer/layer/*.cpp
    ${PROJECT_SOURCE_DIR}/tinyinfer/net/*.cpp
    ${PROJECT_SOURCE_DIR}/tinyinfer/net/*.cpp
)
add_library(ti STATIC ${srcs})
add_executable(mnist_test tinyinfer/net/mnist.cpp)
target_link_libraries(mnist_test PRIVATE ti)
