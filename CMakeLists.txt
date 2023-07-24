cmake_minimum_required(VERSION 3.16)

project(TinyInfer)

if (TI_ENABLE_SANITIZER)
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif()

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tinyinfer)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -msse3 -funroll-loops")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g")

file(GLOB_RECURSE srcs
    ${PROJECT_SOURCE_DIR}/tinyinfer/common/*.h
    ${PROJECT_SOURCE_DIR}/tinyinfer/common/*.cpp
    ${PROJECT_SOURCE_DIR}/tinyinfer/layer/*.cpp
    ${PROJECT_SOURCE_DIR}/tinyinfer/layer/*.cpp
    ${PROJECT_SOURCE_DIR}/tinyinfer/net/*.h
    ${PROJECT_SOURCE_DIR}/tinyinfer/net/*.cpp
    ${PROJECT_SOURCE_DIR}/tinyinfer/pp/im2col/*.h
    ${PROJECT_SOURCE_DIR}/tinyinfer/pp/im2col/*.cpp
    ${PROJECT_SOURCE_DIR}/tinyinfer/pp/gemm/*.h
    ${PROJECT_SOURCE_DIR}/tinyinfer/pp/gemm/*.cpp
)
add_library(ti STATIC ${srcs})

add_executable(mnist_unittest tests/mnist_unittest.cpp)
target_link_libraries(mnist_unittest PRIVATE ti)

add_executable(gemm_pp_unittest tests/gemm_pp_unittest.cpp)
target_link_libraries(gemm_pp_unittest PRIVATE ti)
