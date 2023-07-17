file(GLOB srcs
    ${PROJECT_SOURCE_DIR}/tinyinfer/layer/pp/*.cpp
    ${PROJECT_SOURCE_DIR}/tinyinfer/layer/pp/*.h
)
add_executable(matmul_pp_unittest
    ${PROJECT_SOURCE_DIR}/tinyinfer/layer/pp/matmul/matmul_pp_unittest.cpp)
target_link_libraries(matmul_pp_unittest PRIVATE ti)
