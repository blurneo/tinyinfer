file(GLOB srcs
    ${PROJECT_SOURCE_DIR}/tinyinfer/layer/pp/*.cpp
    ${PROJECT_SOURCE_DIR}/tinyinfer/layer/pp/*.h
)
add_executable(gemm_pp_unittest
    ${PROJECT_SOURCE_DIR}/tinyinfer/layer/pp/gemm/gemm_pp_unittest.cpp)
target_link_libraries(gemm_pp_unittest PRIVATE ti)
