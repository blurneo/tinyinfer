include_directories(${PROJECT_SOURCE_DIR})
add_library(tflow
    ${PROJECT_SOURCE_DIR}/flow/optical_flow.h
    ${PROJECT_SOURCE_DIR}/flow/optical_flow.cpp
    ${PROJECT_SOURCE_DIR}/flow/resize/fast_resize.h
    ${PROJECT_SOURCE_DIR}/flow/resize/fast_resize.cpp
    ${PROJECT_SOURCE_DIR}/flow/matrix.h)
