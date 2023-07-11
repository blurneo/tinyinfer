#pragma once

#include <iostream>
#include <chrono>

#define CHECK(x, expected, str) \
    if ((x) != expected) { \
        std::cerr << "CHECK Failed: " << str << "\n"; \
    }

#define CHECK_BOOL_RET(x, expected, str) \
    if ((x) != expected) { \
        std::cerr << "CHECK Ret Failed: " << str << "\n"; \
        return false; \
    }

#define CHECK_RET(x, expected, ret, str) \
    if ((x) != expected) { \
        std::cerr << "CHECK Int Ret Failed: " << str << "\n"; \
        return ret; \
    }

#define __TIC__(mark) \
    auto begin_##mark = std::chrono::high_resolution_clock::now();

#define __TOC__(mark) \
    auto end_##mark = std::chrono::high_resolution_clock::now(); \
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end_##mark - begin_##mark); \
    printf("%s time measured: %.3f ms.\n", #mark, elapsed.count() * 1e-6);


