#pragma once

#include <chrono>
#include <cmath>
#include <iostream>

#define CHECK(x, expected, str)                   \
  if ((x) != expected)                            \
  {                                               \
    std::cerr << "CHECK Failed: " << str << "\n"; \
  }

#define CHECK_BOOL_RET(x, expected, str)              \
  if ((x) != expected)                                \
  {                                                   \
    std::cerr << "CHECK Ret Failed: " << str << "\n"; \
    return false;                                     \
  }

#define CHECK_RET(x, expected, ret, str)                  \
  if ((x) != expected)                                    \
  {                                                       \
    std::cerr << "CHECK Int Ret Failed: " << str << "\n"; \
    return ret;                                           \
  }

#define CHECK_VEC_EQUAL_RET(x, y, ret, str)                                         \
  {                                                                                 \
    if (x.size() != y.size())                                                       \
    {                                                                               \
      std::cerr << "CHECK Vec Equal Size not matched Failed: " << str << "\n";      \
      return ret;                                                                   \
    }                                                                               \
    int cnt = x.size();                                                             \
    for (int i = 0; i < cnt; i++)                                                   \
    {                                                                               \
      if (std::fabs(x[i] - y[i]) > 0.0000001)                                       \
      {                                                                             \
        std::cerr << "CHECK Vec Equal Ret Failed at: " << i << ", " << str << "\n"; \
        return ret;                                                                 \
      }                                                                             \
    }                                                                               \
  }

#define __TIC__(mark) \
  auto begin_##mark = std::chrono::high_resolution_clock::now();

#define __TOC__(mark)                                                      \
  auto end_##mark = std::chrono::high_resolution_clock::now();             \
  auto elapsed_ms_##mark =                                                 \
      (std::chrono::duration_cast<std::chrono::nanoseconds>(end_##mark -   \
                                                            begin_##mark)) \
          .count() *                                                       \
      1e-6;                                                                \
  printf("%s time measured: %.3f ms.\n", #mark, elapsed_ms_##mark);

#define __TIME_IN_MS__(mark) elapsed_ms_##mark
