#pragma once

#include <iostream>

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

