#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include "tinyinfer/common/check_macro.h"
#include <mmintrin.h>
#include <emmintrin.h>

int main()
{
    uint64_t loop = 1024 * 1024 * 1024;
    __TIC__(LOOP)
    for (int i = 0; i < loop; i++)
    {
        __asm__("addps %xmm0, %xmm0;"
                "addps %xmm1, %xmm1;"
                "addps %xmm2, %xmm2;"
                "addps %xmm3, %xmm3;"
                "addps %xmm4, %xmm4;"
                "addps %xmm5, %xmm5;"
                "addps %xmm6, %xmm6;"
                "addps %xmm7, %xmm7;"
                "addps %xmm8, %xmm8;"
                "addps %xmm9, %xmm9;"
                "addps %xmm10, %xmm10;"
                "addps %xmm11, %xmm11;"
                "addps %xmm12, %xmm12;"
                "addps %xmm13, %xmm13;"
                "addps %xmm14, %xmm14;"
                "addps %xmm15, %xmm15;");
    }
    __TOC__(LOOP)
    float flop = loop * 16 * 4;
    std::cout << "GFLOP/s: " << flop / __TIME_IN_MS__(LOOP) / 1e6 << std::endl;
    return 0;
}
