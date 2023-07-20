#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include "tinyinfer/common/check_macro.h"
#include <mmintrin.h>
#include <emmintrin.h>

void random_vector(std::vector<float> &vec)
{
    std::srand(std::time(0));
    for (auto &val : vec)
    {
        val = (float)(std::rand() / (float)RAND_MAX);
    }
}

typedef union
{
    __m128 v;
    float d[4];
} reg;

int main()
{
    std::vector<float> vec(4);

    random_vector(vec);
    __m128 v1 = _mm_set_ps(vec[0], vec[1], vec[2], vec[3]);
    random_vector(vec);
    __m128 v2 = _mm_set_ps(vec[0], vec[1], vec[2], vec[3]);
    random_vector(vec);
    __m128 v3 = _mm_set_ps(vec[0], vec[1], vec[2], vec[3]);
    random_vector(vec);
    __m128 v4 = _mm_set_ps(vec[0], vec[1], vec[2], vec[3]);
    random_vector(vec);
    __m128 v5 = _mm_set_ps(vec[0], vec[1], vec[2], vec[3]);
    random_vector(vec);
    __m128 v6 = _mm_set_ps(vec[0], vec[1], vec[2], vec[3]);
    random_vector(vec);
    __m128 v7 = _mm_set_ps(vec[0], vec[1], vec[2], vec[3]);
    random_vector(vec);
    __m128 v8 = _mm_set_ps(vec[0], vec[1], vec[2], vec[3]);
    random_vector(vec);
    __m128 v9 = _mm_set_ps(vec[0], vec[1], vec[2], vec[3]);
    random_vector(vec);
    __m128 v10 = _mm_set_ps(vec[0], vec[1], vec[2], vec[3]);
    random_vector(vec);
    __m128 v11 = _mm_set_ps(vec[0], vec[1], vec[2], vec[3]);
    random_vector(vec);
    __m128 v12 = _mm_set_ps(vec[0], vec[1], vec[2], vec[3]);
    random_vector(vec);
    __m128 v13 = _mm_set_ps(vec[0], vec[1], vec[2], vec[3]);
    random_vector(vec);
    __m128 v14 = _mm_set_ps(vec[0], vec[1], vec[2], vec[3]);
    random_vector(vec);
    __m128 v15 = _mm_set_ps(vec[0], vec[1], vec[2], vec[3]);
    random_vector(vec);
    __m128 v16 = _mm_set_ps(vec[0], vec[1], vec[2], vec[3]);

    random_vector(vec);
    reg _v0;
    _v0.v = _mm_set_ps(vec[0], vec[1], vec[2], vec[3]);
    random_vector(vec);
    reg _v1;
    _v1.v = _mm_set_ps(vec[0], vec[1], vec[2], vec[3]);
    random_vector(vec);
    reg _v2;
    _v2.v = _mm_set_ps(vec[0], vec[1], vec[2], vec[3]);
    random_vector(vec);
    reg _v3;
    _v3.v = _mm_set_ps(vec[0], vec[1], vec[2], vec[3]);
    random_vector(vec);
    reg _v4;
    _v4.v = _mm_set_ps(vec[0], vec[1], vec[2], vec[3]);
    random_vector(vec);
    reg _v5;
    _v5.v = _mm_set_ps(vec[0], vec[1], vec[2], vec[3]);
    random_vector(vec);
    reg _v6;
    _v6.v = _mm_set_ps(vec[0], vec[1], vec[2], vec[3]);
    reg _v7;
    _v7.v = _mm_set_ps(vec[0], vec[1], vec[2], vec[3]);
    reg _v8;
    _v8.v = _mm_set_ps(vec[0], vec[1], vec[2], vec[3]);

    uint64_t loop = 1024 * 1024 * 1024;
    __TIC__(LOOP)
    for (int i = 0; i < loop; i++)
    {
        _v0.v += v1 * v2;
        _v1.v += v3 * v4;
        _v2.v += v5 * v6;
        _v3.v += v7 * v8;
        _v4.v += v9 * v10;
        _v5.v += v11 * v12;
        _v6.v += v13 * v14;
        _v7.v += v15 * v16;
    }
    __TOC__(LOOP)
    float flop = loop * 8 * 8;
    printf("Print dummy: %f %f %f %f %f %f %f %f %f\n", _v0.d[0], _v1.d[0], _v2.d[0], _v3.d[0],
           _v4.d[0], _v5.d[0], _v6.d[0], _v7.d[0], _v8.d[0]);

    std::cout << "GFLOP/s: " << flop / __TIME_IN_MS__(LOOP) / 1e6 << std::endl;
    return 0;
}
