#include "tinyinfer/layer/pp/matmul/matmul_pp.h"
#include "tinyinfer/layer/pp/matmul/matmul_ref.h"
#include "tinyinfer/common/check_macro.h"

#include <vector>
#include <cstdlib>
#include <ctime>

void random_vector(std::vector<float> &vec)
{
    std::srand(std::time(0));
    for (auto &val : vec)
    {
        val = (float)(std::rand() / (float)RAND_MAX);
    }
}

using namespace ti;

int main()
{
    int M = 2000, K = 2000, N = 2000;
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C1(M * N);
    std::vector<float> C2(M * N);
    random_vector(A);
    random_vector(B);
    __TIC__(REF)
    for (int i = 0; i < 1; i++)
    {
        matmul_pp_block(M, K, N, A, B, C1);
    }
    __TOC__(REF)
    __TIC__(PP)
    for (int i = 0; i < 1; i++)
    {
        matmul_pp_block4_unroll(M, K, N, A, B, C2);
    }
    __TOC__(PP)
    CHECK_VEC_EQUAL_RET(C1, C2, -1, "C1 C2 not equal");

    return 0;
}
