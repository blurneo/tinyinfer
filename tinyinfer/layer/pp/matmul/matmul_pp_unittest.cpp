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
    int M = 100, K = 320, N = 1280;
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C0(M * N);
    std::vector<float> C1(M * N);
    std::vector<float> C2(M * N);
    std::vector<float> C3(M * N);
    std::vector<float> C4(M * N);
    random_vector(A);
    random_vector(B);
    __TIC__(REF)
    for (int i = 0; i < 1; i++)
    {
        matmul_ref(M, K, N, A, B, C0);
    }
    __TOC__(REF)
    __TIC__(PP_BLOCK)
    for (int i = 0; i < 1; i++)
    {
        matmul_pp_block(M, K, N, A, B, C1);
    }
    __TOC__(PP_BLOCK)
    __TIC__(PP_BLOCK4)
    for (int i = 0; i < 1; i++)
    {
        matmul_pp_block4_unroll(M, K, N, A, B, C2);
    }
    __TOC__(PP_BLOCK4)
    __TIC__(PP_BLOCK4_PACKB)
    for (int i = 0; i < 1; i++)
    {
        matmul_pp_block4_packb_unroll(M, K, N, A, B, C3);
    }
    __TOC__(PP_BLOCK4_PACKB)
    __TIC__(PP_BLOCK4_PACKAB)
    for (int i = 0; i < 1; i++)
    {
        matmul_pp_block4_packab_unroll(M, K, N, A, B, C4);
    }
    __TOC__(PP_BLOCK4_PACKAB)
    CHECK_VEC_EQUAL_RET(C0, C1, -1, "C0 C1 not equal");
    CHECK_VEC_EQUAL_RET(C0, C2, -1, "C0 C2 not equal");
    CHECK_VEC_EQUAL_RET(C0, C3, -1, "C0 C3 not equal");
    CHECK_VEC_EQUAL_RET(C0, C4, -1, "C0 C4 not equal");

    return 0;
}
