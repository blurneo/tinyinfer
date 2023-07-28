#include "tinyinfer/layer/pp/gemm/gemm_pp.h"
#include "tinyinfer/layer/pp/gemm/gemm_ref.h"
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
    unsigned long M = 100, K = 320, N = 1280;
    int loop = 1;
    float gflop = loop * 2 * M * K * N / 1024.0 / 1024.0 / 1024.0;
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C0(M * N);
    std::vector<float> C1(M * N);
    std::vector<float> C2(M * N);
    std::vector<float> C3(M * N);
    std::vector<float> C4(M * N);
    std::vector<float> C5(M * N);
    std::vector<float> C6(M * N);
    std::vector<float> C7(M * N);
    std::vector<float> C8(M * N);
    random_vector(A);
    random_vector(B);
    __TIC__(REF)
    for (int i = 0; i < loop; i++)
    {
        gemm_ref(M, K, N, A.data(), B.data(), C0.data());
    }
    __TOC__(REF)
    std::cout << "GFlops: " << gflop / elapsed_ms_REF * 1000 << "\n";
    __TIC__(PP_BLOCK)
    for (int i = 0; i < loop; i++)
    {
        gemm_pp_block(M, K, N, A, B, C1);
    }
    __TOC__(PP_BLOCK)
    std::cout << "GFlops: " << gflop / elapsed_ms_PP_BLOCK * 1000 << "\n";
    __TIC__(PP_BLOCK4)
    for (int i = 0; i < loop; i++)
    {
        gemm_pp_block4_unroll(M, K, N, A, B, C2);
    }
    __TOC__(PP_BLOCK4)
    std::cout << "GFlops: " << gflop / elapsed_ms_PP_BLOCK4 * 1000 << "\n";
    __TIC__(PP_BLOCK4_PACKB)
    for (int i = 0; i < loop; i++)
    {
        gemm_pp_block4_packb_unroll(M, K, N, A, B, C3);
    }
    __TOC__(PP_BLOCK4_PACKB)
    std::cout << "GFlops: " << gflop / elapsed_ms_PP_BLOCK4_PACKB * 1000 << "\n";
    __TIC__(PP_BLOCK4_PACKAB)
    for (int i = 0; i < loop; i++)
    {
        gemm_pp_block4_packab_unroll(M, K, N, A, B, C4);
    }
    __TOC__(PP_BLOCK4_PACKAB)
    std::cout << "GFlops: " << gflop / elapsed_ms_PP_BLOCK4_PACKAB * 1000 << "\n";
    __TIC__(PP_BLOCK4x8_PACKAB)
    for (int i = 0; i < loop; i++)
    {
        gemm_pp_block4x8_packab_unroll(M, K, N, A.data(), B.data(), C5.data());
    }
    __TOC__(PP_BLOCK4x8_PACKAB)
    std::cout << "GFlops: " << gflop / elapsed_ms_PP_BLOCK4x8_PACKAB * 1000 << "\n";
    __TIC__(PP_BLOCK8x8_PACKAB)
    for (int i = 0; i < loop; i++)
    {
        gemm_pp_block8x8_packab_unroll(M, K, N, A, B, C6);
    }
    __TOC__(PP_BLOCK8x8_PACKAB)
    std::cout << "GFlops: " << gflop / elapsed_ms_PP_BLOCK8x8_PACKAB * 1000 << "\n";
    __TIC__(PP_BLOCK4x16_PACKAB)
    for (int i = 0; i < loop; i++)
    {
        gemm_pp_block4x16_packab_unroll(M, K, N, A.data(), B.data(), C7.data());
    }
    __TOC__(PP_BLOCK4x16_PACKAB)
    std::cout << "GFlops: " << gflop / elapsed_ms_PP_BLOCK4x16_PACKAB * 1000 << "\n";
    __TIC__(PP_BLOCK4x16_NR_PACKAB)
    for (int i = 0; i < loop; i++)
    {
        gemm_pp_block4x16_nr_packab_unroll(M, K, N, A.data(), B.data(), C8.data());
    }
    __TOC__(PP_BLOCK4x16_NR_PACKAB)
    std::cout << "GFlops: " << gflop / elapsed_ms_PP_BLOCK4x16_NR_PACKAB * 1000 << "\n";
    CHECK_VEC_EQUAL_RET(C0, C1, -1, "C0 C1 not equal");
    CHECK_VEC_EQUAL_RET(C0, C2, -1, "C0 C2 not equal");
    CHECK_VEC_EQUAL_RET(C0, C3, -1, "C0 C3 not equal");
    CHECK_VEC_EQUAL_RET(C0, C4, -1, "C0 C4 not equal");
    CHECK_VEC_EQUAL_RET(C0, C5, -1, "C0 C5 not equal");
    CHECK_VEC_EQUAL_RET(C0, C6, -1, "C0 C6 not equal");
    CHECK_VEC_EQUAL_RET(C0, C7, -1, "C0 C7 not equal");

    return 0;
}
