#pragma once
#include <vector>
#include <stdio.h>
#include <iostream>
#ifdef __x86_64__
#include <immintrin.h>
#endif

namespace ti
{

    void block4_pack_a(int M, int K, const float *a, float *packed_a);
    void block4_pack_b(int K, int N, const float *b, float *packed_b);
    void block8_pack_a(int M, int K, const float *b, float *packed_a);
    void block8_pack_b(int K, int N, const float *b, float *packed_b);

    void mul_add_4x4(int m, int k, int n, const float *a, const float *b, float *c);
    void mul_add_4x4_packedb(int m, int k, int n, const float *a, const float *b, float *c);
    void mul_add_4x4_packedab(int m, int k, int n, const float *a, const float *b, float *c);
#ifdef __x86_64__
    void mul_add_4x4_packedab_simd(int m, int k, int n, const float *a, const float *b, float *c);
#endif
    void mul_add_4x8_packedab_simd(int m, int k, int n, const float *a, const float *b, float *c);
    void mul_add_8x8_packedab_simd(int m, int k, int n, const float *a, const float *b, float *c);

    // Matrix A: MxK
    // Matrix B: KxN
    // Matrix C: MxN
    // gemm using block and pack methods
    void gemm_pp_block(int M, int K, int N, const std::vector<float> &A,
                       const std::vector<float> &B, std::vector<float> &C);
    // Matrix A: MxK
    // Matrix B: KxN
    // Matrix C: MxN
    // gemm using block and pack methods
    void gemm_pp_block4_unroll(int M, int K, int N, const std::vector<float> &A,
                               const std::vector<float> &B, std::vector<float> &C);
    // Matrix A: MxK
    // Matrix B: KxN
    // Matrix C: MxN
    // gemm using block and pack b methods
    void gemm_pp_block4_packb_unroll(int M, int K, int N, const std::vector<float> &A,
                                     const std::vector<float> &B, std::vector<float> &C);
    // Matrix A: MxK
    // Matrix B: KxN
    // Matrix C: MxN
    // gemm using block and pack b methods
    void gemm_pp_block4_packab_unroll(int M, int K, int N, const std::vector<float> &A,
                                      const std::vector<float> &B, std::vector<float> &C);

    // Matrix A: MxK
    // Matrix B: KxN
    // Matrix C: MxN
    // gemm using block and pack b methods
    void gemm_pp_block4x8_packab_unroll(int M, int K, int N, const std::vector<float> &A,
                                        const std::vector<float> &B, std::vector<float> &C);
    // Matrix A: MxK
    // Matrix B: KxN
    // Matrix C: MxN
    // gemm using block and pack b methods
    void gemm_pp_block8x8_packab_unroll(int M, int K, int N, const std::vector<float> &A,
                                        const std::vector<float> &B, std::vector<float> &C);

}
