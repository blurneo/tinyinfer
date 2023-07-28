#include <vector>
#include <stdio.h>
#include <iostream>
#include "gemm_pp.h"

namespace ti
{

    void block8_pack_a(int M, int K, const float *a, float *packed_a)
    {
        int loop_M = M / 8 * 8;
        int loop_K = K;
        float *packed0 = packed_a;
        float *packed1 = packed0 + 1;
        float *packed2 = packed1 + 1;
        float *packed3 = packed2 + 1;
        float *packed4 = packed3 + 1;
        float *packed5 = packed4 + 1;
        float *packed6 = packed5 + 1;
        float *packed7 = packed6 + 1;
        for (int m = 0; m < loop_M; m += 8)
        {
            const float *a0_p = a + m * K;
            const float *a1_p = a0_p + K;
            const float *a2_p = a1_p + K;
            const float *a3_p = a2_p + K;
            const float *a4_p = a3_p + K;
            const float *a5_p = a4_p + K;
            const float *a6_p = a5_p + K;
            const float *a7_p = a6_p + K;
            for (int k = 0; k < loop_K; k++)
            {
                *packed0 = *a0_p;
                *packed1 = *a1_p;
                *packed2 = *a2_p;
                *packed3 = *a3_p;
                *packed4 = *a4_p;
                *packed5 = *a5_p;
                *packed6 = *a6_p;
                *packed7 = *a7_p;
                packed0 += 8;
                packed1 += 8;
                packed2 += 8;
                packed3 += 8;
                packed4 += 8;
                packed5 += 8;
                packed6 += 8;
                packed7 += 8;
                a0_p += 1;
                a1_p += 1;
                a2_p += 1;
                a3_p += 1;
                a4_p += 1;
                a5_p += 1;
                a6_p += 1;
                a7_p += 1;
            }
        }
    }

    void mul_add_8x8_packedab_avx(int m, int k, int n, const float *a, const float *b, float *c)
    {
        const float *a0 = a;
        const float *b0 = b;

        __m256 _c0 = _mm256_loadu_ps(c);
        __m256 _c1 = _mm256_loadu_ps(c + n);
        __m256 _c2 = _mm256_loadu_ps(c + 2 * n);
        __m256 _c3 = _mm256_loadu_ps(c + 3 * n);

        __m256 _c4 = _mm256_loadu_ps(c + 4 * n);
        __m256 _c5 = _mm256_loadu_ps(c + 5 * n);
        __m256 _c6 = _mm256_loadu_ps(c + 6 * n);
        __m256 _c7 = _mm256_loadu_ps(c + 7 * n);

        for (int i = 0; i < k; i++)
        {
            __m256 b8 = _mm256_loadu_ps(b0);

            __m256 a0_8 = _mm256_broadcast_ss(a0 + 0); // a0, a0, a0, a0
            __m256 a1_8 = _mm256_broadcast_ss(a0 + 1); // a1, a1, a1, a1
            __m256 a2_8 = _mm256_broadcast_ss(a0 + 2); // a2, a2, a2, a2
            __m256 a3_8 = _mm256_broadcast_ss(a0 + 3); // a3, a3, a3, a3

            __m256 a4_8 = _mm256_broadcast_ss(a0 + 4); // a4, a4, a4, a4
            __m256 a5_8 = _mm256_broadcast_ss(a0 + 5); // a5, a5, a5, a5
            __m256 a6_8 = _mm256_broadcast_ss(a0 + 6); // a6, a6, a6, a6
            __m256 a7_8 = _mm256_broadcast_ss(a0 + 7); // a7, a7, a7, a7

            _c0 = _c0 + a0_8 * b8;
            _c1 = _c1 + a1_8 * b8;
            _c2 = _c2 + a2_8 * b8;
            _c3 = _c3 + a3_8 * b8;
            _c4 = _c4 + a4_8 * b8;
            _c5 = _c5 + a5_8 * b8;
            _c6 = _c6 + a6_8 * b8;
            _c7 = _c7 + a7_8 * b8;

            a0 += 8;
            b0 += 8;
        }

        _mm256_storeu_ps(c, _c0);
        _mm256_storeu_ps(c + n, _c1);
        _mm256_storeu_ps(c + 2 * n, _c2);
        _mm256_storeu_ps(c + 3 * n, _c3);

        _mm256_storeu_ps(c + 4 * n, _c4);
        _mm256_storeu_ps(c + 5 * n, _c5);
        _mm256_storeu_ps(c + 6 * n, _c6);
        _mm256_storeu_ps(c + 7 * n, _c7);
    }

    // Matrix A: MxK
    // Matrix B: KxN
    // Matrix C: MxN
    // gemm using block and pack b methods
    void gemm_pp_block8x8_packab_unroll(int M, int K, int N, const std::vector<float> &A,
                                        const std::vector<float> &B, std::vector<float> &C)
    {
        const int OUT_COUNT = M * N;
        C.resize(OUT_COUNT, 0);
        int BLOCK_SIZE_A = 8;
        int BLOCK_SIZE_B = 8;
        int BLOCK_NUM_A = M / BLOCK_SIZE_A;
        int BLOCK_NUM_B = N / BLOCK_SIZE_B;
        float *packeda = (float *)malloc(M * K * sizeof(float)); // packed into K*N/4 row 4 col
        float *packedb = (float *)malloc(K * N * sizeof(float)); // packed into K*N/4 row 4 col
        // pack b only once, only pack into block times shape, residuals no need to pack
        block8_pack_a(M, K, A.data(), packeda);
        block8_pack_b(K, N, B.data(), packedb);
        // divide all rows into multiple slices
        for (int a_block_idx = 0; a_block_idx < BLOCK_NUM_A; a_block_idx++)
        {
            int a_start_row = a_block_idx * BLOCK_SIZE_A;
            // iterate through each col
            int b_block_idx = 0;
            for (; b_block_idx < BLOCK_NUM_B; b_block_idx++)
            {
                int b_start_col = b_block_idx * BLOCK_SIZE_B;
                mul_add_8x8_packedab_avx(M, K, N, packeda + a_start_row * K, packedb + b_start_col * K, C.data() + a_start_row * N + b_start_col);
            }
            // process the col residuals
            int b_col = b_block_idx * BLOCK_SIZE_B;
            for (; b_col < N; b_col++)
            {
                for (int k = 0; k < K; k++)
                {
                    // iterate through each block from left to right and top to bottom
                    for (int a_block_row = 0; a_block_row < BLOCK_SIZE_A; a_block_row++)
                    {
                        int a_row = a_start_row + a_block_row;
                        C[a_row * N + b_col] += A[a_row * K + k] * B[k * N + b_col];
                    }
                }
            }
        }
        // process the row residuals
        int a_start_row = BLOCK_NUM_A * BLOCK_SIZE_A;
        int b_start_col = 0;
        for (int a_row = a_start_row; a_row < M; a_row++)
        {
            for (int b_col = b_start_col; b_col < N; b_col++)
            {
                float sum = 0.f;
                for (int k = 0; k < K; k++)
                {
                    sum += A[a_row * K + k] * B[k * N + b_col];
                }
                C[a_row * N + b_col] = sum;
            }
        }
        free(packeda);
        free(packedb);
    }
}
