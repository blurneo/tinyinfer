#include <vector>
#include <stdio.h>
#include <iostream>
#include "gemm_pp.h"
#include "tinyinfer/common/check_macro.h"

namespace ti
{
    void mul_add_4x16_packedab_avx(int m, int k, int n, int nr, const float *a, const float *b, float *c)
    {
        const float *a0 = a;
        const float *b0 = b;
        const float *b1 = b0 + 8;

        __m256 _c0 = _mm256_loadu_ps(c);
        __m256 _c1 = _mm256_loadu_ps(c + n);
        __m256 _c2 = _mm256_loadu_ps(c + 2 * n);
        __m256 _c3 = _mm256_loadu_ps(c + 3 * n);

        __m256 _c4 = _mm256_loadu_ps(c + 8);
        __m256 _c5 = _mm256_loadu_ps(c + n + 8);
        __m256 _c6 = _mm256_loadu_ps(c + 2 * n + 8);
        __m256 _c7 = _mm256_loadu_ps(c + 3 * n + 8);
#pragma unroll
        for (int i = 0; i < k; i++)
        {
            __m256 b0_8 = _mm256_loadu_ps(b0);
            __m256 b1_8 = _mm256_loadu_ps(b1);

            __m256 a0_8 = _mm256_broadcast_ss(a0 + 0); // a0, a0, a0, a0
            __m256 a1_8 = _mm256_broadcast_ss(a0 + 1); // a1, a1, a1, a1
            __m256 a2_8 = _mm256_broadcast_ss(a0 + 2); // a2, a2, a2, a2
            __m256 a3_8 = _mm256_broadcast_ss(a0 + 3); // a3, a3, a3, a3

            _c0 = _c0 + a0_8 * b0_8;
            _c1 = _c1 + a1_8 * b0_8;
            _c2 = _c2 + a2_8 * b0_8;
            _c3 = _c3 + a3_8 * b0_8;
            _c4 = _c4 + a0_8 * b1_8;
            _c5 = _c5 + a1_8 * b1_8;
            _c6 = _c6 + a2_8 * b1_8;
            _c7 = _c7 + a3_8 * b1_8;

            a0 += 4;
            b0 += nr;
            b1 += nr;
        }

        _mm256_storeu_ps(c, _c0);
        _mm256_storeu_ps(c + n, _c1);
        _mm256_storeu_ps(c + 2 * n, _c2);
        _mm256_storeu_ps(c + 3 * n, _c3);

        _mm256_storeu_ps(c + 8, _c4);
        _mm256_storeu_ps(c + n + 8, _c5);
        _mm256_storeu_ps(c + 2 * n + 8, _c6);
        _mm256_storeu_ps(c + 3 * n + 8, _c7);
    }

    // Matrix A: MxK
    // Matrix B: KxN
    // Matrix C: MxN
    // gemm using block and pack b methods
    void gemm_pp_block4x16_nr_packab_unroll(int M, int K, int N, const float *A,
                                         const float *B, float *C)
    {
        const int OUT_COUNT = M * N;
        // C.resize(OUT_COUNT, 0);
        int BLOCK_SIZE_A = 4;
        int BLOCK_SIZE_B = 16;
        int BLOCK_SIZE_Nr = 32;
        int BLOCK_NUM_A = M / BLOCK_SIZE_A;
        int BLOCK_NUM_B = BLOCK_SIZE_Nr / BLOCK_SIZE_B;
        int BLOCK_NUM_NR = N / BLOCK_SIZE_Nr;
        float *packeda = (float *)malloc(M * K * sizeof(float)); // packed into K*N/4 row 4 col
        float *packedb = (float *)malloc(K * N * sizeof(float)); // packed into K*N/4 row 4 col
        // pack b only once, only pack into block times shape, residuals no need to pack
        block4_pack_a(M, K, A, packeda);
        block_nr_pack_b(K, N, BLOCK_SIZE_Nr, B, packedb);
        // divide all rows into multiple slices
        for (int a_block_idx = 0; a_block_idx < BLOCK_NUM_A; a_block_idx++)
        {
            int a_start_row = a_block_idx * BLOCK_SIZE_A;
            int nr_block_idx = 0;
            for (; nr_block_idx < BLOCK_NUM_NR; nr_block_idx++)
            {
                // iterate through each col
                int b_start_col = nr_block_idx * BLOCK_SIZE_Nr;
                for (int b_block_idx = 0; b_block_idx < BLOCK_NUM_B; b_block_idx++)
                {
                    int small_block_col = b_block_idx * BLOCK_SIZE_B;
                    mul_add_4x16_packedab_avx(M, K, N, BLOCK_SIZE_Nr, packeda + a_start_row * K,
                                              packedb + b_start_col * K + small_block_col,
                                              C + a_start_row * N + b_start_col + small_block_col);
                }
            }
            // process the col residuals
            int b_col = nr_block_idx * BLOCK_SIZE_B;
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
