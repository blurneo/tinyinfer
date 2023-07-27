#include <vector>
#include <stdio.h>
#include <iostream>
#include "gemm_pp.h"

namespace ti
{

    void block8_pack_b(int K, int N, const float *b, float *packed_b)
    {
        int loop_N = N / 8 * 8;
        int loop_K = K;
        float *packed0 = packed_b;
        float *packed1 = packed0 + 1;
        float *packed2 = packed1 + 1;
        float *packed3 = packed2 + 1;
        float *packed4 = packed3 + 1;
        float *packed5 = packed4 + 1;
        float *packed6 = packed5 + 1;
        float *packed7 = packed6 + 1;
        for (int n = 0; n < loop_N; n += 8)
        {
            const float *b0_p = b + n;
            const float *b1_p = b0_p + 1;
            const float *b2_p = b1_p + 1;
            const float *b3_p = b2_p + 1;
            const float *b4_p = b3_p + 1;
            const float *b5_p = b4_p + 1;
            const float *b6_p = b5_p + 1;
            const float *b7_p = b6_p + 1;
            for (int k = 0; k < loop_K; k++)
            {
                *packed0 = *b0_p;
                *packed1 = *b1_p;
                *packed2 = *b2_p;
                *packed3 = *b3_p;
                *packed4 = *b4_p;
                *packed5 = *b5_p;
                *packed6 = *b6_p;
                *packed7 = *b7_p;
                packed0 += 8;
                packed1 += 8;
                packed2 += 8;
                packed3 += 8;
                packed4 += 8;
                packed5 += 8;
                packed6 += 8;
                packed7 += 8;
                b0_p += N;
                b1_p += N;
                b2_p += N;
                b3_p += N;
                b4_p += N;
                b5_p += N;
                b6_p += N;
                b7_p += N;
            }
        }
    }

    void mul_add_4x8_packedab_simd(int m, int k, int n, const float *a, const float *b, float *c)
    {
        const float *a0 = a;
        const float *b0 = b;

        __m128 _c0 = _mm_load_ps(c);
        __m128 _c1 = _mm_load_ps(c + n);
        __m128 _c2 = _mm_load_ps(c + 2 * n);
        __m128 _c3 = _mm_load_ps(c + 3 * n);

        __m128 _c4 = _mm_load_ps(c + 4);
        __m128 _c5 = _mm_load_ps(c + n + 4);
        __m128 _c6 = _mm_load_ps(c + 2 * n + 4);
        __m128 _c7 = _mm_load_ps(c + 3 * n + 4);
        for (int i = 0; i < k; i++)
        {
            // __m128 a4 = _mm_load_ps(a0); // a0, a1, a2, a3

            __m128 b4_0 = _mm_load_ps(b0);     // b0, b1, b2, b3
            __m128 b4_1 = _mm_load_ps(b0 + 4); // b0, b1, b2, b3

            // __m128 alo = _mm_unpacklo_ps(a4, a4); // a0, a0, a1, a1
            // __m128 ahi = _mm_unpackhi_ps(a4, a4); // a2, a2, a3, a3

            // __m128 a0_4 = _mm_unpacklo_ps(alo, alo); // a0, a0, a0, a0
            // __m128 a1_4 = _mm_unpackhi_ps(alo, alo); // a1, a1, a1, a1
            // __m128 a2_4 = _mm_unpacklo_ps(ahi, ahi); // a2, a2, a2, a2
            // __m128 a3_4 = _mm_unpackhi_ps(ahi, ahi); // a3, a3, a3, a3

            __m128 a0_4 = _mm_load1_ps(a0+0);
            __m128 a1_4 = _mm_load1_ps(a0+1);
            __m128 a2_4 = _mm_load1_ps(a0+2);
            __m128 a3_4 = _mm_load1_ps(a0+3);
            _c0 = _c0 + a0_4 * b4_0;
            _c1 = _c1 + a1_4 * b4_0;
            _c2 = _c2 + a2_4 * b4_0;
            _c3 = _c3 + a3_4 * b4_0;
            _c4 = _c4 + a0_4 * b4_1;
            _c5 = _c5 + a1_4 * b4_1;
            _c6 = _c6 + a2_4 * b4_1;
            _c7 = _c7 + a3_4 * b4_1;

            a0 += 4;
            b0 += 8;
        }

        _mm_store_ps(c, _c0);
        _mm_store_ps(c + n, _c1);
        _mm_store_ps(c + 2 * n, _c2);
        _mm_store_ps(c + 3 * n, _c3);

        _mm_store_ps(c + 4, _c4);
        _mm_store_ps(c + n + 4, _c5);
        _mm_store_ps(c + 2 * n + 4, _c6);
        _mm_store_ps(c + 3 * n + 4, _c7);
    }

    // Matrix A: MxK
    // Matrix B: KxN
    // Matrix C: MxN
    // gemm using block and pack b methods
    void gemm_pp_block4x8_packab_unroll(int M, int K, int N, const float *A,
                                        const float *B, float *C)
    {
        const int OUT_COUNT = M * N;
        // C.resize(OUT_COUNT, 0);
        int BLOCK_SIZE_A = 4;
        int BLOCK_SIZE_B = 8;
        int BLOCK_NUM_A = M / BLOCK_SIZE_A;
        int BLOCK_NUM_B = N / BLOCK_SIZE_B;
        float *packeda = (float *)malloc(M * K * sizeof(float)); // packed into K*N/4 row 4 col
        float *packedb = (float *)malloc(K * N * sizeof(float)); // packed into K*N/4 row 4 col
        // pack b only once, only pack into block times shape, residuals no need to pack
        block4_pack_a(M, K, A, packeda);
        block8_pack_b(K, N, B, packedb);
        // divide all rows into multiple slices
        for (int a_block_idx = 0; a_block_idx < BLOCK_NUM_A; a_block_idx++)
        {
            int a_start_row = a_block_idx * BLOCK_SIZE_A;
            // iterate through each col
            int b_block_idx = 0;
            for (; b_block_idx < BLOCK_NUM_B; b_block_idx++)
            {
                int b_start_col = b_block_idx * BLOCK_SIZE_B;
                mul_add_4x8_packedab_simd(M, K, N, packeda + a_start_row * K, packedb + b_start_col * K, C + a_start_row * N + b_start_col);
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
