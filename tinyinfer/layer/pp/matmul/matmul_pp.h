#pragma once
#include <vector>
#include <stdio.h>
#include <iostream>

namespace ti
{

    // Matrix A: MxK
    // Matrix B: KxN
    // Matrix C: MxN
    // matmul using block and pack methods
    void matmul_pp_block(int M, int K, int N, const std::vector<float> &A,
                         const std::vector<float> &B, std::vector<float> &C)
    {
        const int OUT_COUNT = M * N;
        C.resize(OUT_COUNT, 0);
        int BLOCK_SIZE = 8;
        int BLOCK_NUM_A = M / BLOCK_SIZE;
        int BLOCK_NUM_B = N / BLOCK_SIZE;
        // divide all rows into multiple slices
        for (int a_block_idx = 0; a_block_idx < BLOCK_NUM_A; a_block_idx++)
        {
            int a_start_row = a_block_idx * BLOCK_SIZE;
            // iterate through each col
            int b_block_idx = 0;
            for (; b_block_idx < BLOCK_NUM_B; b_block_idx++)
            {
                int b_start_col = b_block_idx * BLOCK_SIZE;
                for (int k = 0; k < K; k++)
                {
                    // iterate through each block from left to right and top to bottom
                    for (int a_block_row = 0; a_block_row < BLOCK_SIZE; a_block_row++)
                    {
                        for (int b_block_col = 0; b_block_col < BLOCK_SIZE; b_block_col++)
                        {
                            int a_row = a_start_row + a_block_row;
                            int b_col = b_start_col + b_block_col;
                            C[a_row * N + b_col] += A[a_row * K + k] * B[k * N + b_col];
                        }
                    }
                }
            }
            // process the col residuals
            int b_col = b_block_idx * BLOCK_SIZE;
            for (; b_col < N; b_col++)
            {
                for (int k = 0; k < K; k++)
                {
                    // iterate through each block from left to right and top to bottom
                    for (int a_block_row = 0; a_block_row < BLOCK_SIZE; a_block_row++)
                    {
                        int a_row = a_start_row + a_block_row;
                        C[a_row * N + b_col] += A[a_row * K + k] * B[k * N + b_col];
                    }
                }
            }
        }
        // process the row residuals
        int a_start_row = BLOCK_NUM_A * BLOCK_SIZE;
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
    }

    void mul_add_4x4(int m, int k, int n, const float *a, const float *b, float *c)
    {
        const float *a0 = a;
        const float *a1 = a + k;
        const float *a2 = a1 + k;
        const float *a3 = a2 + k;
        const float *b0 = b;
        const float *b1 = b + 1;
        const float *b2 = b1 + 1;
        const float *b3 = b2 + 1;
        register float c00 = 0.f, c01 = 0.f, c02 = 0.f, c03 = 0.f;
        register float c10 = 0.f, c11 = 0.f, c12 = 0.f, c13 = 0.f;
        register float c20 = 0.f, c21 = 0.f, c22 = 0.f, c23 = 0.f;
        register float c30 = 0.f, c31 = 0.f, c32 = 0.f, c33 = 0.f;
        register float a0i, a1i, a2i, a3i;
        register float b0i, b1i, b2i, b3i;
        for (int i = 0; i < k; i++)
        {
            a0i = *a0++;
            a1i = *a1++;
            a2i = *a2++;
            a3i = *a3++;

            b0i = *b0;
            b1i = *b1;
            b2i = *b2;
            b3i = *b3;

            c00 += a0i * b0i;
            c01 += a0i * b1i;
            c02 += a0i * b2i;
            c03 += a0i * b3i;
            c10 += a1i * b0i;
            c11 += a1i * b1i;
            c12 += a1i * b2i;
            c13 += a1i * b3i;
            c20 += a2i * b0i;
            c21 += a2i * b1i;
            c22 += a2i * b2i;
            c23 += a2i * b3i;
            c30 += a3i * b0i;
            c31 += a3i * b1i;
            c32 += a3i * b2i;
            c33 += a3i * b3i;

            b0 += n;
            b1 += n;
            b2 += n;
            b3 += n;
        }
        c[0] += c00;
        c[1] += c01;
        c[2] += c02;
        c[3] += c03;
        c[n] += c10;
        c[n + 1] += c11;
        c[n + 2] += c12;
        c[n + 3] += c13;
        c[2 * n + 0] += c20;
        c[2 * n + 1] += c21;
        c[2 * n + 2] += c22;
        c[2 * n + 3] += c23;
        c[3 * n + 0] += c30;
        c[3 * n + 1] += c31;
        c[3 * n + 2] += c32;
        c[3 * n + 3] += c33;
    }

    // Matrix A: MxK
    // Matrix B: KxN
    // Matrix C: MxN
    // matmul using block and pack methods
    void matmul_pp_block4_unroll(int M, int K, int N, const std::vector<float> &A,
                                 const std::vector<float> &B, std::vector<float> &C)
    {
        const int OUT_COUNT = M * N;
        C.resize(OUT_COUNT, 0);
        int BLOCK_SIZE = 4;
        int BLOCK_NUM_A = M / BLOCK_SIZE;
        int BLOCK_NUM_B = N / BLOCK_SIZE;
        // divide all rows into multiple slices
        for (int a_block_idx = 0; a_block_idx < BLOCK_NUM_A; a_block_idx++)
        {
            int a_start_row = a_block_idx * BLOCK_SIZE;
            // iterate through each col
            int b_block_idx = 0;
            for (; b_block_idx < BLOCK_NUM_B; b_block_idx++)
            {
                int b_start_col = b_block_idx * BLOCK_SIZE;
                mul_add_4x4(M, K, N, A.data() + a_start_row * K, B.data() + b_start_col, C.data() + a_start_row * N + b_start_col);
            }
            // process the col residuals
            int b_col = b_block_idx * BLOCK_SIZE;
            for (; b_col < N; b_col++)
            {
                for (int k = 0; k < K; k++)
                {
                    // iterate through each block from left to right and top to bottom
                    for (int a_block_row = 0; a_block_row < BLOCK_SIZE; a_block_row++)
                    {
                        int a_row = a_start_row + a_block_row;
                        C[a_row * N + b_col] += A[a_row * K + k] * B[k * N + b_col];
                    }
                }
            }
        }
        // process the row residuals
        int a_start_row = BLOCK_NUM_A * BLOCK_SIZE;
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
    }

    void block4_pack_b(int N, int K, const float *b, float *packed_b)
    {
        int loop_N = N / 4 * 4;
        int loop_K = K / 4 * 4;
        for (int n = 0; n < loop_N; n += 4)
        {
            const float *b0_p = b + n;
            const float *b1_p = b0_p + 1;
            const float *b2_p = b1_p + 1;
            const float *b3_p = b2_p + 1;
            float *packed0 = packed_b + n * K;
            float *packed1 = packed0 + K;
            float *packed2 = packed1 + K;
            float *packed3 = packed2 + K;
            for (int k = 0; k < loop_K; k++)
            {
                *packed0++ = *b0_p;
                *packed1++ = *b1_p;
                *packed2++ = *b2_p;
                *packed3++ = *b3_p;
                b0_p += N;
                b1_p += N;
                b2_p += N;
                b3_p += N;
            }
        }
    }

    void mul_add_4x4_packedb(int m, int k, int n, const float *a, const float *b, float *c)
    {
        const float *a0 = a;
        const float *a1 = a + k;
        const float *a2 = a1 + k;
        const float *a3 = a2 + k;
        const float *b0 = b;
        const float *b1 = b + k;
        const float *b2 = b1 + k;
        const float *b3 = b2 + k;
        register float c00 = 0.f, c01 = 0.f, c02 = 0.f, c03 = 0.f;
        register float c10 = 0.f, c11 = 0.f, c12 = 0.f, c13 = 0.f;
        register float c20 = 0.f, c21 = 0.f, c22 = 0.f, c23 = 0.f;
        register float c30 = 0.f, c31 = 0.f, c32 = 0.f, c33 = 0.f;
        register float a0i, a1i, a2i, a3i;
        register float b0i, b1i, b2i, b3i;
        for (int i = 0; i < k; i++)
        {
            a0i = *a0++;
            a1i = *a1++;
            a2i = *a2++;
            a3i = *a3++;

            b0i = *b0;
            b1i = *b1;
            b2i = *b2;
            b3i = *b3;

            c00 += a0i * b0i;
            c01 += a0i * b1i;
            c02 += a0i * b2i;
            c03 += a0i * b3i;
            c10 += a1i * b0i;
            c11 += a1i * b1i;
            c12 += a1i * b2i;
            c13 += a1i * b3i;
            c20 += a2i * b0i;
            c21 += a2i * b1i;
            c22 += a2i * b2i;
            c23 += a2i * b3i;
            c30 += a3i * b0i;
            c31 += a3i * b1i;
            c32 += a3i * b2i;
            c33 += a3i * b3i;

            b0++;
            b1++;
            b2++;
            b3++;
        }
        c[0] += c00;
        c[1] += c01;
        c[2] += c02;
        c[3] += c03;
        c[n] += c10;
        c[n + 1] += c11;
        c[n + 2] += c12;
        c[n + 3] += c13;
        c[2 * n + 0] += c20;
        c[2 * n + 1] += c21;
        c[2 * n + 2] += c22;
        c[2 * n + 3] += c23;
        c[3 * n + 0] += c30;
        c[3 * n + 1] += c31;
        c[3 * n + 2] += c32;
        c[3 * n + 3] += c33;
    }

    // Matrix A: MxK
    // Matrix B: KxN
    // Matrix C: MxN
    // matmul using block and pack b methods
    void matmul_pp_block4_packb_unroll(int M, int K, int N, const std::vector<float> &A,
                                       const std::vector<float> &B, std::vector<float> &C)
    {
        const int OUT_COUNT = M * N;
        C.resize(OUT_COUNT, 0);
        int BLOCK_SIZE = 4;
        int BLOCK_NUM_A = M / BLOCK_SIZE;
        int BLOCK_NUM_B = N / BLOCK_SIZE;
        float *packedb = (float *)malloc(N * K * sizeof(float)); // packed into N row K col
        // pack b only once, only pack into block times shape, residuals no need to pack
        block4_pack_b(N, K, B.data(), packedb);
        // divide all rows into multiple slices
        for (int a_block_idx = 0; a_block_idx < BLOCK_NUM_A; a_block_idx++)
        {
            int a_start_row = a_block_idx * BLOCK_SIZE;
            // iterate through each col
            int b_block_idx = 0;
            for (; b_block_idx < BLOCK_NUM_B; b_block_idx++)
            {
                int b_start_col = b_block_idx * BLOCK_SIZE;
                mul_add_4x4_packedb(M, K, N, A.data() + a_start_row * K, packedb + b_start_col * K, C.data() + a_start_row * N + b_start_col);
            }
            // process the col residuals
            int b_col = b_block_idx * BLOCK_SIZE;
            for (; b_col < N; b_col++)
            {
                for (int k = 0; k < K; k++)
                {
                    // iterate through each block from left to right and top to bottom
                    for (int a_block_row = 0; a_block_row < BLOCK_SIZE; a_block_row++)
                    {
                        int a_row = a_start_row + a_block_row;
                        C[a_row * N + b_col] += A[a_row * K + k] * B[k * N + b_col];
                    }
                }
            }
        }
        // process the row residuals
        int a_start_row = BLOCK_NUM_A * BLOCK_SIZE;
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
        free(packedb);
    }

}
