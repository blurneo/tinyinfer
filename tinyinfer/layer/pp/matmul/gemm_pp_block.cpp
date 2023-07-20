#include <vector>
#include <stdio.h>
#include <iostream>
#include "gemm_pp.h"

namespace ti
{

    void gemm_pp_block(int M, int K, int N, const std::vector<float> &A,
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
}
