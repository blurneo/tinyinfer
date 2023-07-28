#include "gemm_pp.h"

namespace ti
{

#ifdef __x86_64__
    void mul_add_8x4_packedab_simd(int m, int k, int n, const float *a, const float *b, float *c)
    {
        const float *a0 = a;
        const float *a1 = a + 4;
        const float *b0 = b;

        __m128 _c0 = _mm_load_ps(c);
        __m128 _c1 = _mm_load_ps(c + n);
        __m128 _c2 = _mm_load_ps(c + 2 * n);
        __m128 _c3 = _mm_load_ps(c + 3 * n);

        __m128 _c4 = _mm_load_ps(c + 4 * n);
        __m128 _c5 = _mm_load_ps(c + 5 * n);
        __m128 _c6 = _mm_load_ps(c + 6 * n);
        __m128 _c7 = _mm_load_ps(c + 7 * n);
        for (int i = 0; i < k; i++)
        {
            __m128 a04 = _mm_load_ps(a0); // a0, a1, a2, a3
            __m128 a14 = _mm_load_ps(a1); // a0, a1, a2, a3

            __m128 b4 = _mm_load_ps(b0); // b0, b1, b2, b3

            __m128 alo = _mm_unpacklo_ps(a04, a04); // a0, a0, a1, a1
            __m128 ahi = _mm_unpackhi_ps(a04, a04); // a2, a2, a3, a3

            __m128 a0_4 = _mm_unpacklo_ps(alo, alo); // a0, a0, a0, a0
            __m128 a1_4 = _mm_unpackhi_ps(alo, alo); // a1, a1, a1, a1
            __m128 a2_4 = _mm_unpacklo_ps(ahi, ahi); // a2, a2, a2, a2
            __m128 a3_4 = _mm_unpackhi_ps(ahi, ahi); // a3, a3, a3, a3

            _c0 = _c0 + a0_4 * b4;
            _c1 = _c1 + a1_4 * b4;
            _c2 = _c2 + a2_4 * b4;
            _c3 = _c3 + a3_4 * b4;

            alo = _mm_unpacklo_ps(a14, a14); // a0, a0, a1, a1
            ahi = _mm_unpackhi_ps(a14, a14); // a2, a2, a3, a3

            a0_4 = _mm_unpacklo_ps(alo, alo); // a0, a0, a0, a0
            a1_4 = _mm_unpackhi_ps(alo, alo); // a1, a1, a1, a1
            a2_4 = _mm_unpacklo_ps(ahi, ahi); // a2, a2, a2, a2
            a3_4 = _mm_unpackhi_ps(ahi, ahi); // a3, a3, a3, a3

            _c4 = _c4 + a0_4 * b4;
            _c5 = _c5 + a1_4 * b4;
            _c6 = _c6 + a2_4 * b4;
            _c7 = _c7 + a3_4 * b4;

            a0 += 8;
            b0 += 4;
        }
        _mm_store_ps(c, _c0);
        _mm_store_ps(c + n, _c1);
        _mm_store_ps(c + 2 * n, _c2);
        _mm_store_ps(c + 3 * n, _c3);
        _mm_store_ps(c + 4 * n, _c0);
        _mm_store_ps(c + 5 * n, _c1);
        _mm_store_ps(c + 6 * n, _c2);
        _mm_store_ps(c + 7 * n, _c3);
    }
#endif

    // Matrix A: MxK
    // Matrix B: KxN
    // Matrix C: MxN
    // gemm using block and pack b methods
    void gemm_pp_block8x4_packab_unroll(int M, int K, int N, const float *A,
                                        const float *B, float *C)
    {
        const int OUT_COUNT = M * N;
        // C.resize(OUT_COUNT, 0);
        int BLOCK_SIZE_A = 8;
        int BLOCK_SIZE_B = 4;
        int BLOCK_NUM_A = M / BLOCK_SIZE_A;
        int BLOCK_NUM_B = N / BLOCK_SIZE_B;
        float *packeda = (float *)malloc(M * K * sizeof(float)); // packed into K*N/4 row 4 col
        float *packedb = (float *)malloc(K * N * sizeof(float)); // packed into K*N/4 row 4 col
        // pack b only once, only pack into block times shape, residuals no need to pack
        block8_pack_a(M, K, A, packeda);
        block4_pack_b(K, N, B, packedb);
        // divide all rows into multiple slices
        for (int a_block_idx = 0; a_block_idx < BLOCK_NUM_A; a_block_idx++)
        {
            int a_start_row = a_block_idx * BLOCK_SIZE_A;
            // iterate through each col
            int b_block_idx = 0;
            for (; b_block_idx < BLOCK_NUM_B; b_block_idx++)
            {
                int b_start_col = b_block_idx * BLOCK_SIZE_B;
#ifdef __x86_64__
                mul_add_8x4_packedab_simd(M, K, N, packeda + a_start_row * K, packedb + b_start_col * K, C + a_start_row * N + b_start_col);
#else
                mul_add_8x4_packedab(M, K, N, packeda + a_start_row * K, packedb + b_start_col * K, C.data() + a_start_row * N + b_start_col);
#endif
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
