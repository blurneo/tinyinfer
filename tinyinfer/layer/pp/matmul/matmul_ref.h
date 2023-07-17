#pragma once

#include <vector>

namespace ti {

void matmul_ref(int M, int K, int N, const std::vector<float> &A,
                    const std::vector<float> &B, std::vector<float> &C) {
    for (int h1 = 0; h1 < M; h1++) {
      for (int w2 = 0; w2 < N; w2++) {
        float sum = 0.f;
        for (int w1 = 0, h2 = 0; w1 < K; w1++, h2++) {
          sum += A[h1 * K + w1] * B[h2 * N + w2];
        }
        C[h1 * N + w2] = sum;
      }
    }
}

}