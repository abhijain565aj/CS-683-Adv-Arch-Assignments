// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>

#include "matmul.h"

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    int BM = 32;
    int BN = 32;
    int prefetch_distance = 64;
    int cache_line_floats = 16;

    for (int ii = 0; ii < M; ii += BM) {
        int i_end = (ii + BM < M) ? ii + BM : M;
        for (int jj = 0; jj < N; jj += BN) {
            int j_end = (jj + BN < N) ? jj + BN : N;
            for (int i = ii; i < i_end; ++i) {
                const float* a = A + static_cast<long>(i) * lda;
                for (int j = jj; j < j_end; ++j) {
                    const float* b = B + static_cast<long>(j) * ldb;
                    float acc = 0.0f;
                    for (int p = 0; p < K; ++p) {
                        if (p % cache_line_floats == 0 && p + prefetch_distance < K) {
                            _mm_prefetch(reinterpret_cast<const char*>(a + p + prefetch_distance),_MM_HINT_T0);
                            _mm_prefetch(reinterpret_cast<const char*>(b + p + prefetch_distance),_MM_HINT_T0);
                        }
                        acc += a[p] * b[p];
                    }
                    C[static_cast<long>(i) * ldc + j] = acc;
                }
            }
        }
    }
}
