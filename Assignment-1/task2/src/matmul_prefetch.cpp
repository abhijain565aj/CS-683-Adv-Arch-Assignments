// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>

#include "matmul.h"

static inline float horizontal_sum(__m256 value) {
    __m128 low = _mm256_castps256_ps128(value);
    __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(low, high);

    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);

    return _mm_cvtss_f32(sum);
}

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    int prefetch_distance = 64;
    int cache_line_floats = 16;

    for (int i = 0; i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;
        for (int j = 0; j < N; ++j) {
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
