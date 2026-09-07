// task2b_simd_128.cpp
// Measurement-only 128-bit SIMD variant for Task 2B.

#include <immintrin.h>
#include "matmul.h"

static inline float horizontal_sum_128(__m128 value) {
    value = _mm_hadd_ps(value, value);
    value = _mm_hadd_ps(value, value);
    return _mm_cvtss_f32(value);
}

void matmul_simd_128(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    int i;
    for (i = 0; i + 1 < M; i += 2) {
        const float* a0 = A + static_cast<long>(i) * lda;
        const float* a1 = A + static_cast<long>(i + 1) * lda;

        int j;
        for (j = 0; j + 3 < N; j += 4) {
            const float* b0 = B + static_cast<long>(j) * ldb;
            const float* b1 = B + static_cast<long>(j + 1) * ldb;
            const float* b2 = B + static_cast<long>(j + 2) * ldb;
            const float* b3 = B + static_cast<long>(j + 3) * ldb;

            __m128 acc00 = _mm_setzero_ps(), acc01 = _mm_setzero_ps();
            __m128 acc02 = _mm_setzero_ps(), acc03 = _mm_setzero_ps();
            __m128 acc10 = _mm_setzero_ps(), acc11 = _mm_setzero_ps();
            __m128 acc12 = _mm_setzero_ps(), acc13 = _mm_setzero_ps();

            int p;
            for (p = 0; p + 3 < K; p += 4) {
                const __m128 av0 = _mm_loadu_ps(a0 + p);
                const __m128 av1 = _mm_loadu_ps(a1 + p);
                const __m128 bv0 = _mm_loadu_ps(b0 + p);
                const __m128 bv1 = _mm_loadu_ps(b1 + p);
                const __m128 bv2 = _mm_loadu_ps(b2 + p);
                const __m128 bv3 = _mm_loadu_ps(b3 + p);

                acc00 = _mm_add_ps(acc00, _mm_mul_ps(av0, bv0));
                acc01 = _mm_add_ps(acc01, _mm_mul_ps(av0, bv1));
                acc02 = _mm_add_ps(acc02, _mm_mul_ps(av0, bv2));
                acc03 = _mm_add_ps(acc03, _mm_mul_ps(av0, bv3));

                acc10 = _mm_add_ps(acc10, _mm_mul_ps(av1, bv0));
                acc11 = _mm_add_ps(acc11, _mm_mul_ps(av1, bv1));
                acc12 = _mm_add_ps(acc12, _mm_mul_ps(av1, bv2));
                acc13 = _mm_add_ps(acc13, _mm_mul_ps(av1, bv3));
            }

            float s00 = horizontal_sum_128(acc00), s01 = horizontal_sum_128(acc01);
            float s02 = horizontal_sum_128(acc02), s03 = horizontal_sum_128(acc03);
            float s10 = horizontal_sum_128(acc10), s11 = horizontal_sum_128(acc11);
            float s12 = horizontal_sum_128(acc12), s13 = horizontal_sum_128(acc13);

            for (; p < K; ++p) {
                const float av0 = a0[p], av1 = a1[p];
                s00 += av0 * b0[p]; s01 += av0 * b1[p];
                s02 += av0 * b2[p]; s03 += av0 * b3[p];
                s10 += av1 * b0[p]; s11 += av1 * b1[p];
                s12 += av1 * b2[p]; s13 += av1 * b3[p];
            }

            C[static_cast<long>(i) * ldc + j] = s00;
            C[static_cast<long>(i) * ldc + j + 1] = s01;
            C[static_cast<long>(i) * ldc + j + 2] = s02;
            C[static_cast<long>(i) * ldc + j + 3] = s03;
            C[static_cast<long>(i + 1) * ldc + j] = s10;
            C[static_cast<long>(i + 1) * ldc + j + 1] = s11;
            C[static_cast<long>(i + 1) * ldc + j + 2] = s12;
            C[static_cast<long>(i + 1) * ldc + j + 3] = s13;
        }

        for (; j < N; ++j) {
            const float* b = B + static_cast<long>(j) * ldb;
            __m128 acc0 = _mm_setzero_ps(), acc1 = _mm_setzero_ps();
            int p;
            for (p = 0; p + 3 < K; p += 4) {
                const __m128 av0 = _mm_loadu_ps(a0 + p);
                const __m128 av1 = _mm_loadu_ps(a1 + p);
                const __m128 bv = _mm_loadu_ps(b + p);
                acc0 = _mm_add_ps(acc0, _mm_mul_ps(av0, bv));
                acc1 = _mm_add_ps(acc1, _mm_mul_ps(av1, bv));
            }
            float s0 = horizontal_sum_128(acc0), s1 = horizontal_sum_128(acc1);
            for (; p < K; ++p) { s0 += a0[p] * b[p]; s1 += a1[p] * b[p]; }
            C[static_cast<long>(i) * ldc + j] = s0;
            C[static_cast<long>(i + 1) * ldc + j] = s1;
        }
    }

    for (; i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;
        for (int j = 0; j < N; ++j) {
            const float* b = B + static_cast<long>(j) * ldb;
            __m128 acc = _mm_setzero_ps();
            int p;
            for (p = 0; p + 3 < K; p += 4) {
                acc = _mm_add_ps(acc, _mm_mul_ps(_mm_loadu_ps(a + p), _mm_loadu_ps(b + p)));
            }
            float s = horizontal_sum_128(acc);
            for (; p < K; ++p) s += a[p] * b[p];
            C[static_cast<long>(i) * ldc + j] = s;
        }
    }
}
