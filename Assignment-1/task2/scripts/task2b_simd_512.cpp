// task2b_simd_512.cpp
// Measurement-only 512-bit SIMD variant for Task 2B.
// Run only on CPUs that support AVX-512F.

#include <immintrin.h>
#include "matmul.h"

static inline float horizontal_sum_512(__m512 value) {
    return _mm512_reduce_add_ps(value);
}

void matmul_simd_512(const float* A, const float* B, float* C,
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

            __m512 acc00 = _mm512_setzero_ps(), acc01 = _mm512_setzero_ps();
            __m512 acc02 = _mm512_setzero_ps(), acc03 = _mm512_setzero_ps();
            __m512 acc10 = _mm512_setzero_ps(), acc11 = _mm512_setzero_ps();
            __m512 acc12 = _mm512_setzero_ps(), acc13 = _mm512_setzero_ps();

            int p;
            for (p = 0; p + 15 < K; p += 16) {
                const __m512 av0 = _mm512_loadu_ps(a0 + p);
                const __m512 av1 = _mm512_loadu_ps(a1 + p);
                const __m512 bv0 = _mm512_loadu_ps(b0 + p);
                const __m512 bv1 = _mm512_loadu_ps(b1 + p);
                const __m512 bv2 = _mm512_loadu_ps(b2 + p);
                const __m512 bv3 = _mm512_loadu_ps(b3 + p);

                acc00 = _mm512_fmadd_ps(av0, bv0, acc00);
                acc01 = _mm512_fmadd_ps(av0, bv1, acc01);
                acc02 = _mm512_fmadd_ps(av0, bv2, acc02);
                acc03 = _mm512_fmadd_ps(av0, bv3, acc03);
                acc10 = _mm512_fmadd_ps(av1, bv0, acc10);
                acc11 = _mm512_fmadd_ps(av1, bv1, acc11);
                acc12 = _mm512_fmadd_ps(av1, bv2, acc12);
                acc13 = _mm512_fmadd_ps(av1, bv3, acc13);
            }

            float s00 = horizontal_sum_512(acc00), s01 = horizontal_sum_512(acc01);
            float s02 = horizontal_sum_512(acc02), s03 = horizontal_sum_512(acc03);
            float s10 = horizontal_sum_512(acc10), s11 = horizontal_sum_512(acc11);
            float s12 = horizontal_sum_512(acc12), s13 = horizontal_sum_512(acc13);

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
            __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
            int p;
            for (p = 0; p + 15 < K; p += 16) {
                const __m512 av0 = _mm512_loadu_ps(a0 + p);
                const __m512 av1 = _mm512_loadu_ps(a1 + p);
                const __m512 bv = _mm512_loadu_ps(b + p);
                acc0 = _mm512_fmadd_ps(av0, bv, acc0);
                acc1 = _mm512_fmadd_ps(av1, bv, acc1);
            }
            float s0 = horizontal_sum_512(acc0), s1 = horizontal_sum_512(acc1);
            for (; p < K; ++p) { s0 += a0[p] * b[p]; s1 += a1[p] * b[p]; }
            C[static_cast<long>(i) * ldc + j] = s0;
            C[static_cast<long>(i + 1) * ldc + j] = s1;
        }
    }

    for (; i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;
        for (int j = 0; j < N; ++j) {
            const float* b = B + static_cast<long>(j) * ldb;
            __m512 acc = _mm512_setzero_ps();
            int p;
            for (p = 0; p + 15 < K; p += 16) {
                acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + p), _mm512_loadu_ps(b + p), acc);
            }
            float s = horizontal_sum_512(acc);
            for (; p < K; ++p) s += a[p] * b[p];
            C[static_cast<long>(i) * ldc + j] = s;
        }
    }
}
