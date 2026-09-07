// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
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


void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your register-tiled AVX2 implementation.int i;
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

            __m256 acc00 = _mm256_setzero_ps();
            __m256 acc01 = _mm256_setzero_ps();
            __m256 acc02 = _mm256_setzero_ps();
            __m256 acc03 = _mm256_setzero_ps();
            __m256 acc10 = _mm256_setzero_ps();
            __m256 acc11 = _mm256_setzero_ps();
            __m256 acc12 = _mm256_setzero_ps();
            __m256 acc13 = _mm256_setzero_ps();

            int p;
            for (p = 0; p + 7 < K; p += 8) {
                __m256 a0_vec = _mm256_loadu_ps(a0 + p);
                __m256 a1_vec = _mm256_loadu_ps(a1 + p);

                __m256 b0_vec = _mm256_loadu_ps(b0 + p);
                acc00 = _mm256_fmadd_ps(a0_vec, b0_vec, acc00);
                acc10 = _mm256_fmadd_ps(a1_vec, b0_vec, acc10);

                __m256 b1_vec = _mm256_loadu_ps(b1 + p);
                acc01 = _mm256_fmadd_ps(a0_vec, b1_vec, acc01);
                acc11 = _mm256_fmadd_ps(a1_vec, b1_vec, acc11);

                __m256 b2_vec = _mm256_loadu_ps(b2 + p);
                acc02 = _mm256_fmadd_ps(a0_vec, b2_vec, acc02);
                acc12 = _mm256_fmadd_ps(a1_vec, b2_vec, acc12);

                __m256 b3_vec = _mm256_loadu_ps(b3 + p);
                acc03 = _mm256_fmadd_ps(a0_vec, b3_vec, acc03);
                acc13 = _mm256_fmadd_ps(a1_vec, b3_vec, acc13);
            }

            float sum00 = horizontal_sum(acc00);
            float sum01 = horizontal_sum(acc01);
            float sum02 = horizontal_sum(acc02);
            float sum03 = horizontal_sum(acc03);
            float sum10 = horizontal_sum(acc10);
            float sum11 = horizontal_sum(acc11);
            float sum12 = horizontal_sum(acc12);
            float sum13 = horizontal_sum(acc13);

            for (; p < K; ++p) {
                float a0_value = a0[p];
                float a1_value = a1[p];

                sum00 += a0_value * b0[p];
                sum01 += a0_value * b1[p];
                sum02 += a0_value * b2[p];
                sum03 += a0_value * b3[p];
                sum10 += a1_value * b0[p];
                sum11 += a1_value * b1[p];
                sum12 += a1_value * b2[p];
                sum13 += a1_value * b3[p];
            }

            C[static_cast<long>(i) * ldc + j] = sum00;
            C[static_cast<long>(i) * ldc + j + 1] = sum01;
            C[static_cast<long>(i) * ldc + j + 2] = sum02;
            C[static_cast<long>(i) * ldc + j + 3] = sum03;
            C[static_cast<long>(i + 1) * ldc + j] = sum10;
            C[static_cast<long>(i + 1) * ldc + j + 1] = sum11;
            C[static_cast<long>(i + 1) * ldc + j + 2] = sum12;
            C[static_cast<long>(i + 1) * ldc + j + 3] = sum13;
        }

        for (; j < N; ++j) {
            const float* b = B + static_cast<long>(j) * ldb;

            float acc0 = 0.0f;
            float acc1 = 0.0f;

            for (int p = 0; p < K; ++p) {
                acc0 += a0[p] * b[p];
                acc1 += a1[p] * b[p];
            }

            C[static_cast<long>(i) * ldc + j] = acc0;
            C[static_cast<long>(i + 1) * ldc + j] = acc1;
        }
    }

    for (; i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;

        for (int j = 0; j < N; ++j) {
            const float* b = B + static_cast<long>(j) * ldb;
            float acc = 0.0f;

            for (int p = 0; p < K; ++p) {
                acc += a[p] * b[p];
            }

            C[static_cast<long>(i) * ldc + j] = acc;
        }
    }
}
