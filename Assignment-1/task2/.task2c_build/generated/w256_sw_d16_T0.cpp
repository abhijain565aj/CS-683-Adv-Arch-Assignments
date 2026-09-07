// matmul_optimized.cpp  STAGE 3: PUT IT ALL TOGETHER
//
// This is the graded function AND the kernel that gets injected into llama.cpp. Combine
// everything you have learned across the whole assignment  loop reordering, register
// blocking and unrolling (Task 1 / Stage 1 here), cache tiling and software prefetch
// (Stage 2)  and TUNE it to be as fast as you can. Your speedup over matmul_naive determines
// your score (see the tier table the harness prints), and this same function will power a
// real LLM inference via `make llama-demo`.

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

void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your best combined implementation.
    const int BM = 32;
    const int BN = 32;
    const int prefetch_distance = 16;

    for (int ii = 0; ii < M; ii += BM) {
        const int i_end = (ii + BM < M) ? ii + BM : M;

        for (int jj = 0; jj < N; jj += BN) {
            const int j_end = (jj + BN < N) ? jj + BN : N;

            int j;
            for (j = jj; j + 3 < j_end; j += 4) {
                const float* b0 = B + static_cast<long>(j) * ldb;
                const float* b1 = B + static_cast<long>(j + 1) * ldb;
                const float* b2 = B + static_cast<long>(j + 2) * ldb;
                const float* b3 = B + static_cast<long>(j + 3) * ldb;

                int i;
                for (i = ii; i + 1 < i_end; i += 2) {
                    const float* a0 = A + static_cast<long>(i) * lda;
                    const float* a1 = A + static_cast<long>(i + 1) * lda;

                    __m256 acc00 = _mm256_setzero_ps();
                    __m256 acc01 = _mm256_setzero_ps();
                    __m256 acc02 = _mm256_setzero_ps();
                    __m256 acc03 = _mm256_setzero_ps();

                    __m256 acc10 = _mm256_setzero_ps();
                    __m256 acc11 = _mm256_setzero_ps();
                    __m256 acc12 = _mm256_setzero_ps();
                    __m256 acc13 = _mm256_setzero_ps();

                    int p = 0;

                    for (; p + 31 < K; p += 32) {
                        if (p + prefetch_distance < K) {
                            _mm_prefetch(
                                reinterpret_cast<const char*>(
                                    a0 + p + prefetch_distance),
                                _MM_HINT_T0);
                            _mm_prefetch(
                                reinterpret_cast<const char*>(
                                    a1 + p + prefetch_distance),
                                _MM_HINT_T0);
                        }

                        __m256 a0_cur = _mm256_loadu_ps(a0 + p);
                        __m256 a1_cur = _mm256_loadu_ps(a1 + p);

                        __m256 b_cur  = _mm256_loadu_ps(b0 + p);
                        __m256 b_next = _mm256_loadu_ps(b1 + p);

                        acc00 = _mm256_fmadd_ps(a0_cur, b_cur, acc00);
                        acc10 = _mm256_fmadd_ps(a1_cur, b_cur, acc10);

                        b_cur = _mm256_loadu_ps(b2 + p);

                        acc01 = _mm256_fmadd_ps(a0_cur, b_next, acc01);
                        acc11 = _mm256_fmadd_ps(a1_cur, b_next, acc11);

                        b_next = _mm256_loadu_ps(b3 + p);

                        acc02 = _mm256_fmadd_ps(a0_cur, b_cur, acc02);
                        acc12 = _mm256_fmadd_ps(a1_cur, b_cur, acc12);

                        __m256 a0_next = _mm256_loadu_ps(a0 + p + 8);
                        __m256 a1_next = _mm256_loadu_ps(a1 + p + 8);

                        acc03 = _mm256_fmadd_ps(a0_cur, b_next, acc03);
                        acc13 = _mm256_fmadd_ps(a1_cur, b_next, acc13);

                        a0_cur = a0_next;
                        a1_cur = a1_next;

                        b_cur  = _mm256_loadu_ps(b0 + p + 8);
                        b_next = _mm256_loadu_ps(b1 + p + 8);

                        acc00 = _mm256_fmadd_ps(a0_cur, b_cur, acc00);
                        acc10 = _mm256_fmadd_ps(a1_cur, b_cur, acc10);

                        b_cur = _mm256_loadu_ps(b2 + p + 8);

                        acc01 = _mm256_fmadd_ps(a0_cur, b_next, acc01);
                        acc11 = _mm256_fmadd_ps(a1_cur, b_next, acc11);

                        b_next = _mm256_loadu_ps(b3 + p + 8);

                        acc02 = _mm256_fmadd_ps(a0_cur, b_cur, acc02);
                        acc12 = _mm256_fmadd_ps(a1_cur, b_cur, acc12);

                        a0_next = _mm256_loadu_ps(a0 + p + 16);
                        a1_next = _mm256_loadu_ps(a1 + p + 16);

                        acc03 = _mm256_fmadd_ps(a0_cur, b_next, acc03);
                        acc13 = _mm256_fmadd_ps(a1_cur, b_next, acc13);

                        a0_cur = a0_next;
                        a1_cur = a1_next;

                        b_cur  = _mm256_loadu_ps(b0 + p + 16);
                        b_next = _mm256_loadu_ps(b1 + p + 16);

                        acc00 = _mm256_fmadd_ps(a0_cur, b_cur, acc00);
                        acc10 = _mm256_fmadd_ps(a1_cur, b_cur, acc10);

                        b_cur = _mm256_loadu_ps(b2 + p + 16);

                        acc01 = _mm256_fmadd_ps(a0_cur, b_next, acc01);
                        acc11 = _mm256_fmadd_ps(a1_cur, b_next, acc11);

                        b_next = _mm256_loadu_ps(b3 + p + 16);

                        acc02 = _mm256_fmadd_ps(a0_cur, b_cur, acc02);
                        acc12 = _mm256_fmadd_ps(a1_cur, b_cur, acc12);

                        a0_next = _mm256_loadu_ps(a0 + p + 24);
                        a1_next = _mm256_loadu_ps(a1 + p + 24);

                        acc03 = _mm256_fmadd_ps(a0_cur, b_next, acc03);
                        acc13 = _mm256_fmadd_ps(a1_cur, b_next, acc13);

                        a0_cur = a0_next;
                        a1_cur = a1_next;

                        b_cur  = _mm256_loadu_ps(b0 + p + 24);
                        b_next = _mm256_loadu_ps(b1 + p + 24);

                        acc00 = _mm256_fmadd_ps(a0_cur, b_cur, acc00);
                        acc10 = _mm256_fmadd_ps(a1_cur, b_cur, acc10);

                        b_cur = _mm256_loadu_ps(b2 + p + 24);

                        acc01 = _mm256_fmadd_ps(a0_cur, b_next, acc01);
                        acc11 = _mm256_fmadd_ps(a1_cur, b_next, acc11);

                        b_next = _mm256_loadu_ps(b3 + p + 24);

                        acc02 = _mm256_fmadd_ps(a0_cur, b_cur, acc02);
                        acc12 = _mm256_fmadd_ps(a1_cur, b_cur, acc12);

                        acc03 = _mm256_fmadd_ps(a0_cur, b_next, acc03);
                        acc13 = _mm256_fmadd_ps(a1_cur, b_next, acc13);
                    }

                    for (; p + 7 < K; p += 8) {
                        __m256 a0_vec = _mm256_loadu_ps(a0 + p);
                        __m256 a1_vec = _mm256_loadu_ps(a1 + p);

                        __m256 b_cur  = _mm256_loadu_ps(b0 + p);
                        __m256 b_next = _mm256_loadu_ps(b1 + p);

                        acc00 = _mm256_fmadd_ps(a0_vec, b_cur, acc00);
                        acc10 = _mm256_fmadd_ps(a1_vec, b_cur, acc10);

                        b_cur = _mm256_loadu_ps(b2 + p);

                        acc01 = _mm256_fmadd_ps(a0_vec, b_next, acc01);
                        acc11 = _mm256_fmadd_ps(a1_vec, b_next, acc11);

                        b_next = _mm256_loadu_ps(b3 + p);

                        acc02 = _mm256_fmadd_ps(a0_vec, b_cur, acc02);
                        acc12 = _mm256_fmadd_ps(a1_vec, b_cur, acc12);

                        acc03 = _mm256_fmadd_ps(a0_vec, b_next, acc03);
                        acc13 = _mm256_fmadd_ps(a1_vec, b_next, acc13);
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
                        const float a0_value = a0[p];
                        const float a1_value = a1[p];

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

                for (; i < i_end; ++i) {
                    const float* a = A + static_cast<long>(i) * lda;

                    float sum0 = 0.0f;
                    float sum1 = 0.0f;
                    float sum2 = 0.0f;
                    float sum3 = 0.0f;

                    for (int p = 0; p < K; ++p) {
                        const float av = a[p];

                        sum0 += av * b0[p];
                        sum1 += av * b1[p];
                        sum2 += av * b2[p];
                        sum3 += av * b3[p];
                    }

                    C[static_cast<long>(i) * ldc + j] = sum0;
                    C[static_cast<long>(i) * ldc + j + 1] = sum1;
                    C[static_cast<long>(i) * ldc + j + 2] = sum2;
                    C[static_cast<long>(i) * ldc + j + 3] = sum3;
                }
            }

            for (; j < j_end; ++j) {
                const float* b = B + static_cast<long>(j) * ldb;

                for (int i = ii; i < i_end; ++i) {
                    const float* a = A + static_cast<long>(i) * lda;
                    float acc = 0.0f;

                    for (int p = 0; p < K; ++p) {
                        acc += a[p] * b[p];
                    }

                    C[static_cast<long>(i) * ldc + j] = acc;
                }
            }
        }
    }
}