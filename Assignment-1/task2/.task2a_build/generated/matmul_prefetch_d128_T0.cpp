// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>

#include "matmul.h"

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your cache-blocked SIMD + prefetch
    // implementation.
    const int BM = 32;
    const int BN = 32;
    const int prefetch_distance = 128;
    const int cache_line_floats = 16;

    for (int ii = 0; ii < M; ii += BM) {
        const int i_end = (ii + BM < M) ? ii + BM : M;

        for (int jj = 0; jj < N; jj += BN) {
            const int j_end = (jj + BN < N) ? jj + BN : N;

            for (int i = ii; i < i_end; ++i) {
                const float* a = A + static_cast<long>(i) * lda;

                int j;
                for (j = jj; j + 3 < j_end; j += 4) {
                    const float* b0 = B + static_cast<long>(j) * ldb;
                    const float* b1 = B + static_cast<long>(j + 1) * ldb;
                    const float* b2 = B + static_cast<long>(j + 2) * ldb;
                    const float* b3 = B + static_cast<long>(j + 3) * ldb;

                    float acc0 = 0.0f;
                    float acc1 = 0.0f;
                    float acc2 = 0.0f;
                    float acc3 = 0.0f;

                    int p = 0;

                    for (; p + cache_line_floats - 1 < K &&
                           p + prefetch_distance < K;
                         p += cache_line_floats) {

                        _mm_prefetch(
                            reinterpret_cast<const char*>(
                                b0 + p + prefetch_distance),
                            _MM_HINT_T0);
                        _mm_prefetch(
                            reinterpret_cast<const char*>(
                                b1 + p + prefetch_distance),
                            _MM_HINT_T0);
                        _mm_prefetch(
                            reinterpret_cast<const char*>(
                                b2 + p + prefetch_distance),
                            _MM_HINT_T0);
                        _mm_prefetch(
                            reinterpret_cast<const char*>(
                                b3 + p + prefetch_distance),
                            _MM_HINT_T0);

                        for (int q = 0; q < cache_line_floats; ++q) {
                            const float av = a[p + q];
                            acc0 += av * b0[p + q];
                            acc1 += av * b1[p + q];
                            acc2 += av * b2[p + q];
                            acc3 += av * b3[p + q];
                        }
                    }

                    for (; p + cache_line_floats - 1 < K;
                         p += cache_line_floats) {
                        for (int q = 0; q < cache_line_floats; ++q) {
                            const float av = a[p + q];
                            acc0 += av * b0[p + q];
                            acc1 += av * b1[p + q];
                            acc2 += av * b2[p + q];
                            acc3 += av * b3[p + q];
                        }
                    }

                    for (; p < K; ++p) {
                        const float av = a[p];
                        acc0 += av * b0[p];
                        acc1 += av * b1[p];
                        acc2 += av * b2[p];
                        acc3 += av * b3[p];
                    }

                    C[static_cast<long>(i) * ldc + j] = acc0;
                    C[static_cast<long>(i) * ldc + j + 1] = acc1;
                    C[static_cast<long>(i) * ldc + j + 2] = acc2;
                    C[static_cast<long>(i) * ldc + j + 3] = acc3;
                }

                for (; j < j_end; ++j) {
                    const float* b = B + static_cast<long>(j) * ldb;
                    float acc = 0.0f;

                    int p = 0;

                    for (; p + cache_line_floats - 1 < K &&
                           p + prefetch_distance < K;
                         p += cache_line_floats) {
                        _mm_prefetch(
                            reinterpret_cast<const char*>(
                                b + p + prefetch_distance),
                            _MM_HINT_T0);

                        for (int q = 0; q < cache_line_floats; ++q) {
                            acc += a[p + q] * b[p + q];
                        }
                    }

                    for (; p + cache_line_floats - 1 < K;
                         p += cache_line_floats) {
                        for (int q = 0; q < cache_line_floats; ++q) {
                            acc += a[p + q] * b[p + q];
                        }
                    }

                    for (; p < K; ++p) {
                        acc += a[p] * b[p];
                    }

                    C[static_cast<long>(i) * ldc + j] = acc;
                }
            }
        }
    }
}