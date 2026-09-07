// task2c_alt_widths.cpp
// Measurement-only alternate-width Task 2C kernel.
// Compile with -DTASK2C_BITS=128 or -DTASK2C_BITS=512.
// The 256-bit measurement deliberately uses the student's real matmul_optimized.cpp.

#include <immintrin.h>
#include "matmul.h"

#ifndef TASK2C_BITS
#error "Compile with -DTASK2C_BITS=128 or 512"
#endif

#if TASK2C_BITS == 128
using Vec = __m128;
static constexpr int VW = 4;
static inline Vec vzero() { return _mm_setzero_ps(); }
static inline Vec vload(const float* p) { return _mm_loadu_ps(p); }
static inline Vec vfmadd(Vec a, Vec b, Vec c) { return _mm_fmadd_ps(a, b, c); }
static inline float hsum(Vec v) {
    v = _mm_hadd_ps(v, v);
    v = _mm_hadd_ps(v, v);
    return _mm_cvtss_f32(v);
}
#elif TASK2C_BITS == 512
using Vec = __m512;
static constexpr int VW = 16;
static inline Vec vzero() { return _mm512_setzero_ps(); }
static inline Vec vload(const float* p) { return _mm512_loadu_ps(p); }
static inline Vec vfmadd(Vec a, Vec b, Vec c) { return _mm512_fmadd_ps(a, b, c); }
static inline float hsum(Vec v) { return _mm512_reduce_add_ps(v); }
#else
#error "TASK2C_BITS must be 128 or 512"
#endif

void task2c_alt_kernel(const float* A, const float* B, float* C,
                       int M, int N, int K, int lda, int ldb, int ldc) {
    const int BM = 32;
    const int BN = 32;
    const int prefetch_distance = 64;

    for (int ii = 0; ii < M; ii += BM) {
        int i_end = (ii + BM < M) ? ii + BM : M;
        for (int jj = 0; jj < N; jj += BN) {
            int j_end = (jj + BN < N) ? jj + BN : N;

            int i;
            for (i = ii; i + 1 < i_end; i += 2) {
                const float* a0 = A + static_cast<long>(i) * lda;
                const float* a1 = A + static_cast<long>(i + 1) * lda;

                int j;
                for (j = jj; j + 3 < j_end; j += 4) {
                    const float* b0 = B + static_cast<long>(j) * ldb;
                    const float* b1 = B + static_cast<long>(j + 1) * ldb;
                    const float* b2 = B + static_cast<long>(j + 2) * ldb;
                    const float* b3 = B + static_cast<long>(j + 3) * ldb;

                    Vec acc00 = vzero(), acc01 = vzero(), acc02 = vzero(), acc03 = vzero();
                    Vec acc10 = vzero(), acc11 = vzero(), acc12 = vzero(), acc13 = vzero();

                    int p;
                    for (p = 0; p + 2 * VW - 1 < K; p += 2 * VW) {

                        Vec a0v = vload(a0 + p);
                        Vec a1v = vload(a1 + p);

                        Vec bv = vload(b0 + p);
                        acc00 = vfmadd(a0v, bv, acc00);
                        acc10 = vfmadd(a1v, bv, acc10);
                        bv = vload(b1 + p);
                        acc01 = vfmadd(a0v, bv, acc01);
                        acc11 = vfmadd(a1v, bv, acc11);
                        bv = vload(b2 + p);
                        acc02 = vfmadd(a0v, bv, acc02);
                        acc12 = vfmadd(a1v, bv, acc12);
                        bv = vload(b3 + p);
                        acc03 = vfmadd(a0v, bv, acc03);
                        acc13 = vfmadd(a1v, bv, acc13);

                        a0v = vload(a0 + p + VW);
                        a1v = vload(a1 + p + VW);

                        bv = vload(b0 + p + VW);
                        acc00 = vfmadd(a0v, bv, acc00);
                        acc10 = vfmadd(a1v, bv, acc10);
                        bv = vload(b1 + p + VW);
                        acc01 = vfmadd(a0v, bv, acc01);
                        acc11 = vfmadd(a1v, bv, acc11);
                        bv = vload(b2 + p + VW);
                        acc02 = vfmadd(a0v, bv, acc02);
                        acc12 = vfmadd(a1v, bv, acc12);
                        bv = vload(b3 + p + VW);
                        acc03 = vfmadd(a0v, bv, acc03);
                        acc13 = vfmadd(a1v, bv, acc13);
                    }

                    for (; p + VW - 1 < K; p += VW) {
                        Vec a0v = vload(a0 + p);
                        Vec a1v = vload(a1 + p);

                        Vec bv = vload(b0 + p);
                        acc00 = vfmadd(a0v, bv, acc00);
                        acc10 = vfmadd(a1v, bv, acc10);
                        bv = vload(b1 + p);
                        acc01 = vfmadd(a0v, bv, acc01);
                        acc11 = vfmadd(a1v, bv, acc11);
                        bv = vload(b2 + p);
                        acc02 = vfmadd(a0v, bv, acc02);
                        acc12 = vfmadd(a1v, bv, acc12);
                        bv = vload(b3 + p);
                        acc03 = vfmadd(a0v, bv, acc03);
                        acc13 = vfmadd(a1v, bv, acc13);
                    }

                    float s00 = hsum(acc00), s01 = hsum(acc01), s02 = hsum(acc02), s03 = hsum(acc03);
                    float s10 = hsum(acc10), s11 = hsum(acc11), s12 = hsum(acc12), s13 = hsum(acc13);

                    for (; p < K; ++p) {
                        float x0 = a0[p], x1 = a1[p];
                        s00 += x0 * b0[p]; s01 += x0 * b1[p];
                        s02 += x0 * b2[p]; s03 += x0 * b3[p];
                        s10 += x1 * b0[p]; s11 += x1 * b1[p];
                        s12 += x1 * b2[p]; s13 += x1 * b3[p];
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

                for (; j < j_end; ++j) {
                    const float* b = B + static_cast<long>(j) * ldb;
                    float s0 = 0.0f, s1 = 0.0f;
                    for (int p = 0; p < K; ++p) {
                        s0 += a0[p] * b[p];
                        s1 += a1[p] * b[p];
                    }
                    C[static_cast<long>(i) * ldc + j] = s0;
                    C[static_cast<long>(i + 1) * ldc + j] = s1;
                }
            }

            for (; i < i_end; ++i) {
                const float* a = A + static_cast<long>(i) * lda;
                for (int j = jj; j < j_end; ++j) {
                    const float* b = B + static_cast<long>(j) * ldb;
                    float s = 0.0f;
                    for (int p = 0; p < K; ++p) s += a[p] * b[p];
                    C[static_cast<long>(i) * ldc + j] = s;
                }
            }
        }
    }
}
