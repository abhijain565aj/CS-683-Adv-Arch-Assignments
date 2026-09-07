// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include <immintrin.h>

#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    // TODO(student): replace this placeholder with your best combined implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    const int TILE = 32;

    for (int oyT = 0; oyT < H; oyT += TILE) {

        const int oyEnd =
            (oyT + TILE < H) ? oyT + TILE : H;

        for (int oxT = 0; oxT < W; oxT += TILE) {

            const int oxEnd =
                (oxT + TILE < W) ? oxT + TILE : W;

            int oy = oyT;

            for (; oy + 3 < oyEnd; oy += 4) {

                for (int ox = oxT; ox < oxEnd; ox += 8) {

                    __m256 acc0 = _mm256_setzero_ps();
                    __m256 acc1 = _mm256_setzero_ps();
                    __m256 acc2 = _mm256_setzero_ps();
                    __m256 acc3 = _mm256_setzero_ps();

                    for (int ky = 0; ky < K; ++ky) {

                        const float* row0 =
                            in + (oy + ky) * in_stride + ox;

                        const float* row1 =
                            in + (oy + 1 + ky) * in_stride + ox;

                        const float* row2 =
                            in + (oy + 2 + ky) * in_stride + ox;

                        const float* row3 =
                            in + (oy + 3 + ky) * in_stride + ox;

                        const float* krow =
                            ker + ky * K;

                        
                        int kx = 0;

                        for (; kx < K - 1; kx += 2) {

                            const __m256 w0 =
                                _mm256_set1_ps(krow[kx]);

                            const __m256 w1 =
                                _mm256_set1_ps(krow[kx + 1]);

                            /*
                             * First kernel column.
                             */
                            const __m256 x00 =
                                _mm256_loadu_ps(row0 + kx);

                            const __m256 x10 =
                                _mm256_loadu_ps(row1 + kx);

                            const __m256 x20 =
                                _mm256_loadu_ps(row2 + kx);

                            const __m256 x30 =
                                _mm256_loadu_ps(row3 + kx);

                            acc0 = _mm256_fmadd_ps(x00, w0, acc0);
                            acc1 = _mm256_fmadd_ps(x10, w0, acc1);
                            acc2 = _mm256_fmadd_ps(x20, w0, acc2);
                            acc3 = _mm256_fmadd_ps(x30, w0, acc3);

                            const __m256 x01 =
                                _mm256_loadu_ps(row0 + kx + 1);

                            const __m256 x11 =
                                _mm256_loadu_ps(row1 + kx + 1);

                            const __m256 x21 =
                                _mm256_loadu_ps(row2 + kx + 1);

                            const __m256 x31 =
                                _mm256_loadu_ps(row3 + kx + 1);

                            acc0 = _mm256_fmadd_ps(x01, w1, acc0);
                            acc1 = _mm256_fmadd_ps(x11, w1, acc1);
                            acc2 = _mm256_fmadd_ps(x21, w1, acc2);
                            acc3 = _mm256_fmadd_ps(x31, w1, acc3);
                        }

                        const __m256 w =
                            _mm256_set1_ps(krow[K - 1]);

                        const __m256 x0 =
                            _mm256_loadu_ps(row0 + K - 1);

                        const __m256 x1 =
                            _mm256_loadu_ps(row1 + K - 1);

                        const __m256 x2 =
                            _mm256_loadu_ps(row2 + K - 1);

                        const __m256 x3 =
                            _mm256_loadu_ps(row3 + K - 1);

                        acc0 = _mm256_fmadd_ps(x0, w, acc0);
                        acc1 = _mm256_fmadd_ps(x1, w, acc1);
                        acc2 = _mm256_fmadd_ps(x2, w, acc2);
                        acc3 = _mm256_fmadd_ps(x3, w, acc3);
                    }

                    _mm256_storeu_ps(out + oy * W + ox,
                                     acc0);

                    _mm256_storeu_ps(out + (oy + 1) * W + ox,
                                     acc1);

                    _mm256_storeu_ps(out + (oy + 2) * W + ox,
                                     acc2);

                    _mm256_storeu_ps(out + (oy + 3) * W + ox,
                                     acc3);
                }
            }

            for (; oy < oyEnd; ++oy) {

                for (int ox = oxT; ox < oxEnd; ox += 8) {

                    __m256 acc = _mm256_setzero_ps();

                    for (int ky = 0; ky < K; ++ky) {

                        const float* row =
                            in + (oy + ky) * in_stride + ox;

                        const float* krow =
                            ker + ky * K;

                        int kx = 0;

                        for (; kx < K - 1; kx += 2) {

                            const __m256 w0 =
                                _mm256_set1_ps(krow[kx]);

                            const __m256 w1 =
                                _mm256_set1_ps(krow[kx + 1]);

                            const __m256 x0 =
                                _mm256_loadu_ps(row + kx);

                            const __m256 x1 =
                                _mm256_loadu_ps(row + kx + 1);

                            acc = _mm256_fmadd_ps(x0, w0, acc);
                            acc = _mm256_fmadd_ps(x1, w1, acc);
                        }

                        const __m256 w =
                            _mm256_set1_ps(krow[K - 1]);

                        const __m256 x =
                            _mm256_loadu_ps(row + K - 1);

                        acc = _mm256_fmadd_ps(x, w, acc);
                    }

                    _mm256_storeu_ps(out + oy * W + ox, acc);
                }
            }
        }
    }
}