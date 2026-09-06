// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.
#include <immintrin.h>

#include "convolution.h"

void conv_optimized(const float* __restrict in,
                    float* __restrict out,
                    const float* __restrict ker,
                    int H, int W, int K) {

    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {

        for (int ox = 0; ox < W; ox += 8) {

            __m256 acc = _mm256_setzero_ps();

            for (int ky = 0; ky < K - 1; ky += 2) {

                const float* row0 =
                    in + (oy + ky) * in_stride + ox;

                const float* row1 =
                    in + (oy + ky + 1) * in_stride + ox;

                const float* ker0 =
                    ker + ky * K;

                const float* ker1 =
                    ker + (ky + 1) * K;

                for (int kx = 0; kx < K - 1; kx += 2) {

                    __m256 x00 =
                        _mm256_loadu_ps(row0 + kx);

                    __m256 x01 =
                        _mm256_loadu_ps(row0 + kx + 1);

                    __m256 x10 =
                        _mm256_loadu_ps(row1 + kx);

                    __m256 x11 =
                        _mm256_loadu_ps(row1 + kx + 1);

                    __m256 w00 =
                        _mm256_set1_ps(ker0[kx]);

                    __m256 w01 =
                        _mm256_set1_ps(ker0[kx + 1]);

                    __m256 w10 =
                        _mm256_set1_ps(ker1[kx]);

                    __m256 w11 =
                        _mm256_set1_ps(ker1[kx + 1]);

                    acc = _mm256_fmadd_ps(x00, w00, acc);
                    acc = _mm256_fmadd_ps(x01, w01, acc);

                    acc = _mm256_fmadd_ps(x10, w10, acc);
                    acc = _mm256_fmadd_ps(x11, w11, acc);
                }

                {
                    const int kx = K - 1;

                    __m256 x0 =
                        _mm256_loadu_ps(row0 + kx);

                    __m256 x1 =
                        _mm256_loadu_ps(row1 + kx);

                    __m256 w0 =
                        _mm256_set1_ps(ker0[kx]);

                    __m256 w1 =
                        _mm256_set1_ps(ker1[kx]);

                    acc = _mm256_fmadd_ps(x0, w0, acc);
                    acc = _mm256_fmadd_ps(x1, w1, acc);
                }
            }

            {
                const int ky = K - 1;

                const float* row =
                    in + (oy + ky) * in_stride + ox;

                const float* krow =
                    ker + ky * K;

                for (int kx = 0; kx < K - 1; kx += 2) {

                    __m256 x0 =
                        _mm256_loadu_ps(row + kx);

                    __m256 x1 =
                        _mm256_loadu_ps(row + kx + 1);

                    __m256 w0 =
                        _mm256_set1_ps(krow[kx]);

                    __m256 w1 =
                        _mm256_set1_ps(krow[kx + 1]);

                    acc = _mm256_fmadd_ps(x0, w0, acc);
                    acc = _mm256_fmadd_ps(x1, w1, acc);
                }

                {
                    const int kx = K - 1;

                    __m256 x =
                        _mm256_loadu_ps(row + kx);

                    __m256 w =
                        _mm256_set1_ps(krow[kx]);

                    acc = _mm256_fmadd_ps(x, w, acc);
                }
            }

            _mm256_storeu_ps(out + oy * W + ox, acc);
        }
    }
}