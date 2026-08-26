// conv_simd.cpp  STAGE 4: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "convolution.h"

void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your AVX2 implementation.
    // conv_naive(in, out, ker, H, W, K);
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

   
    std::memset(out, 0, sizeof(float) * H * W);

    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
          
            __m256 v_ker = _mm256_set1_ps(ker[ky * K + kx]);

            for (int oy = 0; oy < H; ++oy) {
                const float* in_ptr = &in[(oy + ky) * in_stride + kx];
                float* out_ptr = &out[oy * W];

                int ox = 0;

                for (; ox <= W - 8; ox += 8) {
                   
                    __m256 v_in  = _mm256_loadu_ps(in_ptr + ox);
                    __m256 v_out = _mm256_loadu_ps(out_ptr + ox);

                   
#if defined(__FMA__)
                    
#else
                    v_out = _mm256_add_ps(v_out, _mm256_mul_ps(v_in, v_ker));
#endif

                    _mm256_storeu_ps(out_ptr + ox, v_out);
                }

                
                for (; ox < W; ++ox) {
                    out_ptr[ox] += in_ptr[ox] * ker[ky * K + kx];
                }
            }
        }
    }
}
