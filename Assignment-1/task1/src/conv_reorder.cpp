// conv_reorder.cpp  STAGE 1: LOOP REORDERING
// Hint: loops from outermost to innermost -> ky, kx, oy, ox.

#include "convolution.h"

void conv_reorder(const float* in, float* out, const float* ker,
                  int H, int W, int K) {
    // TODO(student): replace this placeholder with your reordered implementation.
    // conv_naive(in, out, ker, H, W, K);
    // const int p = K / 2;
    // const int in_stride = W + 2 * p;  // padded row stride

    // for (int oy = 0; oy < H; ++oy) {
    //     for (int ox = 0; ox < W; ++ox) {
    //         float acc = 0.0f;
    //         for (int ky = 0; ky < K; ++ky) {
    //             for (int kx = 0; kx < K; ++kx) {
    //                 acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
    //             }
    //         }
    //         out[oy * W + ox] = acc;
    //     }
    // }
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

    // 1. Clear output buffer since we accumulate directly into out[oy * W + ox]
    std::memset(out, 0, sizeof(float) * H * W);

    // 2. Outermost loops iterate over the kernel coordinates (ky, kx)
    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
            // Hoist kernel weight into a register for maximum reuse
            const float k_val = ker[ky * K + kx];

            // 3. Innermost loops sweep across the output image (oy, ox)
            for (int oy = 0; oy < H; ++oy) {
                const float* in_ptr = &in[(oy + ky) * in_stride + kx];
                float* out_ptr = &out[oy * W];

                for (int ox = 0; ox < W; ++ox) {
                    out_ptr[ox] += in_ptr[ox] * k_val;
                }
            }
        }
    }
}
