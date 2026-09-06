// conv_reorder.cpp  STAGE 1: LOOP REORDERING
// Hint: loops from outermost to innermost -> ky, kx, oy, ox.

#include "convolution.h"

void conv_reorder(const float* in, float* out, const float* ker,
                  int H, int W, int K) {
    // TODO(student): replace this placeholder with your reordered implementation.
    // conv_naive(in, out, ker, H, W, K);
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ++ox) {
                out[oy * W + ox] = 0.0;
            }
        }

    for (int oy = 0; oy < H; ++oy) {
        for (int ky = 0; ky < K; ++ky) {
            for (int kx = 0; kx < K; ++kx) {
                float ker_val = ker[ky * K + kx];
                for (int ox = 0; ox < W; ++ox) {
                    out[oy * W + ox] += in[(oy + ky) * in_stride + (ox + kx)] * ker_val;
                }
            }
        }
    }

}
