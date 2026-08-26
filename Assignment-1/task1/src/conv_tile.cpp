// conv_tile.cpp  STAGE 3: CACHE TILING

#include "convolution.h"

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your tiled/blocked implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    int sz=32;
    for (int oyT=0; oyT < H/sz; ++oyT)
    {
        for (int oxT=0; oxT < W/sz; ++oxT)
        {
            for (int oy = oyT*sz; oy < (oyT+1)*sz; ++oy) {
                for (int ox = oxT*sz; ox < (oxT+1)*sz; ++ox) {
                    float acc = 0.0f;
                    for (int ky = 0; ky < K; ++ky) {
                        for (int kx = 0; kx < K; ++kx) {
                            acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                        }
                    }
                    out[oy * W + ox] = acc;
                }
            }
        }
    }
}
