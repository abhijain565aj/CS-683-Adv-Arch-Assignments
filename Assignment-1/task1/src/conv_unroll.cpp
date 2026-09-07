// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    // TODO(student): replace this placeholder with your unrolled implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ox += 8) {

            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
            float acc4 = 0.0f;
            float acc5 = 0.0f;
            float acc6 = 0.0f;
            float acc7 = 0.0f;

            for (int ky = 0; ky < K - 1; ky += 2) {

                for (int kx = 0; kx < K - 1; kx += 2) {
                    float k0 = ker[ky * K + kx];
                    float k1 = ker[ky * K + kx + 1];

                    acc0 += in[(oy + ky) * in_stride + ox + kx]     * k0;
                    acc0 += in[(oy + ky) * in_stride + ox + kx + 1] * k1;

                    acc1 += in[(oy + ky) * in_stride + ox + 1 + kx]     * k0;
                    acc1 += in[(oy + ky) * in_stride + ox + 1 + kx + 1] * k1;

                    acc2 += in[(oy + ky) * in_stride + ox + 2 + kx]     * k0;
                    acc2 += in[(oy + ky) * in_stride + ox + 2 + kx + 1] * k1;

                    acc3 += in[(oy + ky) * in_stride + ox + 3 + kx]     * k0;
                    acc3 += in[(oy + ky) * in_stride + ox + 3 + kx + 1] * k1;

                    acc4 += in[(oy + ky) * in_stride + ox + 4 + kx]     * k0;
                    acc4 += in[(oy + ky) * in_stride + ox + 4 + kx + 1] * k1;

                    acc5 += in[(oy + ky) * in_stride + ox + 5 + kx]     * k0;
                    acc5 += in[(oy + ky) * in_stride + ox + 5 + kx + 1] * k1;

                    acc6 += in[(oy + ky) * in_stride + ox + 6 + kx]     * k0;
                    acc6 += in[(oy + ky) * in_stride + ox + 6 + kx + 1] * k1;

                    acc7 += in[(oy + ky) * in_stride + ox + 7 + kx]     * k0;
                    acc7 += in[(oy + ky) * in_stride + ox + 7 + kx + 1] * k1;
                }

                {
                    int kx = K - 1;
                    float k0 = ker[ky * K + kx];

                    acc0 += in[(oy + ky) * in_stride + ox + kx]     * k0;
                    acc1 += in[(oy + ky) * in_stride + ox + 1 + kx] * k0;
                    acc2 += in[(oy + ky) * in_stride + ox + 2 + kx] * k0;
                    acc3 += in[(oy + ky) * in_stride + ox + 3 + kx] * k0;
                    acc4 += in[(oy + ky) * in_stride + ox + 4 + kx] * k0;
                    acc5 += in[(oy + ky) * in_stride + ox + 5 + kx] * k0;
                    acc6 += in[(oy + ky) * in_stride + ox + 6 + kx] * k0;
                    acc7 += in[(oy + ky) * in_stride + ox + 7 + kx] * k0;
                }

                int ky2 = ky + 1;

                for (int kx = 0; kx < K - 1; kx += 2) {
                    float k0 = ker[ky2 * K + kx];
                    float k1 = ker[ky2 * K + kx + 1];

                    acc0 += in[(oy + ky2) * in_stride + ox + kx]     * k0;
                    acc0 += in[(oy + ky2) * in_stride + ox + kx + 1] * k1;

                    acc1 += in[(oy + ky2) * in_stride + ox + 1 + kx]     * k0;
                    acc1 += in[(oy + ky2) * in_stride + ox + 1 + kx + 1] * k1;

                    acc2 += in[(oy + ky2) * in_stride + ox + 2 + kx]     * k0;
                    acc2 += in[(oy + ky2) * in_stride + ox + 2 + kx + 1] * k1;

                    acc3 += in[(oy + ky2) * in_stride + ox + 3 + kx]     * k0;
                    acc3 += in[(oy + ky2) * in_stride + ox + 3 + kx + 1] * k1;

                    acc4 += in[(oy + ky2) * in_stride + ox + 4 + kx]     * k0;
                    acc4 += in[(oy + ky2) * in_stride + ox + 4 + kx + 1] * k1;

                    acc5 += in[(oy + ky2) * in_stride + ox + 5 + kx]     * k0;
                    acc5 += in[(oy + ky2) * in_stride + ox + 5 + kx + 1] * k1;

                    acc6 += in[(oy + ky2) * in_stride + ox + 6 + kx]     * k0;
                    acc6 += in[(oy + ky2) * in_stride + ox + 6 + kx + 1] * k1;

                    acc7 += in[(oy + ky2) * in_stride + ox + 7 + kx]     * k0;
                    acc7 += in[(oy + ky2) * in_stride + ox + 7 + kx + 1] * k1;
                }

                {
                    int kx = K - 1;
                    float k0 = ker[ky2 * K + kx];

                    acc0 += in[(oy + ky2) * in_stride + ox + kx]     * k0;
                    acc1 += in[(oy + ky2) * in_stride + ox + 1 + kx] * k0;
                    acc2 += in[(oy + ky2) * in_stride + ox + 2 + kx] * k0;
                    acc3 += in[(oy + ky2) * in_stride + ox + 3 + kx] * k0;
                    acc4 += in[(oy + ky2) * in_stride + ox + 4 + kx] * k0;
                    acc5 += in[(oy + ky2) * in_stride + ox + 5 + kx] * k0;
                    acc6 += in[(oy + ky2) * in_stride + ox + 6 + kx] * k0;
                    acc7 += in[(oy + ky2) * in_stride + ox + 7 + kx] * k0;
                }
            }

            {
                int ky = K - 1;

                for (int kx = 0; kx < K - 1; kx += 2) {
                    float k0 = ker[ky * K + kx];
                    float k1 = ker[ky * K + kx + 1];

                    acc0 += in[(oy + ky) * in_stride + ox + kx]     * k0;
                    acc0 += in[(oy + ky) * in_stride + ox + kx + 1] * k1;

                    acc1 += in[(oy + ky) * in_stride + ox + 1 + kx]     * k0;
                    acc1 += in[(oy + ky) * in_stride + ox + 1 + kx + 1] * k1;

                    acc2 += in[(oy + ky) * in_stride + ox + 2 + kx]     * k0;
                    acc2 += in[(oy + ky) * in_stride + ox + 2 + kx + 1] * k1;

                    acc3 += in[(oy + ky) * in_stride + ox + 3 + kx]     * k0;
                    acc3 += in[(oy + ky) * in_stride + ox + 3 + kx + 1] * k1;

                    acc4 += in[(oy + ky) * in_stride + ox + 4 + kx]     * k0;
                    acc4 += in[(oy + ky) * in_stride + ox + 4 + kx + 1] * k1;

                    acc5 += in[(oy + ky) * in_stride + ox + 5 + kx]     * k0;
                    acc5 += in[(oy + ky) * in_stride + ox + 5 + kx + 1] * k1;

                    acc6 += in[(oy + ky) * in_stride + ox + 6 + kx]     * k0;
                    acc6 += in[(oy + ky) * in_stride + ox + 6 + kx + 1] * k1;

                    acc7 += in[(oy + ky) * in_stride + ox + 7 + kx]     * k0;
                    acc7 += in[(oy + ky) * in_stride + ox + 7 + kx + 1] * k1;
                }

                {
                    int kx = K - 1;
                    float k0 = ker[ky * K + kx];

                    acc0 += in[(oy + ky) * in_stride + ox + kx]     * k0;
                    acc1 += in[(oy + ky) * in_stride + ox + 1 + kx] * k0;
                    acc2 += in[(oy + ky) * in_stride + ox + 2 + kx] * k0;
                    acc3 += in[(oy + ky) * in_stride + ox + 3 + kx] * k0;
                    acc4 += in[(oy + ky) * in_stride + ox + 4 + kx] * k0;
                    acc5 += in[(oy + ky) * in_stride + ox + 5 + kx] * k0;
                    acc6 += in[(oy + ky) * in_stride + ox + 6 + kx] * k0;
                    acc7 += in[(oy + ky) * in_stride + ox + 7 + kx] * k0;
                }
            }

            out[oy * W + ox]     = acc0;
            out[oy * W + ox + 1] = acc1;
            out[oy * W + ox + 2] = acc2;
            out[oy * W + ox + 3] = acc3;
            out[oy * W + ox + 4] = acc4;
            out[oy * W + ox + 5] = acc5;
            out[oy * W + ox + 6] = acc6;
            out[oy * W + ox + 7] = acc7;
        }
    }
    
}

