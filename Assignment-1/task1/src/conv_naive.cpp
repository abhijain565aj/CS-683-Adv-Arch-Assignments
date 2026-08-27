// conv_naive.cpp  PROVIDED reference implementation.
//
// This is the correctness baseline AND the performance baseline that all speedups
// are measured against. Do NOT modify this file.

#include "convolution.h"

void conv_naive(const float* in, float* out, const float* ker,
                int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

    for (int oy = 0; oy < H; oy+=2) {
        for (int ox = 0; ox < W; ox+=2) {
            float acc = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K-1; kx+=2) {
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                    acc += in[(oy + ky) * in_stride + (ox + kx + 1)] * ker[ky * K + kx + 1];
                }
                acc += in[(oy + ky) * in_stride + (ox + K-1)] * ker[ky * K + K-1];
            }
            out[oy * W + ox] = acc;
            acc = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K-1; kx+=2) {
                    acc += in[(oy + ky) * in_stride + (ox + 1 + kx)] * ker[ky * K + kx];
                    acc += in[(oy + ky) * in_stride + (ox + 1 + kx + 1)] * ker[ky * K + kx + 1];
                }
                acc += in[(oy + ky) * in_stride + (ox + 1 + K-1)] * ker[ky * K + K-1];
            }
            out[oy * W + ox + 1] = acc;
            acc = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K-1; kx+=2) {
                    acc += in[(oy + ky + 1) * in_stride + (ox + kx)] * ker[ky * K + kx];
                    acc += in[(oy + ky + 1) * in_stride + (ox + kx + 1)] * ker[ky * K + kx + 1];
                }
                acc += in[(oy + ky + 1) * in_stride + (ox + K-1)] * ker[ky * K + K-1];
            }
            out[oy * W + ox + W] = acc;
            acc = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K-1; kx+=2) {
                    acc += in[(oy + ky +1) * in_stride + (ox + 1 + kx)] * ker[ky * K + kx];
                    acc += in[(oy + ky + 1) * in_stride + (ox + 1 + kx + 1)] * ker[ky * K + kx + 1];
                }
                acc += in[(oy + ky + 1) * in_stride + (ox + 1 + K-1)] * ker[ky * K + K-1];
            }
            out[oy * W + ox +W +1]  = acc;
        }
    }
}
