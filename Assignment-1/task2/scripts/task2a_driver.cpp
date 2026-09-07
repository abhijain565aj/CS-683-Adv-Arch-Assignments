// task2a_driver.cpp
// Benchmark-only driver for Task 2A.
// It does NOT modify any assignment source file.
//
// Compile with either:
//   -DBENCH_FN=matmul_naive
// or
//   -DBENCH_FN=matmul_prefetch
//
// Usage:
//   ./bench <N> [reps] [warmups] [seed]
//
// The assignment uses the NT layout:
// C[i][j] = sum_p A[i][p] * B[j][p]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef BENCH_FN
#error "Compile with -DBENCH_FN=matmul_naive or -DBENCH_FN=matmul_prefetch"
#endif

void BENCH_FN(const float* A, const float* B, float* C,
              int M, int N, int K, int lda, int ldb, int ldc);

static std::uint32_t xorshift32(std::uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

static void fill_random(float* x, std::size_t n, std::uint32_t seed) {
    std::uint32_t s = seed ? seed : 1u;
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t r = xorshift32(s);
        x[i] = static_cast<float>((r & 0x00ffffffu) / 16777216.0) - 0.5f;
    }
}

static double checksum(const float* x, std::size_t n) {
    double s = 0.0;
    const std::size_t step = (n > 4096) ? (n / 4096) : 1;
    for (std::size_t i = 0; i < n; i += step) s += x[i];
    return s;
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 5) {
        std::fprintf(stderr, "Usage: %s <N> [reps] [warmups] [seed]\n", argv[0]);
        return 2;
    }

    const int N = std::atoi(argv[1]);
    const int reps = (argc >= 3) ? std::atoi(argv[2]) : 3;
    const int warmups = (argc >= 4) ? std::atoi(argv[3]) : 1;
    const unsigned seed = (argc >= 5)
        ? static_cast<unsigned>(std::strtoul(argv[4], nullptr, 10))
        : 1234u;

    if (N <= 0 || reps <= 0 || warmups < 0) {
        std::fprintf(stderr, "N and reps must be positive; warmups must be non-negative.\n");
        return 2;
    }

    const int M = N;
    const int K = N;
    const int lda = K;
    const int ldb = K;
    const int ldc = N;

    std::vector<float> A(static_cast<std::size_t>(M) * K);
    std::vector<float> B(static_cast<std::size_t>(N) * K);
    std::vector<float> C(static_cast<std::size_t>(M) * N);

    fill_random(A.data(), A.size(), seed);
    fill_random(B.data(), B.size(), seed + 1u);

    for (int r = 0; r < warmups; ++r) {
        BENCH_FN(A.data(), B.data(), C.data(), M, N, K, lda, ldb, ldc);
    }

    std::vector<double> times_ms;
    times_ms.reserve(static_cast<std::size_t>(reps));

    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        BENCH_FN(A.data(), B.data(), C.data(), M, N, K, lda, ldb, ldc);
        const auto t1 = std::chrono::steady_clock::now();

        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        times_ms.push_back(ms);
    }

    std::sort(times_ms.begin(), times_ms.end());
    const double median_ms = times_ms[times_ms.size() / 2];
    const double flops = 2.0 * static_cast<double>(M) * N * K;
    const double gflops = flops / (median_ms * 1.0e6);
    const double sum = checksum(C.data(), C.size());

    std::printf(
        "{\"N\":%d,\"reps\":%d,\"warmups\":%d,"
        "\"time_ms\":%.9f,\"gflops\":%.9f,\"checksum\":%.9f}\n",
        N, reps, warmups, median_ms, gflops, sum
    );
    return 0;
}
