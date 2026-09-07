// task2c_driver.cpp
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
#error "Compile with -DBENCH_FN=<kernel function>"
#endif

void BENCH_FN(const float* A, const float* B, float* C,
              int M, int N, int K, int lda, int ldb, int ldc);

void matmul_naive(const float* A, const float* B, float* C,
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

static float max_abs(const float* x, std::size_t n) {
    float m = 0.0f;
    for (std::size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(x[i]));
    return m;
}

static float max_abs_diff(const float* a, const float* b, std::size_t n) {
    float m = 0.0f;
    for (std::size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

struct Buffers {
    int N, M, K, lda, ldb, ldc;
    std::vector<float> A, B, C;

    Buffers(int n, unsigned seed)
        : N(n), M(n), K(n), lda(n), ldb(n), ldc(n),
          A(static_cast<std::size_t>(n) * n),
          B(static_cast<std::size_t>(n) * n),
          C(static_cast<std::size_t>(n) * n, 0.0f) {
        fill_random(A.data(), A.size(), seed);
        fill_random(B.data(), B.size(), seed + 1u);
    }

    void run() {
        BENCH_FN(A.data(), B.data(), C.data(), M, N, K, lda, ldb, ldc);
    }
};

static int do_check(int N, unsigned seed) {
    Buffers x(N, seed);
    std::vector<float> ref(static_cast<std::size_t>(N) * N);

    matmul_naive(x.A.data(), x.B.data(), ref.data(), N, N, N, N, N, N);
    x.run();

    const float ref_mag = max_abs(ref.data(), ref.size());
    const float tol = 1e-4f * (ref_mag + 1e-30f);
    const float diff = max_abs_diff(x.C.data(), ref.data(), ref.size());
    const bool ok = diff <= tol;

    std::printf("{\"mode\":\"check\",\"N\":%d,\"correct\":%s,"
                "\"max_abs_diff\":%.9g,\"tolerance\":%.9g}\n",
                N, ok ? "true" : "false", diff, tol);
    return ok ? 0 : 1;
}

static int do_time(int N, int reps, int warmups, unsigned seed) {
    if (reps <= 0 || warmups < 0) return 2;
    Buffers x(N, seed);

    for (int r = 0; r < warmups; ++r) x.run();

    std::vector<double> times_ms;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        x.run();
        const auto t1 = std::chrono::steady_clock::now();
        times_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    std::sort(times_ms.begin(), times_ms.end());
    const double median_ms = times_ms[times_ms.size() / 2];
    const double flops = 2.0 * static_cast<double>(N) * N * N;
    const double gflops = flops / (median_ms * 1.0e6);

    std::printf("{\"mode\":\"time\",\"N\":%d,\"reps\":%d,\"warmups\":%d,"
                "\"time_ms\":%.9f,\"gflops\":%.9f,\"checksum\":%.9f}\n",
                N, reps, warmups, median_ms, gflops,
                checksum(x.C.data(), x.C.size()));
    return 0;
}

static int do_perf(int N, int reps, unsigned seed) {
    if (reps <= 0) return 2;
    Buffers x(N, seed);
    for (int r = 0; r < reps; ++r) x.run();
    std::printf("{\"mode\":\"perf\",\"N\":%d,\"reps\":%d,\"checksum\":%.9f}\n",
                N, reps, checksum(x.C.data(), x.C.size()));
    return 0;
}

static int do_setup(int N, unsigned seed) {
    Buffers x(N, seed);
    std::printf("{\"mode\":\"setup\",\"N\":%d,\"checksum\":%.9f}\n",
                N, checksum(x.C.data(), x.C.size()));
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    const char* mode = argv[1];

    if (std::strcmp(mode, "check") == 0) {
        const int N = (argc >= 3) ? std::atoi(argv[2]) : 129;
        const unsigned seed = (argc >= 4) ? std::strtoul(argv[3], nullptr, 10) : 1234u;
        return (N > 0) ? do_check(N, seed) : 2;
    }

    if (std::strcmp(mode, "time") == 0) {
        if (argc < 3) return 2;
        const int N = std::atoi(argv[2]);
        const int reps = (argc >= 4) ? std::atoi(argv[3]) : 5;
        const int warmups = (argc >= 5) ? std::atoi(argv[4]) : 1;
        const unsigned seed = (argc >= 6) ? std::strtoul(argv[5], nullptr, 10) : 1234u;
        return (N > 0) ? do_time(N, reps, warmups, seed) : 2;
    }

    if (std::strcmp(mode, "perf") == 0) {
        if (argc < 4) return 2;
        const int N = std::atoi(argv[2]);
        const int reps = std::atoi(argv[3]);
        const unsigned seed = (argc >= 5) ? std::strtoul(argv[4], nullptr, 10) : 1234u;
        return (N > 0) ? do_perf(N, reps, seed) : 2;
    }

    if (std::strcmp(mode, "setup") == 0) {
        if (argc < 3) return 2;
        const int N = std::atoi(argv[2]);
        const unsigned seed = (argc >= 4) ? std::strtoul(argv[3], nullptr, 10) : 1234u;
        return (N > 0) ? do_setup(N, seed) : 2;
    }

    return 2;
}
