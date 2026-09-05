#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "convolution.h"
#include "utils.h"

// This executable is intentionally separate from the assignment harness.
// It runs exactly ONE requested convolution implementation per process.
//
// The actual perf counters are collected by perf_compare.sh.
//
// Warmups are used to put the code/data into a steady state before the
// measured repetitions. The measured repetitions are repeated many times
// so process/setup overhead is small compared with kernel execution.

static constexpr int kDefaultH = 2048;
static constexpr int kDefaultW = 2048;
static constexpr int kDefaultK = 3;
static constexpr unsigned kDefaultSeed = 1234u;

static constexpr int kWarmupRuns = 5;
static constexpr int kMeasuredRuns = 20;

struct Stage {
    const char* key;
    ConvFn fn;
};

static const Stage kStages[] = {
    {"naive",     conv_naive},
    {"reorder",   conv_reorder},
    {"unroll",    conv_unroll},
    {"tile",      conv_tile},
    {"simd",      conv_simd},
    {"optimized", conv_optimized},
};

static constexpr int kNumStages =
    static_cast<int>(sizeof(kStages) / sizeof(kStages[0]));

static const Stage* find_stage(const char* key) {
    for (int i = 0; i < kNumStages; ++i) {
        if (std::strcmp(key, kStages[i].key) == 0)
            return &kStages[i];
    }
    return nullptr;
}

static void usage(const char* prog) {
    std::printf(
        "Usage: %s <stage> [H W K] [seed]\n\n"
        "Stages:\n"
        "  naive reorder unroll tile simd optimized\n\n"
        "Examples:\n"
        "  %s naive\n"
        "  %s unroll 2048 2048 3\n"
        "  %s optimized 1024 1024 5 42\n\n"
        "Warmup runs    : %d\n"
        "Measured runs  : %d\n",
        prog, prog, prog, prog, kWarmupRuns, kMeasuredRuns
    );
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 6) {
        usage(argv[0]);
        return 1;
    }

    if (std::strcmp(argv[1], "help") == 0 ||
        std::strcmp(argv[1], "-h") == 0 ||
        std::strcmp(argv[1], "--help") == 0) {
        usage(argv[0]);
        return 0;
    }

    const Stage* stage = find_stage(argv[1]);
    if (!stage) {
        std::fprintf(stderr, "error: unknown stage '%s'\n\n", argv[1]);
        usage(argv[0]);
        return 1;
    }

    int H = kDefaultH;
    int W = kDefaultW;
    int K = kDefaultK;
    unsigned seed = kDefaultSeed;

    if (argc >= 5) {
        H = std::atoi(argv[2]);
        W = std::atoi(argv[3]);
        K = std::atoi(argv[4]);
    } else if (argc != 2) {
        std::fprintf(stderr, "error: give either no dimensions or H W K [seed]\n");
        return 1;
    }

    if (argc == 6)
        seed = static_cast<unsigned>(
            std::strtoul(argv[5], nullptr, 10));

    if (H <= 0 || W <= 0 || K <= 0) {
        std::fprintf(stderr, "error: H, W and K must be positive\n");
        return 1;
    }

    if (W % 8 != 0) {
        std::fprintf(stderr, "error: W (%d) must be a multiple of 8\n", W);
        return 1;
    }

    if (K % 2 == 0) {
        std::fprintf(stderr, "error: K (%d) must be odd\n", K);
        return 1;
    }

    const std::size_t image_size =
        static_cast<std::size_t>(H) * static_cast<std::size_t>(W);

    const std::size_t kernel_size =
        static_cast<std::size_t>(K) * static_cast<std::size_t>(K);

    float* img = pa1::alloc_floats(image_size);
    float* ker = pa1::alloc_floats(kernel_size);
    float* out = pa1::alloc_floats(image_size);

    pa1::fill_random(img, image_size, seed);
    pa1::fill_random(ker, kernel_size, seed + 1u);

    // Same input representation used by the assignment harness.
    float* in = pa1::make_padded(img, H, W, K);

    auto run_kernel = [&]() {
        stage->fn(in, out, ker, H, W, K);
    };

    // Warm up cache/code paths/branch predictor.
    for (int i = 0; i < kWarmupRuns; ++i)
        run_kernel();

    // The checksum prevents the compiler from treating the output as dead.
    // It is deliberately tiny so it has negligible cost compared with the
    // convolution itself.
    volatile float checksum = 0.0f;

    // This is the ONLY convolution implementation called by this process.
    for (int i = 0; i < kMeasuredRuns; ++i)
        run_kernel();

    checksum = out[0] + out[image_size / 2] + out[image_size - 1];
    (void)checksum;

    pa1::free_floats(in);
    pa1::free_floats(img);
    pa1::free_floats(ker);
    pa1::free_floats(out);

    return 0;
}
