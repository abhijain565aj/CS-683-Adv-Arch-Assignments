# CS683 PA-1: Hardware-Conscious Performance Engineering

This assignment consists of two parts. Both start from a correct-but-naive C++ kernel and ask you
to improve its performance by *thinking about the hardware*: the cache hierarchy, instruction-level
parallelism, and the SIMD units of a modern x86 core.

See the assignment [document](https://docs.google.com/document/d/1suURtM3WaensRvABVxlHZmhbbXe-g2v-sNi8gUfuojg/edit?usp=sharing)
for the full problem statement.

## Tasks

| Task | Workload | Techniques | Instructions |
|------|----------|------------|--------------|
| 1 | 2D convolution | Loop reordering, loop unrolling, cache tiling, SIMD (AVX2) | [task1/README.md](task1/README.md) |
| 2 | Matrix multiplication (SGEMM), injected into llama.cpp | SIMD (AVX2), cache blocking + software prefetching, combining everything | [task2/README.md](task2/README.md) |

Each task directory is self-contained: build instructions, the files you need to edit,
grading, and submission requirements all live in that task's own README. Start there.
