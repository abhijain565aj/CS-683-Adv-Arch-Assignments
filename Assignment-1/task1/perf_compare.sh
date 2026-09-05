#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

BIN="./bin/perf_conv"

# Number of independent perf measurements.
# perf averages the counters over these runs.
PERF_REPEATS=20

# Default workload.
H=2048
W=2048
K=3
SEED=1234

usage() {
    cat <<EOF
Usage:
  $0 <stage1> [stage2 ...] [H W K] [seed]

Stages:
  naive reorder unroll tile simd optimized

Examples:
  $0 naive unroll
  $0 naive reorder unroll tile simd optimized
  $0 reorder optimized 1024 1024 3
  $0 naive unroll 2048 2048 3 42

Options:
  Change PERF_REPEATS at the top of this script if needed.
EOF
}

if [[ ! -x "$BIN" ]]; then
    echo "Profiling executable not found; building it now..."
    CXX="${CXX:-g++}"

    CXXFLAGS=(
        -std=c++17
        -O2
        -fno-tree-vectorize
        -mavx2
        -mfma
        -Iinclude
        -Wall
    )

    SOURCES=(
        perf_main.cpp
        src/conv_naive.cpp
        src/conv_reorder.cpp
        src/conv_unroll.cpp
        src/conv_tile.cpp
        src/conv_simd.cpp
        src/conv_optimized.cpp
    )

    mkdir -p bin
    "$CXX" "${CXXFLAGS[@]}" "${SOURCES[@]}" -o "$BIN"
    echo
fi

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

# Parse arguments from left to right. Stage names come first.
# After the first numeric argument, the remaining arguments are H W K [seed].
args=("$@")
n=$#

stage_end=0

while (( stage_end < n )); do
    case "${args[$stage_end]}" in
        naive|reorder|unroll|tile|simd|optimized)
            ((stage_end+=1))
            ;;
        *)
            break
            ;;
    esac
done

if (( stage_end < 1 )); then
    echo "Error: no stages specified."
    usage
    exit 1
fi

remaining=$((n-stage_end))

if (( remaining == 0 )); then
    :
elif (( remaining == 3 || remaining == 4 )); then
    H="${args[$stage_end]}"
    W="${args[$((stage_end+1))]}"
    K="${args[$((stage_end+2))]}"

    if (( remaining == 4 )); then
        SEED="${args[$((stage_end+3))]}"
    fi

    if ! [[ "$H" =~ ^[0-9]+$ && "$W" =~ ^[0-9]+$ &&
            "$K" =~ ^[0-9]+$ && "$SEED" =~ ^[0-9]+$ ]]; then
        echo "Error: H W K [seed] must be integers."
        exit 1
    fi
else
    echo "Error: use either no dimensions or H W K [seed]."
    usage
    exit 1
fi

# Validate stages before starting perf.
for ((i=0; i<stage_end; ++i)); do
    case "${args[$i]}" in
        naive|reorder|unroll|tile|simd|optimized)
            ;;
        *)
            echo "Error: unknown stage '${args[$i]}'"
            usage
            exit 1
            ;;
    esac
done

echo
echo "=============================================================="
echo "       CS683 Convolution Hardware-Counter Comparison"
echo "=============================================================="
echo
echo "Workload       : H=$H W=$W K=$K seed=$SEED"
echo "Warmup         : 5 runs"
echo "Measured runs  : 20 per process"
echo "Perf repeats   : $PERF_REPEATS"
echo
echo "Counters:"
echo "  cache-references"
echo "  cache-misses"
echo "  branches"
echo "  branch-misses"
echo "  instructions"
echo "  cycles"
echo
echo "IMPORTANT: each requested stage is measured in a separate process."
echo "           No naive/reference kernel is executed automatically."
echo
echo "--------------------------------------------------------------"

for ((i=0; i<stage_end; ++i)); do
    stage="${args[$i]}"

    echo
    echo ">>> $stage"
    echo

    # Each process calls ONLY the selected convolution function.
    #
    # -r repeats the complete measurement several times and reports
    #    the average.
    #
    # The kernel itself is executed 20 times inside each measurement.
    #
    # --no-big-num keeps counters readable.
    perf stat \
        --no-big-num \
        -r "$PERF_REPEATS" \
        -e cache-references,cache-misses,branches,branch-misses,instructions,cycles \
        "$BIN" "$stage" "$H" "$W" "$K" "$SEED" 2>&1

    echo
    echo "--------------------------------------------------------------"
done

echo
echo "Comparison complete."
