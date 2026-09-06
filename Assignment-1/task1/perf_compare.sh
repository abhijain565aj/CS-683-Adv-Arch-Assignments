#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

BIN="./bin/perf_conv"

# Number of independent perf measurements.
# perf reports the average counters over these runs.
PERF_REPEATS=20

# Default workload.
H=2048
W=2048
K=3
SEED=1234

# Override this if the generic event is unavailable on your processor.
L1D_EVENT="${L1D_EVENT:-L1-dcache-load-misses}"

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
  Set L1D_EVENT if L1-dcache-load-misses is unsupported.
EOF
}

extract_counter() {
    local output="$1"
    local wanted="$2"

    awk -F',' -v wanted="$wanted" '
        {
            event = $3
            value = $1
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", event)
            gsub(/[[:space:]]/, "", value)

            if (event == wanted && value ~ /^[0-9]+([.][0-9]+)?$/) {
                print value
                found = 1
                exit
            }
        }
        END {
            if (!found) {
                print "N/A"
            }
        }
    ' <<< "$output"
}

calculate_mpki() {
    local misses="$1"
    local instructions="$2"

    if [[ "$misses" == "N/A" || "$instructions" == "N/A" ]]; then
        echo "N/A"
    else
        awk -v misses="$misses" -v instructions="$instructions" 'BEGIN {
            if (instructions > 0) {
                printf "%.3f", 1000.0 * misses / instructions
            } else {
                printf "N/A"
            }
        }'
    fi
}

format_count() {
    local value="$1"

    if [[ "$value" == "N/A" ]]; then
        printf "N/A"
    else
        awk -v value="$value" 'BEGIN { printf "%.0f", value }'
    fi
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

if (( H <= 0 || W <= 0 || K <= 0 )); then
    echo "Error: H, W, and K must be positive."
    exit 1
fi

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

declare -a result_stage
declare -a result_cache_references
declare -a result_cache_misses
declare -a result_l1d_misses
declare -a result_l1d_mpki
declare -a result_branches
declare -a result_branch_misses
declare -a result_instructions
declare -a result_cycles

echo
echo "=============================================================="
echo "       CS683 Convolution Hardware-Counter Comparison"
echo "=============================================================="
echo
echo "Workload       : H=$H W=$W K=$K seed=$SEED"
echo "Warmup         : 5 runs"
echo "Measured runs  : 20 per process"
echo "Perf repeats   : $PERF_REPEATS"
echo "L1-D event     : $L1D_EVENT"
echo
echo "L1-D MPKI = 1000 * L1-D load misses / retired instructions"
echo
echo "IMPORTANT: each requested stage is measured in a separate process."
echo "           No naive/reference kernel is executed automatically."
echo
echo "--------------------------------------------------------------"

result_index=0

for ((i=0; i<stage_end; ++i)); do
    stage="${args[$i]}"

    echo
    echo ">>> $stage"
    echo

    set +e
    perf_output=$(perf stat \
        -x, \
        --no-big-num \
        -r "$PERF_REPEATS" \
        -e "cache-references,cache-misses,$L1D_EVENT,branches,branch-misses,instructions,cycles" \
        "$BIN" "$stage" "$H" "$W" "$K" "$SEED" 2>&1)
    perf_status=$?
    set -e

    printf '%s\n' "$perf_output"

    if (( perf_status != 0 )); then
        echo
        echo "Error: perf failed while measuring '$stage'."
        exit "$perf_status"
    fi

    cache_references=$(extract_counter "$perf_output" "cache-references")
    cache_misses=$(extract_counter "$perf_output" "cache-misses")
    l1d_misses=$(extract_counter "$perf_output" "$L1D_EVENT")
    branches=$(extract_counter "$perf_output" "branches")
    branch_misses=$(extract_counter "$perf_output" "branch-misses")
    instructions=$(extract_counter "$perf_output" "instructions")
    cycles=$(extract_counter "$perf_output" "cycles")
    l1d_mpki=$(calculate_mpki "$l1d_misses" "$instructions")

    result_stage[$result_index]="$stage"
    result_cache_references[$result_index]="$cache_references"
    result_cache_misses[$result_index]="$cache_misses"
    result_l1d_misses[$result_index]="$l1d_misses"
    result_l1d_mpki[$result_index]="$l1d_mpki"
    result_branches[$result_index]="$branches"
    result_branch_misses[$result_index]="$branch_misses"
    result_instructions[$result_index]="$instructions"
    result_cycles[$result_index]="$cycles"
    result_index=$((result_index+1))

    echo
    echo "--------------------------------------------------------------"
done

echo
echo "================================================================================================================================================"
echo "Hardware-counter summary"
echo "================================================================================================================================================"
printf "%-11s %15s %15s %15s %12s %15s %15s %15s %15s\n" \
    "stage" "cache-refs" "cache-misses" "L1D-misses" "L1D-MPKI" \
    "branches" "branch-misses" "instructions" "cycles"
echo "------------------------------------------------------------------------------------------------------------------------------------------------"

for ((i=0; i<result_index; ++i)); do
    printf "%-11s %15s %15s %15s %12s %15s %15s %15s %15s\n" \
        "${result_stage[$i]}" \
        "$(format_count "${result_cache_references[$i]}")" \
        "$(format_count "${result_cache_misses[$i]}")" \
        "$(format_count "${result_l1d_misses[$i]}")" \
        "${result_l1d_mpki[$i]}" \
        "$(format_count "${result_branches[$i]}")" \
        "$(format_count "${result_branch_misses[$i]}")" \
        "$(format_count "${result_instructions[$i]}")" \
        "$(format_count "${result_cycles[$i]}")"
done

echo "================================================================================================================================================"
echo

if printf '%s\n' "${result_l1d_misses[@]}" | grep -q '^N/A$'; then
    echo "Note: L1-D MPKI is N/A where '$L1D_EVENT' was unsupported."
    echo "Check available events with: perf list | grep -i 'l1.*miss'"
fi

