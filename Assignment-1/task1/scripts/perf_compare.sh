#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

BIN="$ROOT/bin/perf_conv"
PERF_REPEATS="${PERF_REPEATS:-20}"
CORE_MODE="${CORE_MODE:-p}"
CPU_PIN="${CPU_PIN:-}"

H=2048
W=2048
K=3
SEED=1234

usage() {
    echo "Usage:"
    echo "  $0 <stage1> [stage2 ...] [H W K] [seed]"
    echo
    echo "Stages:"
    echo "  naive reorder unroll tile simd optimized"
    echo
    echo "Examples:"
    echo "  $0 naive reorder unroll tile simd optimized"
    echo "  $0 naive unroll 1024 1024 3"
    echo "  $0 naive unroll 1024 1024 3 42"
    echo
    echo "Core selection:"
    echo "  CORE_MODE=p       P-core only (default and recommended)"
    echo "  CORE_MODE=e       E-core only"
    echo "  CORE_MODE=both    aggregate P-core and E-core counters"
    echo "  CPU_PIN=2         override the selected CPU in p/e mode"
    echo
    echo "Example:"
    echo "  CORE_MODE=e $0 naive simd"
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

if [[ ! -x "$BIN" ]]; then
    echo "Profiling executable not found. Building it now..."

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

args=("$@")
number_of_args=$#
stage_count=0

while (( stage_count < number_of_args )); do
    case "${args[$stage_count]}" in
        naive|reorder|unroll|tile|simd|optimized)
            ((stage_count += 1))
            ;;
        *)
            break
            ;;
    esac
done

if (( stage_count == 0 )); then
    echo "Error: no valid stage was provided."
    usage
    exit 1
fi

remaining=$((number_of_args - stage_count))

if (( remaining == 3 || remaining == 4 )); then
    H="${args[$stage_count]}"
    W="${args[$((stage_count + 1))]}"
    K="${args[$((stage_count + 2))]}"

    if (( remaining == 4 )); then
        SEED="${args[$((stage_count + 3))]}"
    fi
elif (( remaining != 0 )); then
    echo "Error: provide either no dimensions or H W K [seed]."
    exit 1
fi

if ! [[ "$H" =~ ^[0-9]+$ &&
        "$W" =~ ^[0-9]+$ &&
        "$K" =~ ^[0-9]+$ &&
        "$SEED" =~ ^[0-9]+$ ]]; then
    echo "Error: H, W, K and seed must be integers."
    exit 1
fi

first_cpu() {
    local pmu="$1"
    local cpu_file="/sys/bus/event_source/devices/$pmu/cpus"

    if [[ ! -r "$cpu_file" ]]; then
        return 1
    fi

    local cpu_range
    cpu_range="$(<"$cpu_file")"
    cpu_range="${cpu_range%%,*}"
    echo "${cpu_range%%-*}"
}

declare -a PMUS
declare -a runner

case "$CORE_MODE" in
    p)
        PMUS=(cpu_core)

        if [[ -z "$CPU_PIN" ]]; then
            CPU_PIN="$(first_cpu cpu_core)" || {
                echo "Error: cpu_core PMU is unavailable."
                exit 1
            }
        fi

        runner=(taskset --cpu-list "$CPU_PIN")
        mode_text="P-core (cpu_core), pinned to logical CPU $CPU_PIN"
        ;;
    e)
        PMUS=(cpu_atom)

        if [[ -z "$CPU_PIN" ]]; then
            CPU_PIN="$(first_cpu cpu_atom)" || {
                echo "Error: cpu_atom PMU is unavailable."
                exit 1
            }
        fi

        runner=(taskset --cpu-list "$CPU_PIN")
        mode_text="E-core (cpu_atom), pinned to logical CPU $CPU_PIN"
        ;;
    both)
        PMUS=(cpu_core cpu_atom)
        runner=()
        mode_text="P-core and E-core counters combined, without pinning"
        ;;
    *)
        echo "Error: CORE_MODE must be p, e or both."
        exit 1
        ;;
esac

declare -a events

for pmu in "${PMUS[@]}"; do
    events+=(
        "$pmu/instructions/"
        "$pmu/cycles/"
        "$pmu/branches/"
        "$pmu/branch-misses/"
        "$pmu/cache-references/"
        "$pmu/cache-misses/"
        "$pmu/mem_load_retired.l1_miss/"
    )
done

EVENT_LIST="$(IFS=,; echo "${events[*]}")"

trim_spaces() {
    local text="$1"
    text="${text//[[:space:]]/}"
    echo "$text"
}

extract_event() {
    local perf_file="$1"
    local wanted_event="$2"
    local value
    local unit
    local event
    local remaining_fields

    while IFS=';' read -r value unit event remaining_fields; do
        event="$(trim_spaces "$event")"
        value="$(trim_spaces "$value")"

        if [[ "$event" == "$wanted_event" ]]; then
            if [[ "$value" =~ ^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]]; then
                echo "$value"
            else
                echo "N/A"
            fi
            return
        fi
    done < "$perf_file"

    echo "N/A"
}

add_values() {
    python3 -c 'import sys; print(float(sys.argv[1]) + float(sys.argv[2]))' "$1" "$2"
}

aggregate_event() {
    local perf_file="$1"
    local event_name="$2"
    local total="0"
    local found=0
    local pmu
    local value

    for pmu in "${PMUS[@]}"; do
        value="$(extract_event "$perf_file" "$pmu/$event_name/")"

        if [[ "$value" != "N/A" ]]; then
            total="$(add_values "$total" "$value")"
            found=1
        fi
    done

    if (( found == 1 )); then
        echo "$total"
    else
        echo "N/A"
    fi
}

calculate_ratio() {
    local numerator="$1"
    local denominator="$2"
    local multiplier="$3"
    local decimal_places="$4"

    if [[ "$numerator" == "N/A" || "$denominator" == "N/A" ]]; then
        echo "N/A"
        return
    fi

    python3 -c '
import sys
numerator = float(sys.argv[1])
denominator = float(sys.argv[2])
multiplier = float(sys.argv[3])
places = int(sys.argv[4])

if denominator == 0:
    print("N/A")
else:
    print(f"{multiplier * numerator / denominator:.{places}f}")
' "$numerator" "$denominator" "$multiplier" "$decimal_places"
}

format_count() {
    local value="$1"

    if [[ "$value" == "N/A" ]]; then
        echo "N/A"
    else
        python3 -c 'import sys; print(round(float(sys.argv[1])))' "$value"
    fi
}

declare -a result_stage
declare -a result_instructions
declare -a result_cycles
declare -a result_ipc
declare -a result_branches
declare -a result_branch_misses
declare -a result_branch_rate
declare -a result_cache_references
declare -a result_cache_misses
declare -a result_l1_misses
declare -a result_l1_mpki

echo
echo "=============================================================================="
echo "              CS683 Convolution Hardware-Counter Comparison"
echo "=============================================================================="
printf "Workload       : H=%s, W=%s, K=%s, seed=%s\n" "$H" "$W" "$K" "$SEED"
printf "Core selection : %s\n" "$mode_text"
printf "Perf repeats   : %s\n" "$PERF_REPEATS"
echo "Kernel runs    : 5 warmups and 20 measured calls per perf process"
echo
echo "L1-D MPKI = 1000 × L1-D load misses / retired instructions"

if [[ "$CORE_MODE" == "both" ]]; then
    echo
    echo "Warning: combined mode is affected by scheduler migration."
    echo "Use the default P-core mode for reproducible implementation comparisons."
fi

for ((stage_index = 0; stage_index < stage_count; ++stage_index)); do
    stage="${args[$stage_index]}"

    program_output_file="$(mktemp)"
    perf_output_file="$(mktemp)"

    declare -a perf_command
    perf_command=(
        perf stat
        --no-big-num
        -x ";"
        -r "$PERF_REPEATS"
        -e "$EVENT_LIST"
        "$BIN"
        "$stage"
        "$H"
        "$W"
        "$K"
        "$SEED"
    )

    echo
    echo "------------------------------------------------------------------------------"
    printf "Measuring stage: %s\n" "$stage"
    echo "------------------------------------------------------------------------------"

    set +e
    "${runner[@]}" "${perf_command[@]}" >"$program_output_file" 2>"$perf_output_file"
    perf_status=$?
    set -e

    if [[ -s "$program_output_file" ]]; then
        sed 's/^/  /' "$program_output_file"
    fi

    if (( perf_status != 0 )); then
        echo
        echo "perf failed while measuring '$stage':"
        sed 's/^/  /' "$perf_output_file"
        rm -f "$program_output_file" "$perf_output_file"
        exit "$perf_status"
    fi

    instructions="$(aggregate_event "$perf_output_file" instructions)"
    cycles="$(aggregate_event "$perf_output_file" cycles)"
    branches="$(aggregate_event "$perf_output_file" branches)"
    branch_misses="$(aggregate_event "$perf_output_file" branch-misses)"
    cache_references="$(aggregate_event "$perf_output_file" cache-references)"
    cache_misses="$(aggregate_event "$perf_output_file" cache-misses)"
    l1_misses="$(aggregate_event "$perf_output_file" mem_load_retired.l1_miss)"

    ipc="$(calculate_ratio "$instructions" "$cycles" 1 3)"
    branch_rate="$(calculate_ratio "$branch_misses" "$branches" 100 3)"
    l1_mpki="$(calculate_ratio "$l1_misses" "$instructions" 1000 3)"

    instructions="$(format_count "$instructions")"
    cycles="$(format_count "$cycles")"
    branches="$(format_count "$branches")"
    branch_misses="$(format_count "$branch_misses")"
    cache_references="$(format_count "$cache_references")"
    cache_misses="$(format_count "$cache_misses")"
    l1_misses="$(format_count "$l1_misses")"

    result_stage+=("$stage")
    result_instructions+=("$instructions")
    result_cycles+=("$cycles")
    result_ipc+=("$ipc")
    result_branches+=("$branches")
    result_branch_misses+=("$branch_misses")
    result_branch_rate+=("$branch_rate")
    result_cache_references+=("$cache_references")
    result_cache_misses+=("$cache_misses")
    result_l1_misses+=("$l1_misses")
    result_l1_mpki+=("$l1_mpki")

    echo
    printf "  %-24s %16s\n" "Instructions" "$instructions"
    printf "  %-24s %16s\n" "Cycles" "$cycles"
    printf "  %-24s %16s\n" "IPC" "$ipc"
    printf "  %-24s %16s\n" "Branches" "$branches"
    printf "  %-24s %16s\n" "Branch misses" "$branch_misses"

    if [[ "$branch_rate" == "N/A" ]]; then
        printf "  %-24s %16s\n" "Branch miss rate" "N/A"
    else
        printf "  %-24s %15s%%\n" "Branch miss rate" "$branch_rate"
    fi

    printf "  %-24s %16s\n" "Cache references" "$cache_references"
    printf "  %-24s %16s\n" "Cache misses" "$cache_misses"
    printf "  %-24s %16s\n" "L1-D load misses" "$l1_misses"
    printf "  %-24s %16s\n" "L1-D MPKI" "$l1_mpki"

    rm -f "$program_output_file" "$perf_output_file"
done

echo
echo "=============================================================================="
echo "                              Final comparison"
echo "=============================================================================="

printf "%-10s %14s %14s %7s %12s %10s %12s %12s %12s %10s\n" \
    "Stage" "Instructions" "Cycles" "IPC" "Branches" "BrMiss%" \
    "CacheRefs" "CacheMiss" "L1DMisses" "L1D-MPKI"

printf "%-10s %14s %14s %7s %12s %10s %12s %12s %12s %10s\n" \
    "----------" "--------------" "--------------" "-------" "------------" \
    "----------" "------------" "------------" "------------" "----------"

for ((result_index = 0; result_index < ${#result_stage[@]}; ++result_index)); do
    printf "%-10s %14s %14s %7s %12s %10s %12s %12s %12s %10s\n" \
        "${result_stage[$result_index]}" \
        "${result_instructions[$result_index]}" \
        "${result_cycles[$result_index]}" \
        "${result_ipc[$result_index]}" \
        "${result_branches[$result_index]}" \
        "${result_branch_rate[$result_index]}" \
        "${result_cache_references[$result_index]}" \
        "${result_cache_misses[$result_index]}" \
        "${result_l1_misses[$result_index]}" \
        "${result_l1_mpki[$result_index]}"
done

echo "=============================================================================="
