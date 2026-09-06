#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

BIN="./bin/perf_conv"

# Number of independent perf measurements. perf reports the average.
PERF_REPEATS="${PERF_REPEATS:-20}"

# Core selection:
#   p    = P-core counters (cpu_core), pinned to one P-core; recommended
#   e    = E-core counters (cpu_atom), pinned to one E-core
#   both = aggregate cpu_core and cpu_atom counters; scheduler-dependent
CORE_MODE="${CORE_MODE:-p}"

# Optionally override the automatically selected logical CPU.
# Example: CORE_MODE=p CPU_PIN=2 ./perf_compare.sh naive unroll
CPU_PIN="${CPU_PIN:-}"

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

Core selection:
  CORE_MODE=p       P-core measurement (default and recommended)
  CORE_MODE=e       E-core measurement
  CORE_MODE=both    aggregate P-core and E-core PMU counts
  CPU_PIN=2         override the logical CPU used in p/e mode

Other settings:
  PERF_REPEATS=10   change the number of independent perf measurements
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

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

# Stage names must come first. Numeric arguments afterward are H W K [seed].
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
    echo "Error: no valid stages specified."
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
        echo "Error: H W K [seed] must be non-negative integers."
        exit 1
    fi
else
    echo "Error: after the stages, use either no dimensions or H W K [seed]."
    usage
    exit 1
fi

first_cpu_for_pmu() {
    local pmu="$1"
    local cpu_file="/sys/bus/event_source/devices/$pmu/cpus"

    if [[ ! -r "$cpu_file" ]]; then
        return 1
    fi

    local cpu_list
    cpu_list="$(<"$cpu_file")"
    echo "${cpu_list%%[-,]*}"
}

declare -a ACTIVE_PMUS
declare -a runner

case "$CORE_MODE" in
    p)
        ACTIVE_PMUS=(cpu_core)
        if [[ -z "$CPU_PIN" ]]; then
            CPU_PIN="$(first_cpu_for_pmu cpu_core)" || {
                echo "Error: cpu_core PMU was not found."
                exit 1
            }
        fi
        runner=(taskset -c "$CPU_PIN")
        mode_description="P-core only (cpu_core), logical CPU $CPU_PIN"
        ;;
    e)
        ACTIVE_PMUS=(cpu_atom)
        if [[ -z "$CPU_PIN" ]]; then
            CPU_PIN="$(first_cpu_for_pmu cpu_atom)" || {
                echo "Error: cpu_atom PMU was not found."
                exit 1
            }
        fi
        runner=(taskset -c "$CPU_PIN")
        mode_description="E-core only (cpu_atom), logical CPU $CPU_PIN"
        ;;
    both)
        ACTIVE_PMUS=(cpu_core cpu_atom)
        runner=()
        mode_description="P-core + E-core aggregate (not pinned)"
        ;;
    *)
        echo "Error: CORE_MODE must be p, e, or both."
        exit 1
        ;;
esac

declare -a EVENTS
for pmu in "${ACTIVE_PMUS[@]}"; do
    EVENTS+=(
        "$pmu/cache-references/"
        "$pmu/cache-misses/"
        "$pmu/mem_load_retired.l1_miss/"
        "$pmu/branches/"
        "$pmu/branch-misses/"
        "$pmu/instructions/"
        "$pmu/cycles/"
    )
done

EVENT_LIST="$(IFS=,; echo "${EVENTS[*]}")"

clean_number() {
    local value="$1"
    value="${value// /}"
    value="${value//,/}"

    if [[ -z "$value" || "$value" == *"<not"* ||
          "$value" == *"not supported"* ]]; then
        echo "N/A"
    else
        echo "$value"
    fi
}

event_value() {
    local perf_text="$1"
    local event_name="$2"

    awk -F, -v wanted="$event_name" '
        $3 == wanted {
            gsub(/[[:space:]]/, "", $1)
            print $1
            exit
        }
    ' <<< "$perf_text"
}

aggregate_counter() {
    local perf_text="$1"
    local event_suffix="$2"
    local total=0
    local found=0

    for pmu in "${ACTIVE_PMUS[@]}"; do
        local raw
        local value
        raw="$(event_value "$perf_text" "$pmu/$event_suffix/")"
        value="$(clean_number "$raw")"

        if [[ "$value" != "N/A" ]]; then
            total="$(awk -v a="$total" -v b="$value" 'BEGIN { printf "%.6f", a + b }')"
            found=1
        fi
    done

    if (( found )); then
        echo "$total"
    else
        echo "N/A"
    fi
}

ratio() {
    local numerator="$1"
    local denominator="$2"
    local scale="$3"
    local decimals="$4"

    if [[ "$numerator" == "N/A" || "$denominator" == "N/A" ]]; then
        echo "N/A"
        return
    fi

    awk -v n="$numerator" -v d="$denominator" +        -v s="$scale" -v places="$decimals" '
        BEGIN {
            if (d == 0) {
                print "N/A"
            } else {
                printf "%.*f", places, s * n / d
            }
        }
    '
}

format_count() {
    local value="$1"
    if [[ "$value" == "N/A" ]]; then
        echo "N/A"
    else
        awk -v x="$value" 'BEGIN { printf "%.0f", x }'
    fi
}

declare -a ROW_STAGE
declare -a ROW_INSTRUCTIONS
declare -a ROW_CYCLES
declare -a ROW_IPC
declare -a ROW_BRANCHES
declare -a ROW_BRANCH_MISSES
declare -a ROW_BRANCH_RATE
declare -a ROW_CACHE_REFS
declare -a ROW_CACHE_MISSES
declare -a ROW_L1_MISSES
declare -a ROW_L1_MPKI

echo
echo "=========================================================================="
echo "             CS683 Convolution Hardware-Counter Comparison"
echo "=========================================================================="
printf " Workload       : H=%s W=%s K=%s seed=%s\n" "$H" "$W" "$K" "$SEED"
printf " Core mode      : %s\n" "$mode_description"
printf " Kernel process : 5 warmups + 20 measured calls\n"
printf " Perf repeats   : %s complete process measurements\n" "$PERF_REPEATS"
echo
echo " L1-D MPKI = 1000 × retired L1 data-load misses / retired instructions"

if [[ "$CORE_MODE" == "both" ]]; then
    echo " Note: combined mode is scheduler-dependent; pinned P-core mode is preferred"
    echo "       when comparing implementations."
fi

for ((i=0; i<stage_end; ++i)); do
    stage="${args[$i]}"
    program_output="$(mktemp)"

    echo
    echo "--------------------------------------------------------------------------"
    printf " Measuring %-10s\n" "$stage"
    echo "--------------------------------------------------------------------------"

    set +e
    perf_output=$(
        "${runner[@]}" perf stat +            --no-big-num +            -x, +            -r "$PERF_REPEATS" +            -e "$EVENT_LIST" +            "$BIN" "$stage" "$H" "$W" "$K" "$SEED" +            2>&1 >"$program_output"
    )
    perf_status=$?
    set -e

    if [[ -s "$program_output" ]]; then
        sed 's/^/   /' "$program_output"
    fi
    rm -f "$program_output"

    if (( perf_status != 0 )); then
        echo
        echo "perf failed for stage '$stage':"
        sed 's/^/   /' <<< "$perf_output"
        exit "$perf_status"
    fi

    instructions="$(aggregate_counter "$perf_output" instructions)"
    cycles="$(aggregate_counter "$perf_output" cycles)"
    branches="$(aggregate_counter "$perf_output" branches)"
    branch_misses="$(aggregate_counter "$perf_output" branch-misses)"
    cache_refs="$(aggregate_counter "$perf_output" cache-references)"
    cache_misses="$(aggregate_counter "$perf_output" cache-misses)"
    l1_misses="$(aggregate_counter "$perf_output" mem_load_retired.l1_miss)"

    ipc="$(ratio "$instructions" "$cycles" 1 3)"
    branch_rate="$(ratio "$branch_misses" "$branches" 100 3)"
    l1_mpki="$(ratio "$l1_misses" "$instructions" 1000 3)"

    ROW_STAGE+=("$stage")
    ROW_INSTRUCTIONS+=("$(format_count "$instructions")")
    ROW_CYCLES+=("$(format_count "$cycles")")
    ROW_IPC+=("$ipc")
    ROW_BRANCHES+=("$(format_count "$branches")")
    ROW_BRANCH_MISSES+=("$(format_count "$branch_misses")")
    ROW_BRANCH_RATE+=("$branch_rate")
    ROW_CACHE_REFS+=("$(format_count "$cache_refs")")
    ROW_CACHE_MISSES+=("$(format_count "$cache_misses")")
    ROW_L1_MISSES+=("$(format_count "$l1_misses")")
    ROW_L1_MPKI+=("$l1_mpki")

    branch_rate_text="$branch_rate"
    if [[ "$branch_rate_text" != "N/A" ]]; then
        branch_rate_text+="%"
    fi

    echo
    printf "   %-22s %15s\n" "Instructions" "${ROW_INSTRUCTIONS[-1]}"
    printf "   %-22s %15s\n" "Cycles" "${ROW_CYCLES[-1]}"
    printf "   %-22s %15s\n" "IPC" "$ipc"
    printf "   %-22s %15s\n" "Branches" "${ROW_BRANCHES[-1]}"
    printf "   %-22s %15s\n" "Branch misses" "${ROW_BRANCH_MISSES[-1]}"
    printf "   %-22s %15s\n" "Branch miss rate" "$branch_rate_text"
    printf "   %-22s %15s\n" "Cache references" "${ROW_CACHE_REFS[-1]}"
    printf "   %-22s %15s\n" "Cache misses" "${ROW_CACHE_MISSES[-1]}"
    printf "   %-22s %15s\n" "L1-D load misses" "${ROW_L1_MISSES[-1]}"
    printf "   %-22s %15s\n" "L1-D MPKI" "$l1_mpki"
done

echo
echo "=========================================================================="
echo "                         Final comparison"
echo "=========================================================================="
printf "%-10s %13s %13s %7s %12s %11s %12s %12s %12s %9s\n" +    "stage" "instructions" "cycles" "IPC" "branches" "br-miss%" +    "cache-refs" "cache-miss" "L1D-misses" "L1D-MPKI"
printf "%-10s %13s %13s %7s %12s %11s %12s %12s %12s %9s\n" +    "----------" "-------------" "-------------" "-------" "------------" +    "-----------" "------------" "------------" "------------" "---------"

for ((i=0; i<${#ROW_STAGE[@]}; ++i)); do
    printf "%-10s %13s %13s %7s %12s %11s %12s %12s %12s %9s\n" +        "${ROW_STAGE[$i]}" +        "${ROW_INSTRUCTIONS[$i]}" +        "${ROW_CYCLES[$i]}" +        "${ROW_IPC[$i]}" +        "${ROW_BRANCHES[$i]}" +        "${ROW_BRANCH_RATE[$i]}" +        "${ROW_CACHE_REFS[$i]}" +        "${ROW_CACHE_MISSES[$i]}" +        "${ROW_L1_MISSES[$i]}" +        "${ROW_L1_MPKI[$i]}"
done

echo "=========================================================================="
