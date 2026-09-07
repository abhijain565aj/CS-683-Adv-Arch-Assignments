#!/usr/bin/env bash
# task2a_hw_prefetch.sh
#
# Temporarily forces the four commonly controlled Intel hardware prefetchers
# ON or OFF on ONE CPU using MSR 0x1A4, runs the command you provide, then
# restores the exact original MSR value even if the command fails.
#
# This does NOT edit any assignment file.
#
# Requirements:
#   sudo access
#   msr-tools (rdmsr, wrmsr)
#   Linux on an Intel CPU that implements MSR 0x1A4
#
# Examples:
#
#   ./task2a_hw_prefetch.sh --cpu 0 --state on -- \
#     python3 task2a_collect.py --cpu 0 --hw-label on \
#       --output task2a_results_hw_on.csv
#
#   ./task2a_hw_prefetch.sh --cpu 0 --state off -- \
#     python3 task2a_collect.py --cpu 0 --hw-label off \
#       --output task2a_results_hw_off.csv
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK2_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CPU=0
STATE=""

usage() {
    cat <<EOF
Usage:
  $0 --cpu <cpu-id> --state <on|off> -- <command> [args...]

Examples:
  $0 --cpu 0 --state on  -- python3 task2a_collect.py --cpu 0 --hw-label on
  $0 --cpu 0 --state off -- python3 task2a_collect.py --cpu 0 --hw-label off
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu)
            CPU="$2"
            shift 2
            ;;
        --state)
            STATE="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "$STATE" != "on" && "$STATE" != "off" ]]; then
    echo "--state must be 'on' or 'off'." >&2
    usage >&2
    exit 2
fi

if [[ $# -eq 0 ]]; then
    usage >&2
    exit 2
fi

if ! grep -q '^vendor_id.*GenuineIntel' /proc/cpuinfo; then
    echo "Refusing: this helper is only intended for Intel Linux systems." >&2
    exit 1
fi

for tool in rdmsr wrmsr taskset; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Missing '$tool'. Install msr-tools (for rdmsr/wrmsr) and util-linux (taskset)." >&2
        exit 1
    fi
done

echo "Loading msr kernel module..."
sudo modprobe msr

if ! ORIG_HEX="$(sudo rdmsr -p "$CPU" 0x1a4 2>/dev/null)"; then
    cat >&2 <<EOF
Could not read MSR 0x1A4 on CPU $CPU.
Your Intel CPU/kernel may not expose this prefetch-control MSR.
No hardware-prefetch setting was changed.
EOF
    exit 1--output
fi

ORIG=$((16#$ORIG_HEX))

# Common Intel convention for MSR 0x1A4:
# bits 0..3 are DISABLE bits for:
#   0 L2 hardware prefetcher
#   1 L2 adjacent cache-line prefetcher
#   2 DCU hardware prefetcher
#   3 DCU IP prefetcher
if [[ "$STATE" == "off" ]]; then
    NEW=$((ORIG | 0xF))
else
    NEW=$((ORIG & ~0xF))
fi
printf -v NEW_HEX '%x' "$NEW"

restore() {
    echo
    echo "Restoring CPU $CPU MSR 0x1A4 to 0x$ORIG_HEX ..."
    sudo wrmsr -p "$CPU" 0x1a4 "$ORIG_HEX" || true
}
trap restore EXIT INT TERM

echo "CPU $CPU original MSR 0x1A4: 0x$ORIG_HEX"
echo "Temporarily forcing common hardware prefetchers $STATE on CPU $CPU..."
sudo wrmsr -p "$CPU" 0x1a4 "$NEW_HEX"

VERIFY_HEX="$(sudo rdmsr -p "$CPU" 0x1a4)"
echo "CPU $CPU temporary MSR 0x1A4: 0x$VERIFY_HEX"
echo "Run the collector with --cpu $CPU so the benchmark stays on this CPU."
echo

"$@"
