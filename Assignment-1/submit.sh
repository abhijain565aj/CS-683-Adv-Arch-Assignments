#!/bin/bash
set -e

ROLL_NUMBERS="23b0903_23b0938_23b0942_23b1035"

TAR_NAME="${ROLL_NUMBERS}_pa1.tar.gz"
TMP_DIR="pa1-dhurandhar-microarchitecture"

echo "Preparing submission: $TAR_NAME"

# 1. Clean up any previous attempts
rm -rf "$TMP_DIR"
rm -f "$TAR_NAME"

# 2. Create the exact folder structure required
mkdir -p "$TMP_DIR/task1/src"
mkdir -p "$TMP_DIR/task2/src"
mkdir -p "$TMP_DIR/plots"

echo "Copying files..."

# Task 1
for file in conv_reorder.cpp conv_unroll.cpp conv_tile.cpp conv_simd.cpp conv_optimized.cpp; do
    if [ -f "task1/src/$file" ]; then
        cp "task1/src/$file" "$TMP_DIR/task1/src/"
    else
        echo "  [!] Error: task1/src/$file is missing"
    fi
done

# Task 1 - Report (handling your specific source path)
if [ -f "task1/report/task1_report.pdf" ]; then
    cp "task1/report/task1_report.pdf" "$TMP_DIR/task1/task1_report.pdf"
else
    echo "  [!] Error: task1_report.pdf is missing"
fi

# Task 2
for file in matmul_simd.cpp matmul_prefetch.cpp matmul_optimized.cpp; do
    if [ -f "task2/src/$file" ]; then
        cp "task2/src/$file" "$TMP_DIR/task2/src/"
    else
        echo "  [!] Error: task2/src/$file is missing"
    fi
done

# Task 2 - Report
if [ -f "task2/task2_report.pdf" ]; then
    cp "task2/task2_report.pdf" "$TMP_DIR/task2/"
else
    echo "  [!] Error: task2_report.pdf is missing"
fi

# Exact Plots
for plot in task1_speedup_vs_size.png task1_speedup_vs_kernel.png task2_speedup_vs_size.png task2_prefetch_degree.png task2_prefetch_level.png; do
    if [ -f "plots/$plot" ]; then
        cp "plots/$plot" "$TMP_DIR/plots/"
    else
        echo "  [!] Error: plots/$plot is missing"
    fi
done

# README
if [ -f "README.md" ]; then
    cp README.md "$TMP_DIR/"
else
    echo "  [!] Error: README.md is missing"
fi

# 4. Compress the folder into a tar.gz archive
echo "➔ Compressing into $TAR_NAME..."
tar -czf "$TAR_NAME" "$TMP_DIR"

# 5. Clean up temporary directory
# rm -rf "$TMP_DIR"

echo "Success! Generated $TAR_NAME"