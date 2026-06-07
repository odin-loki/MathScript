#!/bin/bash
# MathScript Build Script

set -e

echo "=== MathScript Build Script ==="

# Configure
echo "Configuring with CMake..."
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# Build
echo "Building MathScript..."
cmake --build build --config Release --parallel

echo "=== Build Complete ==="
echo "Executables:"
ls -la build/bin/