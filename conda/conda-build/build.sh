#!/bin/bash

echo -e "\n\n--------------------- CONDA/CONDA-BUILD/BUILD.SH -----------------------\n"

set -xeo pipefail;

# If run through CI, BUILD_MARCH is set externally. If it is not set, try to set it.
ARCH=$(uname -m)
if [[ -z "${BUILD_MARCH}" ]]; then
    if [[ "${ARCH}" = "aarch64" ]]; then
        # Use the gcc march value used by aarch64 Ubuntu.
        BUILD_MARCH=armv8-a
    else
        # Use uname -m otherwise
        BUILD_MARCH=$(uname -m | tr '_' '-')
    fi
fi

# Rewrite conda's -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY to
#                 -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=BOTH
CMAKE_ARGS="$(echo "$CMAKE_ARGS" | sed -r "s@_INCLUDE=ONLY@_INCLUDE=BOTH@g")"

# Add our options to conda's CMAKE_ARGS
CMAKE_ARGS+="
--log-level=VERBOSE
-DBUILD_SHARED_LIBS=ON
-DBUILD_MARCH=${BUILD_MARCH}
-DCMAKE_BUILD_TYPE=Release
-DCMAKE_VERBOSE_MAKEFILE=ON
-DCMAKE_BUILD_PARALLEL_LEVEL=${JOBS:-$(nproc --ignore=1)}"
if [ -z "$CPU_ONLY" ]; then
  CMAKE_ARGS+="-DCMAKE_CUDA_ARCHITECTURES=all-major"
fi

export CMAKE_GENERATOR=Ninja
export CUDAHOSTCXX=${CXX}
export OPENSSL_DIR="$PREFIX"

echo "Environment"
env

echo "Build starting on $(date)"
CUDAFLAGS="-isystem ${PREFIX}/include -L${PREFIX}/lib"
export CUDAFLAGS

SKBUILD_BUILD_OPTIONS=-j$CPU_COUNT \
$PYTHON -m pip install             \
  --root /                         \
  --no-deps                        \
  --prefix "$PREFIX"               \
  --no-build-isolation             \
  --upgrade                        \
  --cache-dir "$PIP_CACHE_DIR"     \
  --disable-pip-version-check      \
  . -vv

echo "Build ending on $(date)"
