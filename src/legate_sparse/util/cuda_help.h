/* Copyright 2022-2024 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#pragma once

#include <cstdlib>
#include "legate.h"

// For sparse matrix ops like spGEMM and spMv
#include <cusparse.h>

// For direct solvers
#include <cudss.h>

#include <nccl.h>

#define THREADS_PER_BLOCK 128

#define CHECK_CUSPARSE(expr)                    \
  do {                                          \
    cusparseStatus_t result = (expr);           \
    check_cusparse(result, __FILE__, __LINE__); \
  } while (false)

#define CHECK_CUDSS(expr)                    \
  do {                                       \
    cudssStatus_t result = (expr);           \
    check_cudss(result, __FILE__, __LINE__); \
  } while (false)

#define CHECK_NCCL(expr)                    \
  do {                                      \
    ncclResult_t result = (expr);           \
    check_nccl(result, __FILE__, __LINE__); \
  } while (false)

#define LEGATE_SPARSE_CHECK_CUDA(...)           \
  do {                                          \
    cudaError_t __result__ = (__VA_ARGS__);     \
    check_cuda(__result__, __FILE__, __LINE__); \
  } while (false)

#ifdef DEBUG_LEGATE_SPARSE
#define LEGATE_SPARSE_CHECK_CUDA_STREAM(stream)              \
  do {                                                       \
    LEGATE_SPARSE_CHECK_CUDA(cudaStreamSynchronize(stream)); \
    LEGATE_SPARSE_CHECK_CUDA(cudaPeekAtLastError());         \
  } while (false)
#else
#define LEGATE_SPARSE_CHECK_CUDA_STREAM(stream)      \
  do {                                               \
    LEGATE_SPARSE_CHECK_CUDA(cudaPeekAtLastError()); \
  } while (false)
#endif

namespace sparse {

__host__ inline void check_cuda(cudaError_t error, const char* file, int line)
{
  if (error != cudaSuccess) {
    fprintf(stderr,
            "Internal CUDA failure with error %s (%s) in file %s at line %d\n",
            cudaGetErrorString(error),
            cudaGetErrorName(error),
            file,
            line);
#ifdef DEBUG_LEGATE_SPARSE
    assert(false);
#else
    exit(error);
#endif
  }
}

__device__ inline size_t global_tid_1d()
{
  return static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

inline size_t get_num_blocks_1d(size_t threads)
{
  return (threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}

__host__ inline void check_cusparse(cusparseStatus_t status, const char* file, int line)
{
  if (status != CUSPARSE_STATUS_SUCCESS) {
    fprintf(stderr,
            "Internal CUSPARSE failure with error code %d (%s) in file %s at line %d\n",
            status,
            cusparseGetErrorString(status),
            file,
            line);
#ifdef DEBUG_LEGATE_SPARSE
    assert(false);
#else
    exit(status);
#endif
  }
}

__host__ inline void check_cudss(cudssStatus_t status, const char* file, int line)
{
  // TODO: Need to get the equivalent error message from cuDSS
  if (status != CUDSS_STATUS_SUCCESS) {
    fprintf(stderr,
            "Internal CUDSS failure with error code %d in file %s at line %d\n",
            status,
            // TODO
            file,
            line);
#ifdef DEBUG_LEGATE_SPARSE
    assert(false);
#else
    exit(status);
#endif
  }
}

__host__ inline void check_nccl(ncclResult_t error, const char* file, int line)
{
  if (error != ncclSuccess) {
    fprintf(stderr,
            "Internal NCCL failure with error %s in file %s at line %d\n",
            ncclGetErrorString(error),
            file,
            line);
#ifdef DEBUG_LEGATE_SPARSE
    assert(false);
#else
    exit(error);
#endif
  }
}

// Method to get the CUSPARSE handle associated with the current GPU.
cusparseHandle_t get_cusparse();

cudssHandle_t get_cudss();

}  // namespace sparse
