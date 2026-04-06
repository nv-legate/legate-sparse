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

#include "legate_sparse/sparse.h"
#include "legate_sparse/sparse_c.h"
#include "legate.h"

namespace sparse {

struct GeamCSRCSRSymbolicArgs {
  // Symbolic phase: compute the sparsity pattern of C = alpha * A + beta * B
  // This phase only needs the positions and coordinates, not the values or scalars
  const legate::PhysicalStore& A_pos;
  const legate::PhysicalStore& A_crd;
  const legate::PhysicalStore& B_pos;
  const legate::PhysicalStore& B_crd;
  const legate::PhysicalStore& nnz_per_row;  // output: number of non-zeros per row
};

struct GeamCSRCSRComputeArgs {
  // Compute phase: compute the output C where C = alpha * A + beta * B
  // Inputs
  const legate::PhysicalStore& A_pos;
  const legate::PhysicalStore& A_crd;
  const legate::PhysicalStore& A_vals;
  const legate::PhysicalStore& B_pos;
  const legate::PhysicalStore& B_crd;
  const legate::PhysicalStore& B_vals;

  // C_pos is an INPUT (computed in symbolic phase, read-only here)
  const legate::PhysicalStore& C_pos;

  // C_crd and C_vals are outputs
  const legate::PhysicalStore& C_crd;
  const legate::PhysicalStore& C_vals;

  // Scalar constants
  const legate::PhysicalStore& alpha;
  const legate::PhysicalStore& beta;
};

class GeamCSRCSRCompute : public SparseTask<GeamCSRCSRCompute> {
 public:
  static inline const auto TASK_CONFIG =
    legate::TaskConfig{legate::LocalTaskID{LEGATE_SPARSE_GEAM_CSR_CSR_COMPUTE}};

 public:
  static void cpu_variant(legate::TaskContext context);

#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext context);
#endif

#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext context);
#endif
};

class GeamCSRCSRSymbolic : public SparseTask<GeamCSRCSRSymbolic> {
 public:
  static inline const auto TASK_CONFIG =
    legate::TaskConfig{legate::LocalTaskID{LEGATE_SPARSE_GEAM_CSR_CSR_SYMBOLIC}};

 public:
  static void cpu_variant(legate::TaskContext context);

#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext context);
#endif

#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext context);
#endif
};

}  // namespace sparse
