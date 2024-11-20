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

#include "sparse/sparse.h"
#include "sparse/sparse_c.h"
#include "legate.h"

namespace sparse {

struct DenseToCSRNNZArgs {
  const legate::PhysicalStore& nnz;
  const legate::PhysicalStore& B_vals;
};

class DenseToCSRNNZ : public SparseTask<DenseToCSRNNZ> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{LEGATE_SPARSE_DENSE_TO_CSR_NNZ};
  static void cpu_variant(legate::TaskContext ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext ctx);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext context);
#endif
};

struct DenseToCSRArgs {
  const legate::PhysicalStore& A_pos;
  const legate::PhysicalStore& A_crd;
  const legate::PhysicalStore& A_vals;
  const legate::PhysicalStore& B_vals;
};

class DenseToCSR : public SparseTask<DenseToCSR> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{LEGATE_SPARSE_DENSE_TO_CSR};
  static void cpu_variant(legate::TaskContext ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext ctx);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext context);
#endif
};

}  // namespace sparse
