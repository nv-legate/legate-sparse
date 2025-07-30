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

struct CSRIndexingCSRArgs {
  const legate::PhysicalStore& A_vals;
  const legate::PhysicalStore& A_pos;
  const legate::PhysicalStore& A_crd;
  const legate::PhysicalStore& key_pos;
  const legate::PhysicalStore& key_crd;
  const legate::PhysicalStore& value;
};

class CSRIndexingCSR : public SparseTask<CSRIndexingCSR> {
 public:
  static inline const auto TASK_CONFIG =
    legate::TaskConfig{legate::LocalTaskID{LEGATE_SPARSE_CSR_INDEXING_CSR}};

  // TODO: The implementatio of the below three variants are
  // identical and hence need to be templated (DRY)

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
