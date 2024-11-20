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

#include "legate/mapping/store.h"

namespace sparse {
struct GetCSRDiagonalArgs {
  const legate::PhysicalStore& diag;
  const legate::PhysicalStore& pos;
  const legate::PhysicalStore& crd;
  const legate::PhysicalStore& vals;
};

class GetCSRDiagonal : public SparseTask<GetCSRDiagonal> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{LEGATE_SPARSE_CSR_DIAGONAL};
  // TODO (rohany): We could rewrite this having each implementation just make
  //  a call to thrust::transform, but the implementations are simple enough
  //  anyway.
  static void cpu_variant(legate::TaskContext ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext ctx);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext context);
#endif
};

}  // namespace sparse
