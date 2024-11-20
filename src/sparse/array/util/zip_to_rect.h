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

struct ZipToRect1Args {
  const legate::PhysicalStore& out;
  const legate::PhysicalStore& lo;
  const legate::PhysicalStore& hi;
};

class ZipToRect1 : public SparseTask<ZipToRect1> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{LEGATE_SPARSE_ZIP_TO_RECT_1};
  static void cpu_variant(legate::TaskContext ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext ctx);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext ctx);
#endif
};

}  // namespace sparse
