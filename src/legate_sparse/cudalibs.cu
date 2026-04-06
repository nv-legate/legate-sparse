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

#include "legate_sparse/sparse.h"
#include "legate_sparse/sparse_c.h"
#include "legate_sparse/cudalibs.h"

#include <stdio.h>

namespace sparse {

CUDALibraries::CUDALibraries() : finalized_(false), cusparse_(nullptr), cudss_(nullptr) {}

CUDALibraries::~CUDALibraries() { finalize(); }

void CUDALibraries::finalize()
{
  if (finalized_) {
    return;
  }
  if (cusparse_ != nullptr) {
    finalize_cusparse();
  }
  if (cudss_ != nullptr) {
    finalize_cudss();
  }
  finalized_ = true;
}

void CUDALibraries::finalize_cusparse()
{
  CHECK_CUSPARSE(cusparseDestroy(cusparse_));
  cusparse_ = nullptr;
}

cusparseHandle_t CUDALibraries::get_cusparse()
{
  if (this->cusparse_ == nullptr) {
    CHECK_CUSPARSE(cusparseCreate(&this->cusparse_));
  }
  return this->cusparse_;
}

void CUDALibraries::finalize_cudss()
{
  CHECK_CUDSS(cudssDestroy(cudss_));
  cudss_ = nullptr;
}

cudssHandle_t CUDALibraries::get_cudss()
{
  if (this->cudss_ == nullptr) {
    CHECK_CUDSS(cudssCreate(&this->cudss_));
  }
  return this->cudss_;
}

static CUDALibraries& get_cuda_libraries(legate::Processor proc)
{
  if (proc.kind() != legate::Processor::TOC_PROC) {
    fprintf(stderr, "Illegal request for CUDA libraries for non-GPU processor");
    LEGATE_ABORT("Illegal request for CUDA libraries for non-GPU processor");
  }

  static CUDALibraries cuda_libraries[LEGION_MAX_NUM_PROCS];
  const auto proc_id = proc.id & (LEGION_MAX_NUM_PROCS - 1);
  return cuda_libraries[proc_id];
}

cusparseHandle_t get_cusparse()
{
  const auto proc = legate::Processor::get_executing_processor();
  auto& lib       = get_cuda_libraries(proc);
  return lib.get_cusparse();
}

cudssHandle_t get_cudss()
{
  const auto proc = legate::Processor::get_executing_processor();
  auto& lib       = get_cuda_libraries(proc);
  return lib.get_cudss();
}

class LoadCUDALibsTask : public SparseTask<LoadCUDALibsTask> {
 public:
  static inline const auto TASK_CONFIG =
    legate::TaskConfig{legate::LocalTaskID{LEGATE_SPARSE_LOAD_CUDALIBS}};

 public:
  static void gpu_variant(legate::TaskContext context)
  {
    const auto proc = legate::Processor::get_executing_processor();
    auto& lib       = get_cuda_libraries(proc);
    lib.get_cusparse();
    lib.get_cudss();
  }
};

class UnloadCUDALibsTask : public SparseTask<UnloadCUDALibsTask> {
 public:
  static inline const auto TASK_CONFIG =
    legate::TaskConfig{legate::LocalTaskID{LEGATE_SPARSE_UNLOAD_CUDALIBS}};

 public:
  static void gpu_variant(legate::TaskContext context)
  {
    const auto proc = legate::Processor::get_executing_processor();
    auto& lib       = get_cuda_libraries(proc);
    lib.finalize();
  }
};

static const auto sparse_reg_task_ = []() -> char {
  LoadCUDALibsTask::register_variants();
  UnloadCUDALibsTask::register_variants();
  return 0;
}();

}  // namespace sparse
