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

#include "sparse/array/csr/spgemm_csr_csr_csr.h"
#include "sparse/array/csr/spgemm_csr_csr_csr_template.inl"

#include <omp.h>
#include <thrust/extrema.h>

namespace sparse {

using namespace legate;

template <Type::Code INDEX_CODE>
struct SpGEMMCSRxCSRxCSRNNZImplBody<VariantKind::OMP, INDEX_CODE> {
  using INDEX_TY = type_of<INDEX_CODE>;

  void operator()(const AccessorWO<nnz_ty, 1>& nnz,
                  const AccessorRO<Rect<1>, 1>& B_pos,
                  const AccessorRO<INDEX_TY, 1>& B_crd,
                  const AccessorRO<Rect<1>, 1>& C_pos,
                  const AccessorRO<INDEX_TY, 1>& C_crd,
                  const Rect<1>& rect,
                  const Rect<1>& C_crd_bounds)
  {
    auto num_threads = omp_get_max_threads();
    auto kind        = legate::find_memory_kind_for_executing_processor();

    // Calculate A2_dim by looking at the min and max coordinates in
    // the provided partition of C.
    auto C_crd_ptr = C_crd.ptr(C_crd_bounds.lo);
    auto result =
      thrust::minmax_element(thrust::omp::par, C_crd_ptr, C_crd_ptr + C_crd_bounds.volume());
    INDEX_TY min    = *result.first;
    INDEX_TY max    = *result.second;
    INDEX_TY A2_dim = max - min + 1;

    // Next, initialize the deferred buffers ourselves, instead of using
    // Realm fills (which tend to be slower).
    Buffer<INDEX_TY, 1> index_list_all(kind, Rect<1>{0, (A2_dim * num_threads) - 1});
    Buffer<bool, 1> already_set_all(kind, Rect<1>{0, (A2_dim * num_threads) - 1});
#pragma omp parallel for schedule(static)
    for (INDEX_TY i = 0; i < A2_dim * num_threads; i++) {
      index_list_all[i]  = 0;
      already_set_all[i] = false;
    }

#pragma omp parallel for schedule(monotonic : dynamic, 128)
    for (auto i = rect.lo[0]; i < rect.hi[0] + 1; i++) {
      auto thread_id = omp_get_thread_num();
      // Offset each accessed array by the min coordinate. Importantly,
      // index_list is not offset by min, because it isn't accessed
      // by C's coordinates.
      auto index_list        = index_list_all.ptr(thread_id * A2_dim);
      auto already_set       = already_set_all.ptr(thread_id * A2_dim) - min;
      size_t index_list_size = 0;
      for (size_t kB = B_pos[i].lo; kB < B_pos[i].hi + 1; kB++) {
        auto k = B_crd[kB];
        for (size_t jC = C_pos[k].lo; jC < C_pos[k].hi + 1; jC++) {
          auto j = C_crd[jC];
          if (!already_set[j]) {
            index_list[index_list_size] = j;
            already_set[j]              = true;
            index_list_size++;
          }
        }
      }
      size_t row_nnzs = 0;
      for (auto index_loc = 0; index_loc < index_list_size; index_loc++) {
        auto j         = index_list[index_loc];
        already_set[j] = false;
        row_nnzs++;
      }
      nnz[i] = row_nnzs;
    }
  }
};

template <Type::Code INDEX_CODE, Type::Code VAL_CODE>
struct SpGEMMCSRxCSRxCSRImplBody<VariantKind::OMP, INDEX_CODE, VAL_CODE> {
  using INDEX_TY = type_of<INDEX_CODE>;
  using VAL_TY   = type_of<VAL_CODE>;

  void operator()(const AccessorRW<Rect<1>, 1>& A_pos,
                  const AccessorWO<INDEX_TY, 1>& A_crd,
                  const AccessorWO<VAL_TY, 1>& A_vals,
                  const AccessorRO<Rect<1>, 1>& B_pos,
                  const AccessorRO<INDEX_TY, 1>& B_crd,
                  const AccessorRO<VAL_TY, 1>& B_vals,
                  const AccessorRO<Rect<1>, 1>& C_pos,
                  const AccessorRO<INDEX_TY, 1>& C_crd,
                  const AccessorRO<VAL_TY, 1>& C_vals,
                  const Rect<1>& rect,
                  const Rect<1>& C_crd_bounds)
  {
    auto num_threads = omp_get_max_threads();
    auto kind        = legate::find_memory_kind_for_executing_processor();

    // Calculate A2_dim by looking at the min and max coordinates in
    // the provided partition of C.
    auto C_crd_ptr = C_crd.ptr(C_crd_bounds.lo);
    auto result =
      thrust::minmax_element(thrust::omp::par, C_crd_ptr, C_crd_ptr + C_crd_bounds.volume());
    INDEX_TY min    = *result.first;
    INDEX_TY max    = *result.second;
    INDEX_TY A2_dim = max - min + 1;

    // Next, initialize the deferred buffers ourselves, instead of using
    // Realm fills (which tend to be slower).
    Buffer<INDEX_TY, 1> index_list_all(kind, Rect<1>{0, (A2_dim * num_threads) - 1});
    Buffer<bool, 1> already_set_all(kind, Rect<1>{0, (A2_dim * num_threads) - 1});
    Buffer<VAL_TY, 1> workspace_all(kind, Rect<1>{0, (A2_dim * num_threads) - 1});
#pragma omp parallel for schedule(static)
    for (INDEX_TY i = 0; i < A2_dim * num_threads; i++) {
      index_list_all[i]  = 0;
      already_set_all[i] = false;
      workspace_all[i]   = 0;
    }

    // For this computation, we assume that the rows are partitioned.
#pragma omp parallel for schedule(monotonic : dynamic, 128)
    for (auto i = rect.lo[0]; i < rect.hi[0] + 1; i++) {
      auto thread_id = omp_get_thread_num();
      // Back offset each of the pointers by the min element. Importantly,
      // index_list is not offset by min, because it is not accessed by j.
      auto index_list        = index_list_all.ptr(thread_id * A2_dim);
      auto already_set       = already_set_all.ptr(thread_id * A2_dim) - min;
      auto workspace         = workspace_all.ptr(thread_id * A2_dim) - min;
      size_t index_list_size = 0;
      for (size_t kB = B_pos[i].lo; kB < B_pos[i].hi + 1; kB++) {
        auto k = B_crd[kB];
        for (size_t jC = C_pos[k].lo; jC < C_pos[k].hi + 1; jC++) {
          auto j = C_crd[jC];
          if (!already_set[j]) {
            index_list[index_list_size] = j;
            already_set[j]              = true;
            index_list_size++;
          }
          workspace[j] += B_vals[kB] * C_vals[jC];
        }
      }
      size_t pA2 = A_pos[i].lo;
      for (auto index_loc = 0; index_loc < index_list_size; index_loc++) {
        auto j         = index_list[index_loc];
        already_set[j] = false;
        A_crd[pA2]     = j;
        A_vals[pA2]    = workspace[j];
        pA2++;
        // Zero out the workspace once we have read the value.
        workspace[j] = 0.0;
      }
    }
  }
};

/*static*/ void SpGEMMCSRxCSRxCSRNNZ::omp_variant(TaskContext context)
{
  spgemm_csr_csr_csr_nnz_template<VariantKind::OMP>(context);
}

/*static*/ void SpGEMMCSRxCSRxCSR::omp_variant(TaskContext context)
{
  spgemm_csr_csr_csr_template<VariantKind::OMP>(context);
}

}  // namespace sparse
