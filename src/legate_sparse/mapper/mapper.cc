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

#include "legate.h"
#include "legate_defines.h"

#include "legate_sparse/sparse_c.h"
#include "legate_sparse/mapper/mapper.h"

#include <algorithm>
#include <typeinfo>

using namespace legate;
using namespace legate::mapping;

namespace sparse {

std::vector<StoreMapping> LegateSparseMapper::store_mappings(
  const Task& task, const std::vector<StoreTarget>& options)
{
  const auto& inputs = task.inputs();
  std::vector<StoreMapping> mappings;
  for (size_t i = 0; i < inputs.size(); i++) {
    mappings.push_back(StoreMapping::default_mapping(inputs[i].data(), options.front()));
  }
  return std::move(mappings);
}

std::optional<std::size_t> LegateSparseMapper::allocation_pool_size(const Task& task,
                                                                    StoreTarget memory_kind)
{
  const auto task_id = static_cast<LegateSparseOpCode>(task.task_id());

  auto get_size_with_alignment = [](std::size_t unaligned_size) -> std::size_t {
    return ((unaligned_size + DEFAULT_ALIGNMENT - 1) / DEFAULT_ALIGNMENT) * DEFAULT_ALIGNMENT;
  };

  switch (task_id) {
    case LEGATE_SPARSE_CSR_SPMV_ROW_SPLIT: {
      if (memory_kind == StoreTarget::FBMEM) {
        // GPU variant has two buffers with pre-determined size and
        // another one based on output from cuSparse
        // For the default spmv algorithm using csr format, cuSparse
        // could allocate ceil(nnz  / nthreads_per_block ) * sizeof(double)
        // bytes of temporary memory. Since this expression could change in the
        // future, we use this estimate and use a factor of safety to sheild us
        // from mapper errors while noting that nthreads_per_block is 128 on
        // newer GPUs and 32 on older ones.

        auto pos  = task.inputs()[0];
        auto crd  = task.inputs()[1];
        auto vals = task.inputs()[2];

        std::size_t nrows_plus_one = pos.domain().get_volume() + 1;
        std::size_t nnz            = vals.domain().get_volume();
        // make sure we don't fail; 1.15 is arbitrary
        std::size_t factor_of_safety = static_cast<std::size_t>(1.15);
        std::size_t cusparseSpMV_buffer_size =
          factor_of_safety * std::ceil(nnz / 32.0) * sizeof(double);
        std::size_t legate_buffer_size = nrows_plus_one * (vals.type().size() + crd.type().size());
        std::size_t total_size         = legate_buffer_size + cusparseSpMV_buffer_size;

        return get_size_with_alignment(total_size);
      } else {
        // No temp buffers for OMP and CPU variants
        return 0;
      }
    }

      // spGEMM OMP and CPU Variants
    case LEGATE_SPARSE_SPGEMM_CSR_CSR_CSR: {
      // Allocations done in the omp version:
      // (1) Three arrays of types bool, index_ty, val_ty and of
      //     size: (max_col - min_col of c) * nthreads
      // (2) Extra storage from thrust::minmax_element(). Use O(1) words, say,
      // 2?

      // For the first one, if we assume that datatype size is 17 bytes per word
      // (1 for bool, and 8 each for index and val types),
      // and approximate the total number of words,
      // we might be able to come up with an upper bound for this task.

      // See issue #178 before updating the pool size.

      return std::nullopt;
    }

      // spGEMM OMP and CPU Variants
    case LEGATE_SPARSE_SPGEMM_CSR_CSR_CSR_NNZ: {
      // Almost same as LEGATE_SPARSE_SPGEMM_CSR_CSR_CSR except that
      // there are only two deferred buffers instead of three
      // Wait until #178 is fixed

      return std::nullopt;
    }

    // spGEMM GPU Variant: Buffer size depends on cuSparse output
    // and cannot be predicted
    case LEGATE_SPARSE_SPGEMM_CSR_CSR_CSR_GPU: {
      return std::nullopt;
    }

    case LEGATE_SPARSE_READ_MTX_TO_COO: {
      // Three output buffers created but size depends on the file,
      // so we cannot estimate upper bound
      return std::nullopt;
    }

    case LEGATE_SPARSE_FAST_IMAGE_RANGE: {
      // Thrust allocator temp usage is hard to estimate
      // TODO: replace fill and minmax_element with hand-written kernels
      // and then update the estimate here
      return std::nullopt;
    }

    case LEGATE_SPARSE_SPSOLVE: {
      return std::nullopt;
    }

    default: {
      // Handle any unhandled enum values
      LEGATE_ABORT("Unsupported Legate Sparse task_id: " + std::to_string(task_id));
      return {};
    }
  }
}

Scalar LegateSparseMapper::tunable_value(legate::TunableID tunable_id)
{
  LEGATE_ABORT("Legate_Sparse does not use any tunables");
}

}  // namespace sparse
