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

#include "legate_sparse/array/conv/pos_to_coordinates.h"
#include "legate_sparse/array/conv/pos_to_coordinates_template.inl"
#include "legate_sparse/util/cuda_help.h"
#include "legate_sparse/util/cusparse_utils.h"
#include <stdio.h>

namespace sparse {

using namespace legate;

template <typename INDEX_TY>
__global__ void fill_row_indices(size_t rows,
                                 size_t offset,
                                 AccessorRO<Rect<1>, 1> pos,
                                 AccessorWO<INDEX_TY, 1> row_indices)
{
  const auto idx = global_tid_1d();

  if (idx >= rows) {
    return;
  }

  size_t row = offset + idx;
  for (size_t j_pos = pos[row].lo; j_pos < pos[row].hi + 1; j_pos++) {
    row_indices[j_pos] = row;
  }
}

template <Type::Code INDEX_CODE>
struct ExpandPosToCoordinatesImplBody<VariantKind::GPU, INDEX_CODE> {
  using INDEX_TY = type_of<INDEX_CODE>;

  void operator()(const AccessorRO<Rect<1>, 1>& pos,
                  const AccessorWO<INDEX_TY, 1>& row_indices,
                  const Rect<1>& rect)
  {
    auto stream = get_cached_stream();
    auto blocks = get_num_blocks_1d(rect.volume());
    size_t rows = rect.volume();

    fill_row_indices<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(rows, rect.lo[0], pos, row_indices);
    LEGATE_SPARSE_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void ExpandPosToCoordinates::gpu_variant(TaskContext context)
{
  pos_to_coordinates_template<VariantKind::GPU>(context);
}

}  // namespace sparse
