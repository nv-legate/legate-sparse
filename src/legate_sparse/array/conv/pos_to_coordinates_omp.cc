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

namespace sparse {

using namespace legate;

template <Type::Code INDEX_CODE>
struct ExpandPosToCoordinatesImplBody<VariantKind::OMP, INDEX_CODE> {
  TaskContext context;
  explicit ExpandPosToCoordinatesImplBody(TaskContext context) : context(context) {}

  using INDEX_TY = type_of<INDEX_CODE>;

  void operator()(const AccessorRO<Rect<1>, 1>& pos,
                  const AccessorWO<INDEX_TY, 1>& row_indices,
                  const Rect<1>& rect)
  {
#pragma omp parallel for schedule(monotonic : dynamic, 128)
    for (auto row = rect.lo[0]; row < rect.hi[0] + 1; row++) {
      for (size_t j_pos = pos[row].lo; j_pos < pos[row].hi + 1; j_pos++) {
        row_indices[j_pos] = row;
      }
    }
  }
};

/*static*/ void ExpandPosToCoordinates::omp_variant(TaskContext context)
{
  pos_to_coordinates_template<VariantKind::OMP>(context);
}

}  // namespace sparse
