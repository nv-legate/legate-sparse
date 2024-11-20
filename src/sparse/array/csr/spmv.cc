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

#include "sparse/array/csr/spmv.h"
#include "sparse/array/csr/spmv_template.inl"

namespace sparse {

using namespace legate;

template <Type::Code INDEX_CODE, Type::Code VAL_CODE>
struct CSRSpMVRowSplitImplBody<VariantKind::CPU, INDEX_CODE, VAL_CODE> {
  using INDEX_TY = type_of<INDEX_CODE>;
  using VAL_TY   = type_of<VAL_CODE>;

  void operator()(const AccessorWO<VAL_TY, 1>& y,
                  const AccessorRO<Rect<1>, 1>& A_pos,
                  const AccessorRO<INDEX_TY, 1>& A_crd,
                  const AccessorRO<VAL_TY, 1>& A_vals,
                  const AccessorRO<VAL_TY, 1>& x,
                  const Rect<1>& rect)
  {
    for (coord_t i = rect.lo[0]; i < rect.hi[0] + 1; i++) {
      VAL_TY sum = 0.0;
      for (size_t j_pos = A_pos[i].lo; j_pos < A_pos[i].hi + 1; j_pos++) {
        auto j = A_crd[j_pos];
        sum += A_vals[j_pos] * x[j];
      }
      y[i] = sum;
    }
  }
};

/*static*/ void CSRSpMVRowSplit::cpu_variant(TaskContext context)
{
  csr_spmv_row_split_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  CSRSpMVRowSplit::register_variants();
}
}  // namespace

}  // namespace sparse
