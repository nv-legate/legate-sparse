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

#include "legate_sparse/array/csr/indexing.h"
#include "legate_sparse/array/csr/indexing_template.inl"

namespace sparse {

using namespace legate;

template <Type::Code INDEX_CODE, Type::Code VAL_CODE>
struct CSRIndexingCSRImplBody<VariantKind::CPU, INDEX_CODE, VAL_CODE> {
  TaskContext context;
  explicit CSRIndexingCSRImplBody(TaskContext context) : context(context) {}

  using INDEX_TY = type_of<INDEX_CODE>;
  using VAL_TY   = type_of<VAL_CODE>;

  void operator()(const AccessorRO<Rect<1>, 1>& A_pos,
                  const AccessorRO<INDEX_TY, 1>& A_crd,
                  const AccessorRW<VAL_TY, 1>& A_vals,
                  const AccessorRO<Rect<1>, 1>& mask_pos,
                  const AccessorRO<INDEX_TY, 1>& mask_crd,
                  const AccessorRO<VAL_TY, 1>& value,
                  const Rect<1>& rect)
  {
    for (size_t row = rect.lo[0]; row < rect.hi[0] + 1; row++) {
      size_t j_pos_start = A_pos[row].lo;
      size_t j_pos_end   = A_pos[row].hi + 1;

      size_t m_pos_start = mask_pos[row].lo;
      size_t m_pos_end   = mask_pos[row].hi + 1;

      size_t m_pos = m_pos_start;
      size_t j_pos = j_pos_start;

      // When the if condition is satisfied, the (row, col) of A and
      // mask match. Ideally, we would expect it to match for all
      // elements, even though mask stores only the True elements
      // making its sparsity pattern differ from A.
      // This would be the case if mask was derived from A.
      // However, if mask has entries that are not present in A,
      // then the else conditions will be hit.
      // Note that we don't update the vals array in those cases
      // since updating vals would require changing its size
      // apriori and hence the sparsity pattern of A, which is not
      // supported in this task.

      while (m_pos < m_pos_end && j_pos < j_pos_end) {
        if (mask_crd[m_pos] == A_crd[j_pos]) {
          A_vals[j_pos] = static_cast<VAL_TY>(value[0]);
          j_pos++;
          m_pos++;
        } else if (mask_crd[m_pos] > A_crd[j_pos]) {
          // this element in A is either not found in mask or is False
          // in mask and thus not stored. This means the pointer for
          // mask (m_pos) would have skipped ahead of the pointer
          // for A (j_pos), so A needs to catch-up; increment j_pos
          j_pos++;
        } else {  // mask_crd[m_pos] < A_crd[j_pos]
          // In this case, A is ahead and mask is behind in this row
          // which means mask has an entry (r,c) that was not in A.
          // Increment m_pos and let mask move ahead
          m_pos++;
        }
        // when either one of the pointers reach the end of the row,
        // we are done because we only update vals when (row, col)
        // of mask and A match exactly, and if one of the pointers
        // has reached the end of this row, the vals for this row
        // can never be updated, so exit the loop.
      }
    }
  }
};

/* static */ void CSRIndexingCSR::cpu_variant(legate::TaskContext context)
{
  csr_indexing_csr_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static const auto sparse_reg_task_ = []() -> char {
  CSRIndexingCSR::register_variants();
  return 0;
}();

}  // namespace

}  // namespace sparse
