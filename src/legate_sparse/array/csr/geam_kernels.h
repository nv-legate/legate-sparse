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

#include "legate_sparse/util/typedefs.h"
#include "legate.h"

namespace sparse {
using namespace legate;

// =============================================================================
// Symbolic Phase: Compute nnz per row for C = A + B
// =============================================================================

// Computes the number of non-zeros in a single row of C = A + B
template <typename INDEX_TY>
LEGATE_HOST_DEVICE inline nnz_ty geam_symbolic_row(size_t row,
                                                   const AccessorRO<Rect<1>, 1> A_pos,
                                                   const AccessorRO<INDEX_TY, 1> A_crd,
                                                   const AccessorRO<Rect<1>, 1> B_pos,
                                                   const AccessorRO<INDEX_TY, 1> B_crd)
{
  size_t A_pos_start = A_pos[row].lo;
  size_t A_pos_end   = A_pos[row].hi + 1;
  size_t B_pos_start = B_pos[row].lo;
  size_t B_pos_end   = B_pos[row].hi + 1;

  size_t a_pos = A_pos_start;
  size_t b_pos = B_pos_start;
  nnz_ty count = 0;

  // Merge sorted column indices and count unique entries
  while (a_pos < A_pos_end && b_pos < B_pos_end) {
    if (A_crd[a_pos] < B_crd[b_pos]) {
      a_pos++;
    } else if (A_crd[a_pos] > B_crd[b_pos]) {
      b_pos++;
    } else {
      a_pos++;
      b_pos++;
    }
    count++;
  }

  // Add remaining elements
  count += (A_pos_end - a_pos) + (B_pos_end - b_pos);
  return count;
}

// =============================================================================
// Compute Phase: Compute C = alpha * A + beta * B for a single row
// =============================================================================

// Computes a single row of C = alpha * A + beta * B
template <typename INDEX_TY, typename VAL_TY>
LEGATE_HOST_DEVICE inline void geam_compute_row(size_t row,
                                                const AccessorRO<Rect<1>, 1> A_pos,
                                                const AccessorRO<INDEX_TY, 1> A_crd,
                                                const AccessorRO<VAL_TY, 1> A_vals,
                                                const AccessorRO<Rect<1>, 1> B_pos,
                                                const AccessorRO<INDEX_TY, 1> B_crd,
                                                const AccessorRO<VAL_TY, 1> B_vals,
                                                const AccessorRO<Rect<1>, 1> C_pos,
                                                const AccessorWO<INDEX_TY, 1> C_crd,
                                                const AccessorWO<VAL_TY, 1> C_vals,
                                                VAL_TY alpha,
                                                VAL_TY beta)
{
  size_t A_pos_start = A_pos[row].lo;
  size_t A_pos_end   = A_pos[row].hi + 1;
  size_t B_pos_start = B_pos[row].lo;
  size_t B_pos_end   = B_pos[row].hi + 1;
  size_t C_pos_start = C_pos[row].lo;

  size_t a_pos = A_pos_start;
  size_t b_pos = B_pos_start;
  size_t c_pos = C_pos_start;

  // Merge sorted column indices and compute values
  while (a_pos < A_pos_end && b_pos < B_pos_end) {
    if (A_crd[a_pos] < B_crd[b_pos]) {
      C_crd[c_pos]  = A_crd[a_pos];
      C_vals[c_pos] = alpha * A_vals[a_pos];
      a_pos++;
    } else if (A_crd[a_pos] > B_crd[b_pos]) {
      C_crd[c_pos]  = B_crd[b_pos];
      C_vals[c_pos] = beta * B_vals[b_pos];
      b_pos++;
    } else {
      C_crd[c_pos]  = A_crd[a_pos];
      C_vals[c_pos] = alpha * A_vals[a_pos] + beta * B_vals[b_pos];
      a_pos++;
      b_pos++;
    }
    c_pos++;
  }

  // Add remaining elements from A
  while (a_pos < A_pos_end) {
    C_crd[c_pos]  = A_crd[a_pos];
    C_vals[c_pos] = alpha * A_vals[a_pos];
    a_pos++;
    c_pos++;
  }

  // Add remaining elements from B
  while (b_pos < B_pos_end) {
    C_crd[c_pos]  = B_crd[b_pos];
    C_vals[c_pos] = beta * B_vals[b_pos];
    b_pos++;
    c_pos++;
  }
}

}  // namespace sparse
