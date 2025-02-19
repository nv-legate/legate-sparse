/* Copyright 2023-2024 NVIDIA Corporation
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

#ifndef __LEGATE_SPARSE_CFFI_H__
#define __LEGATE_SPARSE_CFFI_H__

enum LegateSparseOpCode {
  _LEGATE_SPARSE_OP_CODE_BASE = 0,
  LEGATE_SPARSE_CSR_TO_DENSE,
  LEGATE_SPARSE_DENSE_TO_CSR_NNZ,
  LEGATE_SPARSE_DENSE_TO_CSR,
  LEGATE_SPARSE_BOUNDS_FROM_PARTITIONED_COORDINATES,
  LEGATE_SPARSE_SORTED_COORDS_TO_COUNTS,
  LEGATE_SPARSE_EXPAND_POS_TO_COORDINATES,

  // File IO.
  LEGATE_SPARSE_READ_MTX_TO_COO,

  // Operations on matrices that aren't quite tensor algebra related.
  LEGATE_SPARSE_CSR_DIAGONAL,

  // Linear algebra operations
  LEGATE_SPARSE_CSR_SPMV_ROW_SPLIT,
  LEGATE_SPARSE_SPGEMM_CSR_CSR_CSR_NNZ,
  LEGATE_SPARSE_SPGEMM_CSR_CSR_CSR,
  LEGATE_SPARSE_SPGEMM_CSR_CSR_CSR_GPU,

  // Dense linear algebra tasks needed for things
  // like iterative linear solvers.
  LEGATE_SPARSE_AXPBY,

  // Utility tasks.
  LEGATE_SPARSE_ZIP_TO_RECT_1,
  LEGATE_SPARSE_UNZIP_RECT_1,
  LEGATE_SPARSE_SCALE_RECT_1,
  LEGATE_SPARSE_FAST_IMAGE_RANGE,
  LEGATE_SPARSE_UPCAST_FUTURE_TO_REGION,

  // Utility tasks for loading cuda libraries.
  LEGATE_SPARSE_LOAD_CUDALIBS,
  LEGATE_SPARSE_UNLOAD_CUDALIBS,

  LEGATE_SPARSE_LAST_TASK,  // must be last
};

enum LegateSparseProjectionFunctors {
  _LEGATE_SPARSE_PROJ_FN_BASE = 0,
  LEGATE_SPARSE_PROJ_FN_1D_TO_2D,
  LEGATE_SPARSE_LAST_PROJ_FN,  // must be last
};

#endif  // __LEGATE_SPARSE_CFFI_H__
