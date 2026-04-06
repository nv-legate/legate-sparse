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

#include "legate_sparse/array/csr/geam.h"
#include "legate_sparse/array/csr/geam_template.inl"
#include "legate_sparse/array/csr/geam_kernels.h"
#include "legate_sparse/util/cuda_help.h"

namespace sparse {
using namespace legate;

// GPU kernel for symbolic phase: compute nnz_per_row
template <typename INDEX_TY>
__global__ void geam_symbolic_kernel(const size_t nrows,
                                     const AccessorRO<Rect<1>, 1> A_pos,
                                     const AccessorRO<INDEX_TY, 1> A_crd,
                                     const AccessorRO<Rect<1>, 1> B_pos,
                                     const AccessorRO<INDEX_TY, 1> B_crd,
                                     const AccessorRW<nnz_ty, 1> nnz_per_row)
{
  const size_t row = global_tid_1d();
  if (row >= nrows) {
    return;
  }

  nnz_per_row[row] = geam_symbolic_row(row, A_pos, A_crd, B_pos, B_crd);
}

// GPU kernel for compute phase: C = alpha * A + beta * B
template <typename INDEX_TY, typename VAL_TY>
__global__ void geam_compute_kernel(const size_t nrows,
                                    const AccessorRO<Rect<1>, 1> A_pos,
                                    const AccessorRO<INDEX_TY, 1> A_crd,
                                    const AccessorRO<VAL_TY, 1> A_vals,
                                    const AccessorRO<Rect<1>, 1> B_pos,
                                    const AccessorRO<INDEX_TY, 1> B_crd,
                                    const AccessorRO<VAL_TY, 1> B_vals,
                                    const AccessorRO<Rect<1>, 1> C_pos,
                                    const AccessorWO<INDEX_TY, 1> C_crd,
                                    const AccessorWO<VAL_TY, 1> C_vals,
                                    const AccessorRO<VAL_TY, 1> alpha_acc,
                                    const AccessorRO<VAL_TY, 1> beta_acc)
{
  const size_t row = global_tid_1d();
  if (row >= nrows) {
    return;
  }

  VAL_TY alpha = alpha_acc[0];
  VAL_TY beta  = beta_acc[0];

  geam_compute_row(
    row, A_pos, A_crd, A_vals, B_pos, B_crd, B_vals, C_pos, C_crd, C_vals, alpha, beta);
}

// GPU implementation of the symbolic phase
template <Type::Code INDEX_CODE>
struct GeamSymbolicImplBody<VariantKind::GPU, INDEX_CODE> {
  TaskContext context;
  explicit GeamSymbolicImplBody(TaskContext context) : context(context) {}

  using INDEX_TY = type_of<INDEX_CODE>;

  void operator()(const AccessorRO<Rect<1>, 1>& A_pos,
                  const AccessorRO<INDEX_TY, 1>& A_crd,
                  const AccessorRO<Rect<1>, 1>& B_pos,
                  const AccessorRO<INDEX_TY, 1>& B_crd,
                  const AccessorRW<nnz_ty, 1>& nnz_per_row,
                  const Rect<1>& rect)
  {
    auto stream     = context.get_task_stream();
    auto nrows      = rect.hi[0] - rect.lo[0] + 1;
    auto num_blocks = get_num_blocks_1d(nrows);

    if (nrows == 0) {
      return;
    }

    geam_symbolic_kernel<INDEX_TY><<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
      nrows, A_pos, A_crd, B_pos, B_crd, nnz_per_row);
    LEGATE_SPARSE_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void GeamCSRCSRSymbolic::gpu_variant(TaskContext context)
{
  geam_csr_csr_symbolic_template<VariantKind::GPU>(context);
}

// GPU implementation of the compute phase
template <Type::Code INDEX_CODE, Type::Code VAL_CODE>
struct GeamComputeImplBody<VariantKind::GPU, INDEX_CODE, VAL_CODE> {
  TaskContext context;
  explicit GeamComputeImplBody(TaskContext context) : context(context) {}

  using INDEX_TY = type_of<INDEX_CODE>;
  using VAL_TY   = type_of<VAL_CODE>;

  void operator()(const AccessorRO<Rect<1>, 1>& A_pos,
                  const AccessorRO<INDEX_TY, 1>& A_crd,
                  const AccessorRO<VAL_TY, 1>& A_vals,
                  const AccessorRO<Rect<1>, 1>& B_pos,
                  const AccessorRO<INDEX_TY, 1>& B_crd,
                  const AccessorRO<VAL_TY, 1>& B_vals,
                  const AccessorRO<Rect<1>, 1>& C_pos,
                  const AccessorWO<INDEX_TY, 1>& C_crd,
                  const AccessorWO<VAL_TY, 1>& C_vals,
                  const AccessorRO<VAL_TY, 1>& alpha,
                  const AccessorRO<VAL_TY, 1>& beta,
                  const Rect<1>& rect)
  {
    auto stream     = context.get_task_stream();
    auto nrows      = rect.hi[0] - rect.lo[0] + 1;
    auto num_blocks = get_num_blocks_1d(nrows);

    if (nrows == 0) {
      return;
    }

    geam_compute_kernel<INDEX_TY, VAL_TY><<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
      nrows, A_pos, A_crd, A_vals, B_pos, B_crd, B_vals, C_pos, C_crd, C_vals, alpha, beta);
    LEGATE_SPARSE_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void GeamCSRCSRCompute::gpu_variant(TaskContext context)
{
  geam_csr_csr_compute_template<VariantKind::GPU>(context);
}

}  // namespace sparse
