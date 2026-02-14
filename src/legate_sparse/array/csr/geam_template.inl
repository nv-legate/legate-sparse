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

#include "legate_sparse/array/csr/geam.h"
#include "legate_sparse/util/dispatch.h"
#include "legate_sparse/util/typedefs.h"

namespace sparse {
using namespace legate;

// ============================================================================
// Symbolic phase templates
// ============================================================================

template <VariantKind KIND, Type::Code INDEX_TY>
struct GeamSymbolicImplBody;

template <VariantKind KIND>
struct GeamSymbolicImpl {
  TaskContext context;
  explicit GeamSymbolicImpl(TaskContext context) : context(context) {}

  template <Type::Code INDEX_CODE>
  void operator()(const GeamCSRCSRSymbolicArgs& args)
  {
    using INDEX_TY = type_of<INDEX_CODE>;

    auto A_pos = args.A_pos.read_accessor<Rect<1>, 1>();
    auto A_crd = args.A_crd.read_accessor<INDEX_TY, 1>();
    auto B_pos = args.B_pos.read_accessor<Rect<1>, 1>();
    auto B_crd = args.B_crd.read_accessor<INDEX_TY, 1>();

    auto nnz_per_row = args.nnz_per_row.read_write_accessor<nnz_ty, 1>();

    GeamSymbolicImplBody<KIND, INDEX_CODE>{context}(
      A_pos, A_crd, B_pos, B_crd, nnz_per_row, args.A_pos.shape<1>());
  }
};

template <VariantKind KIND>
static void geam_csr_csr_symbolic_template(TaskContext context)
{
  GeamCSRCSRSymbolicArgs args{
    context.inputs()[0],   // A_pos
    context.inputs()[1],   // A_crd
    context.inputs()[2],   // B_pos
    context.inputs()[3],   // B_crd
    context.outputs()[0],  // nnz_per_row
  };

  index_type_dispatch(args.A_crd.code(), GeamSymbolicImpl<KIND>{context}, args);
}

// ============================================================================
// Compute phase templates
// ============================================================================

template <VariantKind KIND, Type::Code INDEX_CODE, Type::Code VAL_CODE>
struct GeamComputeImplBody;

template <VariantKind KIND>
struct GeamComputeImpl {
  TaskContext context;
  explicit GeamComputeImpl(TaskContext context) : context(context) {}

  template <Type::Code INDEX_CODE, Type::Code VAL_CODE>
  void operator()(const GeamCSRCSRComputeArgs& args)
  {
    using INDEX_TY = type_of<INDEX_CODE>;
    using VAL_TY   = type_of<VAL_CODE>;

    auto A_pos  = args.A_pos.read_accessor<Rect<1>, 1>();
    auto A_crd  = args.A_crd.read_accessor<INDEX_TY, 1>();
    auto A_vals = args.A_vals.read_accessor<VAL_TY, 1>();
    auto B_pos  = args.B_pos.read_accessor<Rect<1>, 1>();
    auto B_crd  = args.B_crd.read_accessor<INDEX_TY, 1>();
    auto B_vals = args.B_vals.read_accessor<VAL_TY, 1>();

    // C_pos is read-only (computed in symbolic phase)
    auto C_pos  = args.C_pos.read_accessor<Rect<1>, 1>();
    auto C_crd  = args.C_crd.write_accessor<INDEX_TY, 1>();
    auto C_vals = args.C_vals.write_accessor<VAL_TY, 1>();

    // Read scalar values
    auto alpha = args.alpha.read_accessor<VAL_TY, 1>();
    auto beta  = args.beta.read_accessor<VAL_TY, 1>();

    GeamComputeImplBody<KIND, INDEX_CODE, VAL_CODE>{context}(A_pos,
                                                             A_crd,
                                                             A_vals,
                                                             B_pos,
                                                             B_crd,
                                                             B_vals,
                                                             C_pos,
                                                             C_crd,
                                                             C_vals,
                                                             alpha,
                                                             beta,
                                                             args.A_pos.shape<1>());
  }
};

template <VariantKind KIND>
static void geam_csr_csr_compute_template(TaskContext context)
{
  GeamCSRCSRComputeArgs args{
    context.inputs()[0],   // A_pos
    context.inputs()[1],   // A_crd
    context.inputs()[2],   // A_vals
    context.inputs()[3],   // B_pos
    context.inputs()[4],   // B_crd
    context.inputs()[5],   // B_vals
    context.inputs()[6],   // C_pos (read-only, computed in symbolic phase)
    context.outputs()[0],  // C_crd
    context.outputs()[1],  // C_vals
    context.inputs()[7],   // alpha
    context.inputs()[8],   // beta
  };

  index_type_value_type_dispatch(
    args.A_crd.code(), args.A_vals.code(), GeamComputeImpl<KIND>{context}, args);
}

}  // namespace sparse
