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

#include "legate_sparse/array/csr/indexing.h"
#include "legate_sparse/util/dispatch.h"

namespace sparse {

using namespace legate;

template <VariantKind KIND, Type::Code INDEX_TY, Type::Code VAL_CODE>
struct CSRIndexingCSRImplBody;

template <VariantKind KIND>
struct CSRIndexingCSRImpl {
  TaskContext context;
  explicit CSRIndexingCSRImpl(TaskContext context) : context(context) {}

  template <Type::Code INDEX_CODE, Type::Code VAL_CODE>
  void operator()(const CSRIndexingCSRArgs& args)
  {
    using INDEX_TY = type_of<INDEX_CODE>;
    using VAL_TY   = type_of<VAL_CODE>;

    auto A_pos  = args.A_pos.read_accessor<Rect<1>, 1>();
    auto A_crd  = args.A_crd.read_accessor<INDEX_TY, 1>();
    auto A_vals = args.A_vals.read_write_accessor<VAL_TY, 1>();

    auto key_pos = args.key_pos.read_accessor<Rect<1>, 1>();
    auto key_crd = args.key_crd.read_accessor<INDEX_TY, 1>();

    auto value = args.value.read_accessor<VAL_TY, 1>();

    // TODO: Rect is based on A_pos.shape, is that correct?
    CSRIndexingCSRImplBody<KIND, INDEX_CODE, VAL_CODE>{context}(
      A_pos, A_crd, A_vals, key_pos, key_crd, value, args.A_pos.shape<1>());
  }
};

template <VariantKind KIND>
static void csr_indexing_csr_template(TaskContext context)
{
  CSRIndexingCSRArgs args{
    context.outputs()[0],
    context.inputs()[0],
    context.inputs()[1],
    context.inputs()[2],
    context.inputs()[3],
    context.inputs()[4],  // value
  };

  index_type_value_type_dispatch(
    args.A_crd.code(), args.A_vals.code(), CSRIndexingCSRImpl<KIND>{context}, args);
}

}  // namespace sparse
