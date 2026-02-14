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

// Useful for IDEs.
#include "legate_sparse/array/conv/pos_to_coordinates.h"
#include "legate_sparse/util/dispatch.h"

namespace sparse {
using namespace legate;

template <VariantKind KIND, Type::Code INDEX_CODE>
struct ExpandPosToCoordinatesImplBody;

template <VariantKind KIND>
struct ExpandPosToCoordinatesImpl {
  TaskContext context;
  explicit ExpandPosToCoordinatesImpl(TaskContext context) : context(context) {}

  template <Type::Code INDEX_CODE>
  void operator()(ExpandPosToCoordinatesArgs& args) const
  {
    using INDEX_TY = type_of<INDEX_CODE>;

    auto pos                = args.pos.read_accessor<Rect<1>, 1>();
    auto row_indices        = args.row_indices.write_accessor<INDEX_TY, 1>();
    auto pos_domain         = args.pos.domain();
    auto row_indices_domain = args.row_indices.domain();

    if (pos_domain.empty() || row_indices_domain.empty()) {
      return;
    }
    ExpandPosToCoordinatesImplBody<KIND, INDEX_CODE>{context}(
      pos, row_indices, args.pos.shape<1>());
  }
};

template <VariantKind KIND>
static void pos_to_coordinates_template(TaskContext context)
{
  ExpandPosToCoordinatesArgs args{
    context.outputs()[0],
    context.inputs()[0],
  };
  index_type_dispatch(args.row_indices.code(), ExpandPosToCoordinatesImpl<KIND>{context}, args);
}

}  // namespace sparse
