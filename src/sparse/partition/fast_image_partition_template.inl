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
#include "sparse/util/dispatch.h"
#include "sparse/util/typedefs.h"
#include "sparse/partition/fast_image_partition.h"

namespace sparse {

using namespace legate;

template <VariantKind KIND, Type::Code INDEX_CODE>
struct FastImageRangeImplBody;

template <VariantKind KIND>
struct FastImageRangeImpl {
  template <Type::Code INDEX_CODE>
  void operator()(FastImageRangeArgs& args) const
  {
    using INDEX_TY = type_of<INDEX_CODE>;

    auto output_pos = args.output_pos.write_accessor<Rect<1>, 1>();
    auto input_pos  = args.input_pos.read_accessor<Rect<1>, 1>();
    auto input_crd  = args.input_crd.read_accessor<INDEX_TY, 1>();
    assert(args.input_pos.domain().dense());
    assert(args.input_crd.domain().dense());
    if (args.input_crd.domain().empty()) {
      return;
    }
    FastImageRangeImplBody<KIND, INDEX_CODE>()(
      output_pos, input_pos, input_crd, args.input_pos.shape<1>(), args.input_crd.shape<1>());
  }
};

template <VariantKind KIND>
static void fast_image_range_template(TaskContext context)
{
  FastImageRangeArgs args{context.output(0), context.input(0), context.input(1)};
  index_type_dispatch(args.input_crd.code(), FastImageRangeImpl<KIND>{}, args);
}

}  // namespace sparse
