/* Copyright 2021-2024 NVIDIA Corporation
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
#include "sparse/linalg/axpby.h"
#include "sparse/util/dispatch.h"

namespace sparse {

using namespace legate;

template <VariantKind KIND, Type::Code VAL_CODE, bool IS_ALPHA, bool NEGATE>
struct AXPBYImplBody;

template <VariantKind KIND>
struct AXPBYImpl {
  template <Type::Code VAL_CODE>
  void operator()(AXPBYArgs& args) const
  {
    using VAL_TY = type_of<VAL_CODE>;
    auto y       = args.y.read_write_accessor<VAL_TY, 1>();
    auto x       = args.x.read_accessor<VAL_TY, 1>();
    auto a       = args.a.read_accessor<VAL_TY, 1>();
    auto b       = args.b.read_accessor<VAL_TY, 1>();
    if (args.y.domain().empty()) {
      return;
    }
    if (args.isalpha) {
      if (args.negate) {
        AXPBYImplBody<KIND, VAL_CODE, true, true>()(y, x, a, b, args.y.shape<1>());
      } else {
        AXPBYImplBody<KIND, VAL_CODE, true, false>()(y, x, a, b, args.y.shape<1>());
      }
    } else {
      if (args.negate) {
        AXPBYImplBody<KIND, VAL_CODE, false, true>()(y, x, a, b, args.y.shape<1>());
      } else {
        AXPBYImplBody<KIND, VAL_CODE, false, false>()(y, x, a, b, args.y.shape<1>());
      }
    }
  }
};

template <VariantKind KIND>
static void axpby_template(TaskContext context)
{
  AXPBYArgs args{
    context.outputs()[0],
    context.inputs()[0],
    context.inputs()[1],
    context.inputs()[2],
    context.scalars()[0].value<bool>(),
    context.scalars()[1].value<bool>(),
  };
  value_type_dispatch(args.y.code(), AXPBYImpl<KIND>{}, args);
}

}  // namespace sparse
