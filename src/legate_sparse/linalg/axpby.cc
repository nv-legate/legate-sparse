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

#include "legate_sparse/linalg/axpby.h"
#include "legate_sparse/linalg/axpby_template.inl"

namespace sparse {

using namespace legate;

template <Type::Code VAL_CODE, bool IS_ALPHA, bool NEGATE>
struct AXPBYImplBody<VariantKind::CPU, VAL_CODE, IS_ALPHA, NEGATE> {
  TaskContext context;
  explicit AXPBYImplBody(TaskContext context) : context(context) {}

  using VAL_TY = type_of<VAL_CODE>;

  void operator()(const AccessorRW<VAL_TY, 1>& y,
                  const AccessorRO<VAL_TY, 1>& x,
                  const AccessorRO<VAL_TY, 1>& a,
                  const AccessorRO<VAL_TY, 1>& b,
                  const Rect<1>& rect)
  {
    auto val = a[0] / b[0];
    if (NEGATE) {
      val = static_cast<VAL_TY>(-1) * val;
    }
    for (coord_t i = rect.lo[0]; i < rect.hi[0] + 1; i++) {
      if (IS_ALPHA) {
        y[i] = val * x[i] + y[i];
      } else {
        y[i] = x[i] + val * y[i];
      }
    }
  }
};

/*static*/ void AXPBY::cpu_variant(TaskContext context)
{
  axpby_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static const auto sparse_reg_task_ = []() -> char {
  AXPBY::register_variants();
  return 0;
}();

}  // namespace

}  // namespace sparse
