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

#include "legate_sparse/array/util/scale_rect.h"
#include "legate_sparse/array/util/scale_rect_template.inl"

namespace sparse {

using namespace legate;

template <>
struct ScaleRect1ImplBody<VariantKind::CPU> {
  TaskContext context;
  explicit ScaleRect1ImplBody(TaskContext context) : context(context) {}

  void operator()(const AccessorRW<Rect<1>, 1>& output, const int64_t scale, const Rect<1>& rect)
  {
    for (coord_t i = rect.lo[0]; i < rect.hi[0] + 1; i++) {
      output[i].lo = output[i].lo + scale;
      output[i].hi = output[i].hi + scale;
    }
  }
};

/*static*/ void ScaleRect1::cpu_variant(TaskContext context)
{
  scale_rect_1_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static const auto sparse_reg_task_ = []() -> char {
  ScaleRect1::register_variants();
  return 0;
}();

}  // namespace

}  // namespace sparse
