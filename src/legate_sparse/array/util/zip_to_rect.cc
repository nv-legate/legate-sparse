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

#include "legate_sparse/array/util/zip_to_rect.h"
#include "legate_sparse/array/util/zip_to_rect_template.inl"

namespace sparse {

using namespace legate;

template <typename VAL>
struct ZipToRect1ImplBody<VariantKind::CPU, VAL> {
  TaskContext context;
  explicit ZipToRect1ImplBody(TaskContext context) : context(context) {}

  void operator()(const AccessorWO<Rect<1>, 1>& output,
                  const AccessorRO<VAL, 1>& lo,
                  const AccessorRO<VAL, 1>& hi,
                  const Rect<1>& rect)
  {
    for (coord_t i = rect.lo[0]; i < rect.hi[0] + 1; i++) {
      output[i] = Rect<1>{Point<1>{lo[i]}, Point<1>{hi[i] - 1}};
    }
  }
};

/*static*/ void ZipToRect1::cpu_variant(TaskContext context)
{
  zip_to_rect_1_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static const auto sparse_reg_task_ = []() -> char {
  ZipToRect1::register_variants();
  return 0;
}();

}  // namespace

}  // namespace sparse
