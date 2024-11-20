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

#include "legate.h"
#include "legate_defines.h"

#include "sparse/sparse_c.h"
#include "sparse/mapper/mapper.h"

#include <algorithm>

using namespace legate;
using namespace legate::mapping;

namespace sparse {

TaskTarget LegateSparseMapper::task_target(const Task& task, const std::vector<TaskTarget>& options)
{
  return *options.begin();
}

std::vector<StoreMapping> LegateSparseMapper::store_mappings(
  const Task& task, const std::vector<StoreTarget>& options)
{
  const auto& inputs = task.inputs();
  std::vector<StoreMapping> mappings(inputs.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    mappings[i] = StoreMapping::default_mapping(inputs[i].data(), options.front());
  }
  return std::move(mappings);
}

Scalar LegateSparseMapper::tunable_value(legate::TunableID tunable_id)
{
  LEGATE_ABORT("Legate_Sparse does not use any tunables");
}

}  // namespace sparse
