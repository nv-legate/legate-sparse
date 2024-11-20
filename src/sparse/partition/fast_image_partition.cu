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

#include "sparse/partition/fast_image_partition.h"
#include "sparse/partition/fast_image_partition_template.inl"
#include "sparse/util/cuda_help.h"

#include <thrust/extrema.h>
#include <cub/cub.cuh>

namespace sparse {

using namespace legate;

template <Type::Code INDEX_CODE>
struct FastImageRangeImplBody<VariantKind::GPU, INDEX_CODE> {
  using INDEX_TY = type_of<INDEX_CODE>;

  void operator()(const AccessorWO<Rect<1>, 1>& out_pos,
                  const AccessorRO<Rect<1>, 1>& in_pos,
                  const AccessorRO<INDEX_TY, 1>& in_crd,
                  const Rect<1>& rowbounds,
                  const Rect<1>& bounds)
  {
    auto stream      = get_cached_stream();
    auto thrust_exec = thrust::cuda::par.on(stream);

    thrust::pair<const INDEX_TY*, const INDEX_TY*> result =
      thrust::minmax_element(thrust_exec, in_crd.ptr(bounds.lo[0]), in_crd.ptr(bounds.hi[0]) + 1);

    // out[idx]       = {lo[idx], hi[idx] - 1};
    INDEX_TY lo_idx, hi_idx;
    cudaMemcpyAsync(&lo_idx, result.first, sizeof(INDEX_TY), cudaMemcpyDefault, stream);
    cudaMemcpyAsync(&hi_idx, result.second, sizeof(INDEX_TY), cudaMemcpyDefault, stream);
    thrust::fill(thrust_exec,
                 out_pos.ptr(rowbounds.lo[0]),
                 out_pos.ptr(rowbounds.hi[0]) + 1,
                 Rect<1>({lo_idx, hi_idx}));

    LEGATE_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void FastImageRange::gpu_variant(TaskContext context)
{
  fast_image_range_template<VariantKind::GPU>(context);
}

}  // namespace sparse
