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

#include "sparse/array/csr/spmv.h"
#include "sparse/array/csr/spmv_template.inl"
#include "sparse/util/cusparse_utils.h"
#include "sparse/util/legate_utils.h"
#include "sparse/util/dispatch.h"

#ifndef CUSPARSE_MISALIGNMENT_FIX_VERSION
#define CUSPARSE_MISALIGNMENT_FIX_VERSION 12500
#endif

namespace sparse {

template <>
struct CSRSpMVRowSplitImpl<VariantKind::GPU> {
  template <Type::Code INDEX_CODE, Type::Code VAL_CODE>
  void operator()(CSRSpMVRowSplitArgs& args) const
  {
    using INDEX_TY = type_of<INDEX_CODE>;
    using VAL_TY   = type_of<VAL_CODE>;

    auto& y      = args.y;
    auto& A_pos  = args.A_pos;
    auto& A_crd  = args.A_crd;
    auto& A_vals = args.A_vals;
    auto& x      = args.x;

    // Break out early if the iteration space partition is empty.
    if (y.domain().empty() || A_crd.domain().empty()) {
      return;
    }

    // Get context sensitive objects.
    auto handle = get_cusparse();
    auto stream = get_cached_stream();
    CHECK_CUSPARSE(cusparseSetStream(handle, stream));

    // Older cusparse has bug when output vector is not aligned to 16 bytes
    // It may happen for us since we manipulate input to the SpMV and
    // cannot guarantee such alignment
    int cusparse_version = 0;
    CHECK_CUSPARSE(cusparseGetVersion(handle, &cusparse_version));
    // fixed in 12.03.01
    bool spmv_buffered_output = cusparse_version < CUSPARSE_MISALIGNMENT_FIX_VERSION;
    // for some reason legion doesn't like typed buffer here
    std::unique_ptr<Buffer<char, 1>> y_aligned{nullptr};  // aligned buffer

    // In order to play nicely with cuSPARSE and weak-scale (on distribution
    // friendly inputs), we have to do some trickery. The first happens when
    // we launch our tasks: we take the image of the selected coordinates
    // onto the x vector, resulting in a sparse partition of x. Next, we map
    // x densely, so that we get a dense vector fitted to the size of the
    // coordinates, where communication is done just for the selected pieces.
    // Now, we need to pass a dense vector of the correct size into cuSPARSE's
    // SpMV. We can abuse the fact that a proper SpMV implementation should only
    // read the components of x corresponding to encoded columns in the input matrix.
    // Note that we don't have to do any of this for the CPU/OMP implementations
    // since those codes use the accessor types directly. If we switched to calling
    // an external library we would need to do something similar to this.
    auto x_domain = x.domain();
    // We set the number of columns to the upper bound of the domain of x. This
    // shrinks the number of columns to the largest column index in the selected
    // partition of the input matrix.
    auto cols = x_domain.hi()[0] + 1;
    // Next, we grab a pointer to the start of the x instance. Since x is densely
    // encoded, we can shift this pointer based on the lower bound of the input
    // domain to get a "fake" pointer to the start of a densely encoded vector
    // of length cols. We can bank on cuSPARSE's implementation to not read any
    // of the memory locations before the lower bound of the x_vals_domain due
    // to the properties of the resulting image partition.
    auto x_raw_ptr = x.read_accessor<VAL_TY, 1>().ptr(x_domain.lo());
    auto x_ptr     = x_raw_ptr - size_t(x_domain.lo()[0]);
    cusparseDnVecDescr_t cusparse_x;
    CHECK_CUSPARSE(cusparseCreateDnVec(
      &cusparse_x, cols, const_cast<VAL_TY*>(x_ptr), cusparseDataType<VAL_TY>()));

    auto y_domain      = y.domain();
    auto y_domain_size = y_domain.hi()[0] - y_domain.lo()[0] + 1;
    auto y_raw_ptr     = getPtrFromStore<VAL_TY, 1>(y);
    auto output_ptr    = y_raw_ptr;

    uintptr_t addr = reinterpret_cast<uintptr_t>(output_ptr);
    if ((addr & 15) && spmv_buffered_output) {
      // Older cusparse SpMV expects _output_ vector to be aligned to 16 bytes.
      // So we allocate temporary storage of the domain size with default alignment 16
      // and copy data there
      size_t buffer_size = (y_domain_size - 1) * sizeof(VAL_TY);
      y_aligned.reset(new Buffer<char, 1>(create_1d_extents(0, buffer_size), Memory::GPU_FB_MEM));
      output_ptr = y_aligned->ptr(0);
    }

    cusparseDnVecDescr_t cusparse_y;
    CHECK_CUSPARSE(cusparseCreateDnVec(
      &cusparse_y, y_domain_size /* size */, output_ptr, cusparseDataType<VAL_TY>()));

    auto cusparse_A = makeCuSparseCSR<INDEX_TY, VAL_TY>(A_pos, A_crd, A_vals, cols);

    // Make the CUSPARSE calls.
    VAL_TY alpha   = 1.0;
    VAL_TY beta    = 0.0;
    size_t bufSize = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha,
                                           cusparse_A,
                                           cusparse_x,
                                           &beta,
                                           cusparse_y,
                                           cusparseDataType<VAL_TY>(),
#if (CUSPARSE_VER_MAJOR < 11 || (CUSPARSE_VER_MAJOR == 11 && CUSPARSE_VER_MINOR < 2))
                                           CUSPARSE_MV_ALG_DEFAULT,
#else
                                           CUSPARSE_SPMV_ALG_DEFAULT,
#endif
                                           &bufSize));
    // Allocate a buffer if we need to.
    void* workspacePtr = nullptr;
    if (bufSize > 0) {
      Buffer<char, 1> buf(create_1d_extents(0, bufSize - 1), Memory::GPU_FB_MEM);
      workspacePtr = buf.ptr(0);
    }

    // Finally do the SpMV.
    CHECK_CUSPARSE(cusparseSpMV(handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha,
                                cusparse_A,
                                cusparse_x,
                                &beta,
                                cusparse_y,
                                cusparseDataType<VAL_TY>(),
#if (CUSPARSE_VER_MAJOR < 11 || (CUSPARSE_VER_MAJOR == 11 && CUSPARSE_VER_MINOR < 2))
                                CUSPARSE_MV_ALG_DEFAULT,
#else
                                CUSPARSE_SPMV_ALG_DEFAULT,
#endif
                                workspacePtr));
    // if we used temporary buffer, copy result to output
    if (y_aligned) {
      LEGATE_CHECK_CUDA(cudaMemcpyAsync(
        y_raw_ptr, output_ptr, y_domain_size * sizeof(VAL_TY), cudaMemcpyDeviceToDevice, stream));
    }
    // Destroy the created objects.
    CHECK_CUSPARSE(cusparseDestroyDnVec(cusparse_y));
    CHECK_CUSPARSE(cusparseDestroyDnVec(cusparse_x));
    CHECK_CUSPARSE(cusparseDestroySpMat(cusparse_A));
    LEGATE_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void CSRSpMVRowSplit::gpu_variant(TaskContext context)
{
  csr_spmv_row_split_template<VariantKind::GPU>(context);
}

}  // namespace sparse
