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

#include "legate_sparse/linalg/spsolve.h"
#include "legate_sparse/util/cusparse_utils.h"
#include "legate_sparse/util/cudss_utils.h"
#include "legate_sparse/util/dispatch.h"
#include "legate_sparse/util/legate_utils.h"

namespace sparse {

struct SpSolveImpl {
  TaskContext context;
  explicit SpSolveImpl(TaskContext context) : context(context) {}

  template <Type::Code INDEX_CODE, Type::Code VAL_CODE>
  void operator()(SpSolveArgs& args, int num_gpus) const
  {
    using INDEX_TY = type_of<INDEX_CODE>;
    using VAL_TY   = type_of<VAL_CODE>;

    auto& A_pos      = args.A_pos;
    auto& A_crd      = args.A_crd;
    auto& A_vals     = args.A_vals;
    auto& b          = args.b;
    auto& x          = args.x;  // output
    auto comms       = args.comms;
    uint64_t nrows_g = args.nrows_g;
    uint64_t nnz_g   = args.nnz_g;
    uint64_t ncols_g = nrows_g;

    int hybrid_mode = 0;  // 0 = GPU-only execution in cuDSS

    // cuDSS handle and stream set
    auto handle = get_cudss();
    auto stream = context.get_task_stream();
    CHECK_CUDSS(cudssSetStream(handle, stream));

    // create configuration and data objects
    cudssConfig_t config;
    cudssData_t solverData;

    CHECK_CUDSS(cudssConfigCreate(&config));
    CHECK_CUDSS(cudssConfigSet(config, CUDSS_CONFIG_HYBRID_MODE, &hybrid_mode, sizeof(int)));
    CHECK_CUDSS(cudssDataCreate(handle, &solverData));

    //    A      x   =   b
    // (m, n) (n, 1) = (m, 1); m = nrows, n = ncols
    // _l: local  (e.g., shape of the partitioned array)
    // _g: global (e.g., global shape of the array)

    int64_t nrows_l = A_pos.domain().get_volume();
    int64_t ncols_l = x.domain().get_volume();
    int64_t nnz_l   = A_vals.domain().get_volume();

    int64_t nrhs = 1;        // Number of right-hand side
    int64_t ldb  = nrows_g;  // leading dimension of b
    int64_t ldx  = ncols_g;  // leading dimension of x

    auto A_indptr = CREATE_BUFFER(int64_t, nrows_l + 1, Memory::GPU_FB_MEM, "A_indptr");
    {
      auto blocks = get_num_blocks_1d(nrows_l);
      convertGlobalPosToLocalIndPtr<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        nrows_l, A_pos.read_accessor<Rect<1>, 1>().ptr(A_pos.domain().lo()), A_indptr.ptr(0));
    }

    CHECK_CUDSS(cudssSetStream(handle, stream));

    cudssMatrix_t mat_A, vec_b, vec_x;
    CHECK_CUDSS(cudssMatrixCreateCsr(&mat_A,                  // pointer to the matrix
                                     nrows_g,                 // number of rows
                                     ncols_g,                 // number of columns
                                     nnz_g,                   // number of non-zeros
                                     (void*)A_indptr.ptr(0),  // offsets,
                                     nullptr,                 // end index if start index was used
                                     getPtrFromStore<INDEX_TY, 1>(A_crd),  // column indices
                                     getPtrFromStore<VAL_TY, 1>(A_vals),   // values
                                     cudssIndexType<INDEX_TY>(),           // indexType
                                     cudssDataType<VAL_TY>(),              // valueType
                                     CUDSS_MTYPE_GENERAL,                  // matrix type
                                     CUDSS_MVIEW_FULL,                     // matrix view
                                     CUDSS_BASE_ZERO                       // indexBase
                                     ));

    // NOTE:
    // nrhs should be derived from b (b.shape[1]) and MUST be 1 right now.
    // When we support multi-dimensional right-hand sides, we need to
    // make sure that a column major order is chosen in the mapper

    auto x_ptr = getPtrFromStore<VAL_TY, 1>(x);

    // Create dense output vector, x, of shape (ncol_g, nrhs)
    CHECK_CUDSS(cudssMatrixCreateDn(&vec_x,
                                    ncols_g,                  // number of rows
                                    nrhs,                     // number of RHS, set to 1
                                    ldx,                      // Leading dimension of x
                                    (void*)x_ptr,             // Values of the dense matrix
                                    cudssDataType<VAL_TY>(),  // Data type of the dense vector
                                    CUDSS_LAYOUT_COL_MAJOR)   // Layout
    );

    auto b_ptr = getPtrFromStore<VAL_TY, 1>(b);

    // Create dense RHS vector, b, of shape (nrows_g, nrhs)
    CHECK_CUDSS(cudssMatrixCreateDn(&vec_b,
                                    nrows_g,                  // number of rows
                                    nrhs,                     // number of RHS, set to 1
                                    ldb,                      // Leading dimension of b
                                    (void*)b_ptr,             // Values of the dense matrix
                                    cudssDataType<VAL_TY>(),  // Data type of the dense vector
                                    CUDSS_LAYOUT_COL_MAJOR)   // Layout
    );

    // Matrix and Vectors are partitioned row-wise
    if (num_gpus > 1) {
      ncclComm_t* comm = comms[0].get<ncclComm_t*>();
      cudssMatrixSetDistributionRow1d(mat_A,
                                      static_cast<int64_t>(A_pos.domain().lo()[0]),
                                      static_cast<int64_t>(A_pos.domain().hi()[0]));
      cudssMatrixSetDistributionRow1d(
        vec_b, static_cast<int64_t>(b.domain().lo()[0]), static_cast<int64_t>(b.domain().hi()[0]));
      cudssMatrixSetDistributionRow1d(
        vec_x, static_cast<int64_t>(x.domain().lo()[0]), static_cast<int64_t>(x.domain().hi()[0]));

      // path to libcudss_commlayer_nccl.so is obtained from the env CUDSS_COMM_LIB
      CHECK_CUDSS(cudssSetCommLayer(handle, nullptr));
      CHECK_CUDSS(cudssDataSet(handle, solverData, CUDSS_DATA_COMM, comm, sizeof(ncclComm_t*)));
    }

    // Solve
    CHECK_CUDSS(
      cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, solverData, mat_A, vec_x, vec_b));

    CHECK_CUDSS(
      cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config, solverData, mat_A, vec_x, vec_b));

    CHECK_CUDSS(cudssExecute(handle, CUDSS_PHASE_SOLVE, config, solverData, mat_A, vec_x, vec_b));

    // Destroy matrix, vectors, and setup
    CHECK_CUDSS(cudssMatrixDestroy(mat_A));
    CHECK_CUDSS(cudssMatrixDestroy(vec_x));
    CHECK_CUDSS(cudssMatrixDestroy(vec_b));
    CHECK_CUDSS(cudssDataDestroy(handle, solverData));
    CHECK_CUDSS(cudssConfigDestroy(config));

    LEGATE_SPARSE_CHECK_CUDA(cudaStreamSynchronize(stream));
  }
};

/* static */ void SpSolve::gpu_variant(TaskContext context)
{
  auto inputs  = context.inputs();
  auto outputs = context.outputs();
  auto comms   = context.communicators();

  SpSolveArgs args{inputs[0],                               // A_pos
                   inputs[1],                               // A_crd
                   inputs[2],                               // A_vals
                   inputs[3],                               // b
                   outputs[0],                              // x
                   context.scalars()[0].value<uint64_t>(),  // nrows_g
                   context.scalars()[1].value<uint64_t>(),  // nnz_g
                   comms};
  int num_gpus = static_cast<size_t>(context.get_launch_domain().hi()[0]) + 1;
  index_type_floating_point_value_type_dispatch(
    args.A_crd.code(), args.A_vals.code(), SpSolveImpl{context}, args, num_gpus);
}

using namespace legate;

}  // namespace sparse
