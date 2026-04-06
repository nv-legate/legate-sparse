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

#include "legate_sparse/array/csr/spgemm_csr_csr_csr.h"
#include "legate_sparse/util/cusparse_utils.h"
#include "legate_sparse/util/dispatch.h"
#include "legate_sparse/util/legate_utils.h"
#include "legate_sparse/util/thrust_allocator.h"
#include "legate_sparse/util/legate_utils.h"

#include <thrust/scan.h>

#if (CUSPARSE_VER_MAJOR >= 12)
#define CUSPARSE_HAS_ALG3
#endif

namespace sparse {

using namespace legate;

template <typename DST, typename SRC>
__global__ void cast_and_offset(size_t elems, DST* dst, const SRC* src, int64_t offset)
{
  const auto idx = global_tid_1d();
  if (idx >= elems) {
    return;
  }
  dst[idx] = static_cast<DST>(src[idx] - offset);
}

int64_t local_offset_from_nnz(
  ncclComm_t comm, coord_t task_id, coord_t task_num, int64_t A_nnz, cudaStream_t stream)
{
  ThrustAllocator alloc(Memory::GPU_FB_MEM);
  auto policy         = thrust::cuda::par(alloc).on(stream);
  auto buf            = CREATE_BUFFER(int64_t, task_num, Memory::GPU_FB_MEM, "nnz_reduce_buf");
  auto nnz_reduce_buf = buf.ptr(0);

  // Pageable memory
  cudaMemcpyAsync(
    nnz_reduce_buf + task_id, &A_nnz, sizeof(int64_t), cudaMemcpyHostToDevice, stream);
  CHECK_NCCL(ncclAllGather(nnz_reduce_buf + task_id, nnz_reduce_buf, 1, ncclInt64, comm, stream));

  thrust::exclusive_scan(policy, nnz_reduce_buf, nnz_reduce_buf + task_num, nnz_reduce_buf);

  int64_t offset = 0;
  cudaMemcpyAsync(
    &offset, nnz_reduce_buf + task_id, sizeof(int64_t), cudaMemcpyDeviceToHost, stream);

  // needed to have offset available for the next call
  LEGATE_SPARSE_CHECK_CUDA(cudaStreamSynchronize(stream));

  return offset;
}

struct SpGEMMCSRxCSRxCSRGPUImpl {
  TaskContext context;
  explicit SpGEMMCSRxCSRxCSRGPUImpl(TaskContext context) : context(context) {}

  template <Type::Code INDEX_CODE, Type::Code VAL_CODE>
  void operator()(SpGEMMCSRxCSRxCSRGPUArgs& args, coord_t task_id, coord_t task_size) const
  {
    using INDEX_TY = type_of<INDEX_CODE>;
    using VAL_TY   = type_of<VAL_CODE>;

    auto task_num = task_size + 1;

    auto& A_pos  = args.A_pos;
    auto& A_crd  = args.A_crd;
    auto& A_vals = args.A_vals;
    auto& B_pos  = args.B_pos;
    auto& B_crd  = args.B_crd;
    auto& B_vals = args.B_vals;
    auto& C_pos  = args.C_pos;
    auto& C_crd  = args.C_crd;
    auto& C_vals = args.C_vals;
    auto& A2_dim = args.A2_dim;

    // Due to limitations around the cuSPARSE SpGEMM API, we can't do the standard
    // symbolic and actual execution phases of SpGEMM. Instead, we'll have each GPU
    // task output a local CSR matrix, and then we'll collapse the results of each
    // task into a global CSR matrix in Python land. The computation here and
    // interaction with cuSPARSE has gone through several iterations, and has
    // settled on an implementation that avoids all pointer offsetting to be
    // non-trusting of what cuSPARSE may do when reading pointers. In this task,
    // we have a row-partitioned B matrix, and use an image from the coordinates
    // in each partition of B to construct a row partition of the C matrix. Instead
    // of offsetting any pointers, we'll attempt to construct two new local matrices
    // that we can pass to cuSPARSE that are themselves valid. In particular, we use
    // the fact that we took an image from B to construct a matrix B', where each
    // coordinate in B' has been offset from the minimum coordinate in each partition
    // of B. The range of min and max coordinates in B is exactly equal to the number
    // of rows of C. We use this to construct a related matrix of C named C' that
    // doesn't offset the arrays at all, but uses the results of the images directly,
    // as the referencing coordinates from B' have been offset already.

    // Get context sensitive objects.
    auto handle = get_cusparse();
    auto stream = context.get_task_stream();
    CHECK_CUSPARSE(cusparseSetStream(handle, stream));

    auto B_rows      = B_pos.domain().get_volume();
    auto B_min_coord = C_pos.domain().lo()[0];
    auto B_max_coord = C_pos.domain().hi()[0];
    auto C_rows      = B_max_coord - B_min_coord + 1;

    // If there are no rows to process, then return empty output instances.
    if (B_rows == 0 || C_rows == 0 || B_crd.domain().empty() || C_crd.domain().empty()) {
      auto crd_buf = A_crd.create_output_buffer<INDEX_TY, 1>(0, true /* return_data */);
      auto val_buf = A_vals.create_output_buffer<VAL_TY, 1>(0, true /* return_data */);
      return;
    }

    // Convert the pos arrays into local indptr arrays.
    auto B_indptr = CREATE_BUFFER(int32_t, B_rows + 1, Memory::GPU_FB_MEM, "B_indptr");
    auto C_indptr = CREATE_BUFFER(int32_t, C_rows + 1, Memory::GPU_FB_MEM, "C_indptr");

    std::vector<int> tmem(1000, 0);
    {
      auto blocks = get_num_blocks_1d(B_rows);
      convertGlobalPosToLocalIndPtr<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        B_rows, B_pos.read_accessor<Rect<1>, 1>().ptr(B_pos.domain().lo()), B_indptr.ptr(0));
    }
    {
      auto blocks = get_num_blocks_1d(C_rows);
      convertGlobalPosToLocalIndPtr<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        C_rows, C_pos.read_accessor<Rect<1>, 1>().ptr(C_pos.domain().lo()), C_indptr.ptr(0));
    }

    auto B_crd_int =
      CREATE_BUFFER(int32_t, B_crd.domain().get_volume(), Memory::GPU_FB_MEM, "B_crd_int");

    // Importantly, don't use the volume for C, as the image optimization
    // is being applied. Compute an upper bound on the volume directly.
    auto C_nnz     = C_crd.domain().hi()[0] - C_crd.domain().lo()[0] + 1;
    auto C_crd_int = CREATE_BUFFER(int32_t, C_nnz, Memory::GPU_FB_MEM, "C_crd_int");
    {
      auto dom    = B_crd.domain();
      auto elems  = dom.get_volume();
      auto blocks = get_num_blocks_1d(elems);
      cast_and_offset<int32_t, INDEX_TY><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        elems, B_crd_int.ptr(0), B_crd.read_accessor<INDEX_TY, 1>().ptr(dom.lo()), B_min_coord);
    }
    {
      auto blocks = get_num_blocks_1d(C_nnz);
      cast<int32_t, INDEX_TY><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        C_nnz, C_crd_int.ptr(0), C_crd.read_accessor<INDEX_TY, 1>().ptr(C_crd.domain().lo()));
    }

    // Initialize the cuSPARSE matrices.
    cusparseSpMatDescr_t cusparse_A, cusparse_B, cusparse_C;
    CHECK_CUSPARSE(cusparseCreateCsr(&cusparse_B,
                                     B_rows,
                                     C_rows /* cols */,
                                     B_crd.domain().get_volume() /* nnz */,
                                     B_indptr.ptr(0),
                                     B_crd_int.ptr(0),
                                     getPtrFromStore<VAL_TY, 1>(B_vals),
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     cusparseDataType<VAL_TY>()));
    CHECK_CUSPARSE(cusparseCreateCsr(&cusparse_C,
                                     C_rows,
                                     A2_dim /* cols */,
                                     C_nnz,
                                     C_indptr.ptr(0),
                                     C_crd_int.ptr(0),
                                     (VAL_TY*)getPtrFromStore<VAL_TY, 1>(C_vals),
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     cusparseDataType<VAL_TY>()));
    CHECK_CUSPARSE(cusparseCreateCsr(&cusparse_A,
                                     B_rows /* rows */,
                                     A2_dim /* cols */,
                                     0 /* nnz */,
                                     nullptr,
                                     nullptr,
                                     nullptr,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     cusparseDataType<VAL_TY>()));

    // Allocate the SpGEMM descriptor.
    cusparseSpGEMMDescr_t descr;
    CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&descr));

    VAL_TY alpha       = static_cast<VAL_TY>(1);
    VAL_TY beta        = static_cast<VAL_TY>(0);
    size_t bufferSize1 = 0, bufferSize2 = 0, bufferSize3 = 0;
    float alg3_spgemm_fraction = 0.1f;
    int64_t A_rows = 0, A_cols = 0, A_nnz = 0;

    // ALG3 is slow but has less memory footprint, which can be
    // controlled using alg3_spgemm_fraction.
    // ALG1 is fast but is memory hungry.
    // Defaults:
    //    cusparse version < 12 : ALG1
    //    cusparse version > 12 and args.fast_switch : ALG1
    //    else: ALG3
    // fast_switch uses LEGATE_SPARSE_FAST_SPGEMM env and is FALSE by default
    auto cusparse_alg =
#ifndef CUSPARSE_HAS_ALG3
      CUSPARSE_SPGEMM_ALG1;
#else
      CUSPARSE_SPGEMM_ALG3;
    if (args.fast_switch) {
      cusparse_alg = CUSPARSE_SPGEMM_ALG1;
    }
#endif

    if (cusparse_alg == CUSPARSE_SPGEMM_ALG1) {
      CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle,
                                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   &alpha,
                                                   cusparse_B,
                                                   cusparse_C,
                                                   &beta,
                                                   cusparse_A,
                                                   cusparseDataType<VAL_TY>(),
                                                   cusparse_alg,
                                                   descr,
                                                   &bufferSize1,
                                                   nullptr));
      void* buffer1 = nullptr;
      if (bufferSize1 > 0) {
        auto buf = CREATE_BUFFER(char, bufferSize1, Memory::GPU_FB_MEM, "buffer1");
        buffer1  = buf.ptr(0);
      }
      CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle,
                                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   &alpha,
                                                   cusparse_B,
                                                   cusparse_C,
                                                   &beta,
                                                   cusparse_A,
                                                   cusparseDataType<VAL_TY>(),
                                                   cusparse_alg,
                                                   descr,
                                                   &bufferSize1,
                                                   buffer1));
      CHECK_CUSPARSE(cusparseSpGEMM_compute(handle,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha,
                                            cusparse_B,
                                            cusparse_C,
                                            &beta,
                                            cusparse_A,
                                            cusparseDataType<VAL_TY>(),
                                            cusparse_alg,
                                            descr,
                                            &bufferSize2,
                                            nullptr));
      void* buffer2 = nullptr;
      if (bufferSize2 > 0) {
        auto buf = CREATE_BUFFER(char, bufferSize2, Memory::GPU_FB_MEM, "buffer2");
        buffer2  = buf.ptr(0);
      }
      CHECK_CUSPARSE(cusparseSpGEMM_compute(handle,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha,
                                            cusparse_B,
                                            cusparse_C,
                                            &beta,
                                            cusparse_A,
                                            cusparseDataType<VAL_TY>(),
                                            cusparse_alg,
                                            descr,
                                            &bufferSize2,
                                            buffer2));
      // Allocate buffers for the 32-bit version of the A matrix.
      int64_t A_rows, A_cols, A_nnz;
      CHECK_CUSPARSE(cusparseSpMatGetSize(cusparse_A, &A_rows, &A_cols, &A_nnz));
      auto A_indptr = CREATE_BUFFER(int32_t, A_rows + 1, Memory::GPU_FB_MEM, "A_indptr");
      // Handle the creation of the A_crd buffer depending on whether the result
      // type is the type of data we are supposed to create.
      legate::Buffer<int32_t, 1> A_crd_int;
      if constexpr (INDEX_CODE == Type::Code::INT32) {
        A_crd_int = A_crd.create_output_buffer<INDEX_TY, 1>(A_nnz, true /* return_buffer */);
        LOG_BUFFER(INDEX_TY, A_nnz, "A matrix coordinates (create_output_buffer)");
      } else {
        A_crd_int = legate::Buffer<int32_t, 1>(
          create_1d_extents(0, A_nnz - 1), Memory::GPU_FB_MEM, NULL, BUFFER_DEFAULT_ALIGNMENT);
        LOG_BUFFER(int32_t, A_nnz, "A matrix coordinates (create_output_buffer)");
      }
      auto A_vals_acc = A_vals.create_output_buffer<VAL_TY, 1>(A_nnz, true /* return_buffer */);
      LOG_BUFFER(VAL_TY, A_nnz, "A matrix values (create_output_buffer)");

      CHECK_CUSPARSE(
        cusparseCsrSetPointers(cusparse_A, A_indptr.ptr(0), A_crd_int.ptr(0), A_vals_acc.ptr(0)));
      CHECK_CUSPARSE(cusparseSpGEMM_copy(handle,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &alpha,
                                         cusparse_B,
                                         cusparse_C,
                                         &beta,
                                         cusparse_A,
                                         cusparseDataType<VAL_TY>(),
                                         cusparse_alg,
                                         descr));
      // Cast the A coordinates back into 64 bits, if that is the desired
      // data type.
      if constexpr (INDEX_CODE != Type::Code::INT32) {
        auto blocks = get_num_blocks_1d(A_nnz);
        auto buf    = A_crd.create_output_buffer<INDEX_TY, 1>(A_nnz, true /* return_buffer */);
        LOG_BUFFER(INDEX_TY, A_nnz, "A matrix coordinates casting (output buffer)");
        cast<INDEX_TY, int32_t>
          <<<blocks, THREADS_PER_BLOCK, 0, stream>>>(A_nnz, buf.ptr(0), A_crd_int.ptr(0));
      }

      int64_t offset_nnz = 0;
      // scan to create global `pos` partition
      if (task_num > 1) {
        //@TODO (marsaev): we don't really need nccl comm here
        // latency for 1 int and host comm should be much better
        ncclComm_t* comm = args.comms[0].get<ncclComm_t*>();
        offset_nnz       = local_offset_from_nnz(*comm, task_id, task_num, A_nnz, stream);
      }

      // Convert the A_indptr array into a pos array.
      {
        auto blocks = get_num_blocks_1d(A_rows);
        localIndptrToPos<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
          A_rows,
          A_pos.write_accessor<Rect<1>, 1>().ptr(A_pos.domain().lo()),
          A_indptr.ptr(0),
          offset_nnz);
      }
    }  // cusparse alg1
#ifdef CUSPARSE_HAS_ALG3
    else if (cusparse_alg == CUSPARSE_SPGEMM_ALG3)

    {
      CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle,
                                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   &alpha,
                                                   cusparse_B,
                                                   cusparse_C,
                                                   &beta,
                                                   cusparse_A,
                                                   cusparseDataType<VAL_TY>(),
                                                   cusparse_alg,
                                                   descr,
                                                   &bufferSize1,
                                                   nullptr));
      void* buffer1 = nullptr;
      if (bufferSize1 > 0) {
        auto buf = CREATE_BUFFER(char, bufferSize1, Memory::GPU_FB_MEM, "buffer1");
        buffer1  = buf.ptr(0);
      }
      CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle,
                                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   &alpha,
                                                   cusparse_B,
                                                   cusparse_C,
                                                   &beta,
                                                   cusparse_A,
                                                   cusparseDataType<VAL_TY>(),
                                                   cusparse_alg,
                                                   descr,
                                                   &bufferSize1,
                                                   buffer1));

      CHECK_CUSPARSE(cusparseSpGEMM_estimateMemory(handle,
                                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   &alpha,
                                                   cusparse_B,
                                                   cusparse_C,
                                                   &beta,
                                                   cusparse_A,
                                                   cusparseDataType<VAL_TY>(),
                                                   cusparse_alg,
                                                   descr,
                                                   alg3_spgemm_fraction,
                                                   &bufferSize3,
                                                   nullptr,
                                                   nullptr));
      void* buffer3 = nullptr;
      if (bufferSize3 > 0) {
        auto buf = CREATE_BUFFER(char, bufferSize3, Memory::GPU_FB_MEM, "buffer3");
        buffer3  = buf.ptr(0);
      }

      CHECK_CUSPARSE(cusparseSpGEMM_estimateMemory(handle,
                                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   &alpha,
                                                   cusparse_B,
                                                   cusparse_C,
                                                   &beta,
                                                   cusparse_A,
                                                   cusparseDataType<VAL_TY>(),
                                                   cusparse_alg,
                                                   descr,
                                                   alg3_spgemm_fraction,
                                                   &bufferSize3,
                                                   buffer3,
                                                   &bufferSize2));

      void* buffer2 = nullptr;
      if (bufferSize2 > 0) {
        auto buf = CREATE_BUFFER(char, bufferSize2, Memory::GPU_FB_MEM, "buffer2");
        buffer2  = buf.ptr(0);
      }

      CHECK_CUSPARSE(cusparseSpGEMM_compute(handle,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha,
                                            cusparse_B,
                                            cusparse_C,
                                            &beta,
                                            cusparse_A,
                                            cusparseDataType<VAL_TY>(),
                                            cusparse_alg,
                                            descr,
                                            &bufferSize2,
                                            buffer2));
      // Allocate buffers for the 32-bit version of the A matrix.
      CHECK_CUSPARSE(cusparseSpMatGetSize(cusparse_A, &A_rows, &A_cols, &A_nnz));
      auto A_indptr = CREATE_BUFFER(int32_t, A_rows + 1, Memory::GPU_FB_MEM, "A_indptr");
      // Handle the creation of the A_crd buffer depending on whether the result
      // type is the type of data we are supposed to create.
      legate::Buffer<int32_t, 1> A_crd_int;
      if constexpr (INDEX_CODE == Type::Code::INT32) {
        A_crd_int = A_crd.create_output_buffer<INDEX_TY, 1>(A_nnz, true /* return_buffer */);
        LOG_BUFFER(INDEX_TY, A_nnz, "A matrix coordinates (create_output_buffer)");
      } else {
        A_crd_int = legate::Buffer<int32_t, 1>(
          create_1d_extents(0, A_nnz - 1), Memory::GPU_FB_MEM, NULL, BUFFER_DEFAULT_ALIGNMENT);
        LOG_BUFFER(int32_t, A_nnz, "A matrix coordinates (create_output_buffer)");
      }
      auto A_vals_acc = A_vals.create_output_buffer<VAL_TY, 1>(A_nnz, true /* return_buffer */);
      LOG_BUFFER(VAL_TY, A_nnz, "A matrix values (create_output_buffer)");

      CHECK_CUSPARSE(
        cusparseCsrSetPointers(cusparse_A, A_indptr.ptr(0), A_crd_int.ptr(0), A_vals_acc.ptr(0)));
      CHECK_CUSPARSE(cusparseSpGEMM_copy(handle,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &alpha,
                                         cusparse_B,
                                         cusparse_C,
                                         &beta,
                                         cusparse_A,
                                         cusparseDataType<VAL_TY>(),
                                         cusparse_alg,
                                         descr));
      int64_t offset_nnz = 0;
      // scan to create global `pos` partition
      if (task_num > 1) {
        //@TODO (marsaev): we don't really need nccl comm here
        // latency for 1 int and host comm should be much better
        ncclComm_t* comm = args.comms[0].get<ncclComm_t*>();
        offset_nnz       = local_offset_from_nnz(*comm, task_id, task_num, A_nnz, stream);
      }

      // Convert the A_indptr array into a pos array.
      {
        auto blocks = get_num_blocks_1d(A_rows);
        localIndptrToPos<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
          A_rows,
          A_pos.write_accessor<Rect<1>, 1>().ptr(A_pos.domain().lo()),
          A_indptr.ptr(0),
          offset_nnz);
      }
      // Cast the A coordinates back into 64 bits, if that is the desired
      // data type.
      if constexpr (INDEX_CODE != Type::Code::INT32) {
        auto blocks = get_num_blocks_1d(A_nnz);
        auto buf    = A_crd.create_output_buffer<INDEX_TY, 1>(A_nnz, true /* return_buffer */);
        LOG_BUFFER(INDEX_TY, A_nnz, "A matrix coordinates casting (output buffer)");
        cast<INDEX_TY, int32_t>
          <<<blocks, THREADS_PER_BLOCK, 0, stream>>>(A_nnz, buf.ptr(0), A_crd_int.ptr(0));
      }
    }  // cusparse alg3
#endif

    // Destroy all of the resources that we allocated.
    CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(descr));
    CHECK_CUSPARSE(cusparseDestroySpMat(cusparse_A));
    CHECK_CUSPARSE(cusparseDestroySpMat(cusparse_B));
    CHECK_CUSPARSE(cusparseDestroySpMat(cusparse_C));
    LEGATE_SPARSE_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void SpGEMMCSRxCSRxCSRGPU::gpu_variant(TaskContext context)
{
  auto inputs  = context.inputs();
  auto outputs = context.outputs();
  SpGEMMCSRxCSRxCSRGPUArgs args{outputs[0],
                                outputs[1],
                                outputs[2],
                                inputs[0],
                                inputs[1],
                                inputs[2],
                                inputs[3],
                                inputs[4],
                                inputs[5],
                                context.scalars()[0].value<uint64_t>(),
                                context.scalars()[1].value<uint64_t>(),
                                context.scalars()[2].value<uint64_t>(),
                                context.communicators()};
  index_type_floating_point_value_type_dispatch(args.A_crd.code(),
                                                args.A_vals.code(),
                                                SpGEMMCSRxCSRxCSRGPUImpl{context},
                                                args,
                                                context.get_task_index()[0],
                                                context.get_launch_domain().hi()[0]);
}

}  // namespace sparse
