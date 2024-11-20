# Copyright 2022-2024 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Portions of this file are also subject to the following license:
#
# Copyright (c) 2001-2002 Enthought, Inc. 2003-2022, SciPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided
# with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived
# from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import warnings

import cupynumeric
import numpy
import scipy  # type: ignore
from legate.core import ImageComputationHint  # type: ignore[attr-defined]
from legate.core import Shape  # type: ignore[attr-defined]
from legate.core import align  # type: ignore[attr-defined]
from legate.core import broadcast  # type: ignore[attr-defined]
from legate.core import image  # type: ignore[attr-defined]
from legate.core import types  # type: ignore[attr-defined]

from .base import (
    CompressedBase,
    DenseSparseBase,
    pack_to_rect1_store,
    unpack_rect1_store,
)
from .config import SparseOpCode, rect1
from .coverage import clone_scipy_arr_kind
from .runtime import runtime
from .settings import settings
from .types import coord_ty, nnz_ty
from .utils import (
    SUPPORTED_DATATYPES,
    array_from_store_or_array,
    cast_arr,
    cast_to_common_type,
    cast_to_store,
    copy_store,
    find_last_user_stacklevel,
    get_storage_type,
    get_store_from_cupynumeric_array,
    is_dtype_supported,
    store_from_store_or_array,
    store_to_cupynumeric_array,
)


@clone_scipy_arr_kind(scipy.sparse.csr_array)
class csr_array(CompressedBase, DenseSparseBase):
    def __init__(self, arg, shape=None, dtype=None, copy=False):
        self.ndim = 2
        self.indices_sorted = False
        self.canonical_format = False
        super().__init__()

        dtype = cupynumeric.dtype(dtype)

        # If from numpy.array - convert to cupynumeric array first
        if isinstance(arg, numpy.ndarray):
            arg = cupynumeric.array(arg)

        # from scipy.sparse.csr_array
        if isinstance(arg, scipy.sparse.csr_array) or isinstance(
            arg, scipy.sparse.csr_matrix
        ):
            shape = arg.shape
            arg = (arg.data, arg.indices, arg.indptr)

        # from dense cupynumeric array
        if isinstance(arg, cupynumeric.ndarray):
            assert arg.ndim == 2

            shape = arg.shape

            # We'll do a row-wise distribution and use a two-pass algorithm that
            # first counts the non-zeros per row and then fills them in.
            src_store = get_store_from_cupynumeric_array(arg)

            q_nnz = runtime.create_store(nnz_ty, shape=Shape((shape[0],)))
            task = runtime.create_auto_task(SparseOpCode.DENSE_TO_CSR_NNZ)
            promoted_q_nnz = q_nnz.promote(1, shape[1])
            nnz_per_row_part = task.add_output(promoted_q_nnz)
            src_part = task.add_input(src_store)
            task.add_constraint(broadcast(nnz_per_row_part, (1,)))
            task.add_constraint(align(nnz_per_row_part, src_part))
            task.execute()

            # Assemble the output CSR array using the non-zeros per row.
            self.pos, nnz = self.nnz_to_pos(q_nnz)
            # Block and convert the nnz future into an int.
            nnz = int(nnz)
            self.crd = runtime.create_store(coord_ty, shape=((nnz,)))
            self.vals = runtime.create_store(arg.dtype, shape=((nnz,)))

            # TODO (marsaev): since in Legate we cannot align 1-D arrays of CSR data
            # and 2-D input array, our only option is launch single process
            # which will handle all of the data, which makes this funciton not usable
            # on scale.
            task = runtime.create_manual_task(SparseOpCode.DENSE_TO_CSR, (1,))

            promoted_pos = self.pos.promote(1, shape[1])
            task.add_input(promoted_pos)
            src_part = task.add_input(src_store)
            task.add_output(self.crd)
            task.add_output(self.vals)
            task.execute()

            # we ignore dtype (TODO: is this behaviour matches SciPy?) and use arg.dtype
            dtype = arg.dtype

        # Ctor that copies csr_array
        elif isinstance(arg, csr_array):
            shape = arg.shape
            self.pos = copy_store(arg.pos)
            self.crd = copy_store(arg.crd)
            self.vals = copy_store(arg.vals)
            self.indices_sorted = arg.indices_sorted
            self.canonical_format = arg.canonical_format

        elif isinstance(arg, tuple):
            # Couple of options here
            if len(arg) == 2:
                # empty array ctor, see scipy.sparse
                # csr_array((M, N), [dtype])
                if not isinstance(arg[1], tuple):
                    (M, N) = arg
                    if not isinstance(M, (int, numpy.integer)) or not isinstance(
                        N, (int, numpy.integer)
                    ):
                        NotImplementedError(
                            "Input tuple for empty CSR ctor should be it's shape"
                        )
                    shape = arg
                    if dtype is None:
                        dtype = cupynumeric.float64
                    else:
                        dtype = cupynumeric.dtype(dtype)
                    nnz_arr = cupynumeric.zeros(0, dtype=dtype)
                    ci_arr = cupynumeric.zeros(0, dtype=coord_ty)
                    rptr_arr = cupynumeric.zeros(M + 1, dtype=coord_ty)
                    # and pass this to next ctor
                    arg = (nnz_arr, ci_arr, rptr_arr)

                # Otherwise assume arg is COO data : (data, (row_ind, col_ind))
                else:
                    if shape is None:
                        raise AssertionError("Cannot infer shape in this case.")

                    st_data, (st_row, st_col) = arg

                    # if passed numpy arrays - convert them
                    if isinstance(st_data, numpy.ndarray):
                        st_data = cupynumeric.array(st_data)
                    if isinstance(st_row, numpy.ndarray):
                        st_row = cupynumeric.array(st_row)
                    if isinstance(st_col, numpy.ndarray):
                        st_col = cupynumeric.array(st_col)

                    # we assume nothing is sorted (be we can pass this information to ctor)
                    # so sort by row indices:
                    row_array = array_from_store_or_array(st_row)
                    # if we would know that column indices are pre-sorted,
                    # then we can use kind='stable' and mark csr_array as
                    # with 'indices_sorted'
                    row_sort = cupynumeric.argsort(row_array, kind="stable")

                    # sort data based on rows
                    new_data = array_from_store_or_array(st_data, copy=copy)[row_sort]
                    new_col_ind = array_from_store_or_array(st_col, copy=copy)[row_sort]
                    new_row_offsets = cupynumeric.append(
                        cupynumeric.array([0]),
                        cupynumeric.cumsum(
                            cupynumeric.bincount(row_array, minlength=shape[0])
                        ),
                    )

                    # pass to next ctor
                    arg = (new_data, new_col_ind, new_row_offsets)
                    # we created copies already if necessary
                    copy = False

            # ctor from CSR arrays
            # Tuple of (vals, col_ind, row_offsets)
            if len(arg) == 3:
                if shape is None or len(shape) != 2:
                    raise AssertionError("Cannot infer shape in this case.")

                (data, indices, indptr) = arg

                # if passed numpy arrays - convert them
                if isinstance(data, numpy.ndarray):
                    data = cupynumeric.array(data)
                if isinstance(indices, numpy.ndarray):
                    indices = cupynumeric.array(indices).astype(coord_ty)
                if isinstance(indptr, numpy.ndarray):
                    indptr = cupynumeric.array(indptr).astype(coord_ty)

                # checking that shape matches with expectations for row_offsets
                if indptr.shape[0] == shape[0] + 1:
                    indptr_storage = array_from_store_or_array(indptr, copy=False)
                    los = indptr_storage[:-1]
                    his = indptr_storage[1:]
                    self.pos = pack_to_rect1_store(
                        get_store_from_cupynumeric_array(los),
                        get_store_from_cupynumeric_array(his),
                    )
                    # copy explicitly, just in case (there are paths that won't create temp object)
                    # For crd we enforce our internal type
                    self.crd = store_from_store_or_array(
                        cast_arr(indices, coord_ty), copy
                    )
                    self.vals = store_from_store_or_array(cast_to_store(data), copy)

                # Otherwise we assume that we are passing pos store from existing csr_array
                # This is internal only functionality, and we assume here only Store or cupynumeric.array
                elif indptr.shape[0] == shape[0]:
                    self.pos = store_from_store_or_array(indptr, copy)
                    self.crd = store_from_store_or_array(indices, copy)
                    self.vals = store_from_store_or_array(data, copy)

                else:
                    raise AssertionError(
                        "Can't understand tuple of inputs for csr_array constructor"
                    )

                dtype = get_storage_type(data)
        else:
            raise NotImplementedError("Can't convert to CSR from the input")

        assert shape is not None
        # Ensure that we don't accidentally include ndarray
        # objects as the elements of our shapes, as that can
        # lead to reference cycles or issues when talking to
        # legate under the hood.
        self.shape = tuple(int(i) for i in shape)

        # Use the user's dtype if requested, otherwise infer it from
        # the input data.
        temp_vals_type = get_storage_type(self.vals)
        if dtype is None:
            dtype = temp_vals_type
        if temp_vals_type is not dtype:
            self.data = self.data.astype(dtype)
        if not isinstance(dtype, numpy.dtype):
            dtype = numpy.dtype(dtype)
        # Saving the type
        self._dtype = dtype

    @property
    def dim(self):
        return self.ndim

    @property
    def nnz(self):
        return self.vals.shape[0]

    @property
    def dtype(self):
        # We can just return self.vals.type, but bookkeep type separately now
        return self._dtype

    # Enable direct operation on the values array.
    def get_data(self):
        return store_to_cupynumeric_array(self.vals)

    # From array,
    def set_data(self, data):
        if isinstance(data, numpy.ndarray):
            data = cupynumeric.array(data)
        assert isinstance(data, cupynumeric.ndarray)
        self.vals = get_store_from_cupynumeric_array(data)
        self._dtype = data.dtype

    data = property(fget=get_data, fset=set_data)

    # Enable direct operation on the indices array.
    def get_indices(self):
        return store_to_cupynumeric_array(self.crd)

    def set_indices(self, indices):
        if isinstance(indices, numpy.ndarray):
            indices = cupynumeric.array(indices)
        assert isinstance(indices, cupynumeric.ndarray)
        self.crd = get_store_from_cupynumeric_array(indices)
        # we can't guarantee new indices are sorted
        self.canonical_format = False
        self.indices_sorted = False

    indices = property(fget=get_indices, fset=set_indices)

    def get_indptr(self):
        row_start_st, row_end_st = unpack_rect1_store(self.pos)
        row_start = store_to_cupynumeric_array(row_start_st)
        return cupynumeric.append(row_start, [self.nnz])

    # Disallow changing intptrs directly
    indptr = property(fget=get_indptr)

    def has_sorted_indices(self):
        return self.indices_sorted

    def has_canonical_format(self):
        return self.canonical_format

    # The rest of the methods
    def diagonal(self, k=0):
        rows, cols = self.shape
        if k <= -rows or k >= cols:
            return cupynumeric.empty(0, dtype=self.dtype)
        output = runtime.create_store(
            self.dtype, shape=Shape((min(rows + min(k, 0), cols - max(k, 0)),))
        )

        # Only k = 0 is supported, returm main diagonal
        if k != 0:
            raise NotImplementedError

        task = runtime.create_auto_task(SparseOpCode.CSR_DIAGONAL)

        out_part = task.add_output(output)
        pos_part = task.add_input(self.pos)
        crd_part = task.add_input(self.crd)
        val_part = task.add_input(self.vals)

        task.add_constraint(align(out_part, pos_part))
        task.add_constraint(image(pos_part, crd_part))
        task.add_constraint(align(crd_part, val_part))
        task.execute()
        return store_to_cupynumeric_array(output)

    def todense(self, order=None, out=None):
        if order is not None:
            raise NotImplementedError
        if out is not None:
            out = cupynumeric.array(out)
            if out.dtype != self.dtype:
                raise ValueError(
                    f"Output type {out.dtype} is not consistent with dtype {self.dtype}"
                )
            out = get_store_from_cupynumeric_array(out)
        elif out is None:
            out = runtime.create_store(self.dtype, shape=self.shape)

        task = runtime.create_manual_task(SparseOpCode.CSR_TO_DENSE, (1,))
        self.pos.promote(1, self.shape[1])
        task.add_output(out)
        task.add_input(self.pos)
        task.add_input(self.crd)
        task.add_input(self.vals)
        task.execute()
        return store_to_cupynumeric_array(out)

    def multiply(self, other):
        return self * other

    def __rmul__(self, other):
        return self * other

    # This is an element-wise operation now.
    def __mul__(self, other):
        if isinstance(other, numpy.ndarray):
            other = cupynumeric.array(other)

        if cupynumeric.ndim(other) == 0:
            # If we have a scalar, then do an element-wise multiply on the
            # values array.
            new_vals = store_to_cupynumeric_array(self.vals) * other
            return self._with_data(new_vals)
        else:
            raise NotImplementedError

    # rmatmul represents the operation other @ self.
    def __rmatmul__(self, other):
        # Handle dense @ CSR
        raise NotImplementedError

    def __matmul__(self, other):
        return self.dot(other)

    def dot(self, other, out=None):
        # If output specified - it should be cupynumeric array
        if out is not None:
            assert isinstance(out, cupynumeric.ndarray)

        # only floating point operations are supported at the moment
        if not is_dtype_supported(self.dtype) or not is_dtype_supported(other.dtype):
            msg = (
                "Only the following datatypes are currently supported:"
                f" {SUPPORTED_DATATYPES}."
            )
            raise NotImplementedError(msg)

        # If other.shape = (M,) then it's SpMV
        if len(other.shape) == 1 or (len(other.shape) == 2 and other.shape[1] == 1):
            # convert X to the cupynumeric array if needed
            if not isinstance(other, cupynumeric.ndarray):
                other = cupynumeric.array(other)
            assert self.shape[1] == other.shape[0]
            # for the case of X shape == (M, 1)
            other_originally_2d = False
            if len(other.shape) == 2 and other.shape[1] == 1:
                other = other.squeeze(1)
                other_originally_2d = True

            other_store = get_store_from_cupynumeric_array(other)
            if other_store.transformed:
                level = find_last_user_stacklevel()
                warnings.warn(
                    "CSR SpMV creating an implicit copy due to transformed x vector.",
                    category=RuntimeWarning,
                    stacklevel=level,
                )
                other = cupynumeric.array(other)

            # Coerce A and x into a common type. Use that coerced type
            # to find the type of the output.
            A, x = cast_to_common_type(self, other)
            if out is None:
                y = store_to_cupynumeric_array(
                    runtime.create_store(A.dtype, shape=(self.shape[0],))
                )
            else:
                # We can't use the output if it not the correct type,
                # as then we can't guarantee that we would write into
                # it. So, error out if the output type doesn't match
                # the resolved type of A and x.
                if out.dtype != A.dtype:
                    raise ValueError(
                        f"Output type {out.dtype} is not consistent "
                        f"with resolved dtype {A.dtype}"
                    )
                if other_originally_2d:
                    assert out.shape == (self.shape[0], 1)
                    out = out.squeeze(1)
                else:
                    assert out.shape == (self.shape[0],)
                y = out

            # Invoke the SpMV after the setup.
            spmv(A, x, y)

            output = y
            if other_originally_2d:
                output = output.reshape((-1, 1))

            return output
        # If other is CSR array - it's SpGEMM: CSRxCSR -> CSR
        elif isinstance(other, csr_array):
            if out is not None:
                raise ValueError("Cannot provide out for CSRxCSR matmul.")
            assert self.shape[1] == other.shape[0]
            return spgemm_csr_csr_csr(*cast_to_common_type(self, other))
        else:
            raise NotImplementedError

    # Misc
    def _getpos(self):
        row_start_st, row_end_st = unpack_rect1_store(self.pos)
        row_start = store_to_cupynumeric_array(row_start_st)
        row_end = store_to_cupynumeric_array(row_end_st)
        return [(i, j) for (i, j) in zip(row_start, row_end)]

    def copy(self):
        return csr_array(self)

    def conj(self, copy=True):
        if copy:
            return self.copy().conj(copy=False)
        return self._with_data(
            get_store_from_cupynumeric_array(self.data.conj()), copy=False
        )

    def transpose(self, axes=None, copy=False):
        if axes is not None:
            raise AssertionError("axes parameter should be None")

        # Currently we have only CSR format. That means that transpose here
        # is CSR -> CSR, which always will involve a copy of internal arrays

        # if copy:
        #    return self.copy().transpose(copy=False)

        rows_expanded = runtime.create_store(coord_ty, shape=self.crd.shape)
        task = runtime.create_auto_task(SparseOpCode.EXPAND_POS_TO_COORDINATES)
        src_part = task.add_input(self.pos)
        dst_part = task.add_output(rows_expanded)
        task.add_constraint(image(src_part, dst_part))

        task.execute()

        # sort
        sort_mask = cupynumeric.argsort(self.crd, kind="stable")
        new_rows = self.get_indices()[sort_mask]
        new_ci = store_to_cupynumeric_array(rows_expanded)[sort_mask]
        new_data = self.get_data()[sort_mask]

        # use freshly created arrays
        return csr_array(
            (new_data, (new_rows, new_ci)),
            shape=(self.shape[1], self.shape[0]),
            dtype=self.dtype,
            copy=False,
        )

    T = property(transpose)

    def asformat(seld, format, copy=False):
        if format == "csr":
            return self.copy() if copy else self
        else:
            raise NotImplementedError("Only CSR format is supported right now")

    def tocsr(self, copy=False):
        if copy:
            return self.copy().tocsr(copy=False)
        return self


csr_matrix = csr_array


# spmv computes y = A @ x.
def spmv(A: csr_array, x: cupynumeric.ndarray, y: cupynumeric.ndarray):
    """
    Perform sparse matrix vector product y = A @ x

    Parameters:
    -----------
    A: csr_array
        Input sparse matrix
    x: cupynumeric.ndarray
        Dense vector for the dot product
    y: cupynumeric.ndarray
        Output array
    """

    x_store = get_store_from_cupynumeric_array(x)
    y_store = get_store_from_cupynumeric_array(y)

    # An auto-parallelized version of the kernel.
    task = runtime.create_auto_task(SparseOpCode.CSR_SPMV_ROW_SPLIT)
    y_var = task.add_output(y_store)
    pos_var = task.add_input(A.pos)
    crd_var = task.add_input(A.crd)
    vals_var = task.add_input(A.vals)
    x_var = task.add_input(x_store)

    task.add_constraint(align(y_var, pos_var))
    task.add_constraint(image(pos_var, crd_var, hint=ImageComputationHint.FIRST_LAST))
    task.add_constraint(image(pos_var, vals_var, hint=ImageComputationHint.FIRST_LAST))
    # exact or approximate image to X
    task.add_constraint(image(crd_var, x_var, hint=ImageComputationHint.MIN_MAX))

    task.execute()


# spgemm_csr_csr_csr computes C = A @ B when A and B and
# both csr matrices, and returns the result C as a csr matrix.
def spgemm_csr_csr_csr(A: csr_array, B: csr_array) -> csr_array:
    # Due to limitations in cuSPARSE, we cannot use a uniform task
    # implementation for CSRxCSRxCSR SpGEMM across CPUs, OMPs and GPUs.
    # The GPU implementation will create a set of local CSR matrices
    # that will be aggregated into a global CSR.
    if runtime.num_gpus > 0:
        # replacement for the ImagePartition functor to get dense image
        # for rows of B, run separate task for this
        pos_rect = runtime.create_store(rect1, shape=(A.shape[0],))  # type: ignore
        task = runtime.create_auto_task(SparseOpCode.FAST_IMAGE_RANGE)
        A_pos_part = task.add_input(A.pos)
        A_crd_part = task.add_input(A.crd)
        B_pos_image_part = task.add_output(pos_rect)

        task.add_constraint(align(A_pos_part, B_pos_image_part))
        task.add_constraint(
            image(A_pos_part, A_crd_part, hint=ImageComputationHint.MIN_MAX)
        )

        task.execute()

        pos = runtime.create_store(rect1, shape=(A.shape[0],))  # type: ignore
        crd = runtime.create_store(coord_ty, ndim=1)
        vals = runtime.create_store(A.dtype, ndim=1)

        task = runtime.create_auto_task(SparseOpCode.SPGEMM_CSR_CSR_CSR_GPU)
        C_pos_part = task.add_output(pos)
        C_crd_part = task.add_output(crd)
        C_vals_part = task.add_output(vals)
        A_pos_part = task.add_input(A.pos)
        A_crd_part = task.add_input(A.crd)
        A_vals_part = task.add_input(A.vals)
        B_pos_part = task.add_input(B.pos)
        B_crd_part = task.add_input(B.crd)
        B_vals_part = task.add_input(B.vals)
        B_pos_image_part = task.add_input(pos_rect)

        # for inter-partition reduction and scans
        # Add communicator even for 1 proc, because we expect it in the task
        task.add_communicator("nccl")

        # Constraints
        # By-row split - same way for A and C
        task.add_constraint(align(A_pos_part, C_pos_part))
        task.add_constraint(
            image(A_pos_part, A_crd_part, hint=ImageComputationHint.MIN_MAX)
        )
        task.add_constraint(
            image(A_pos_part, A_vals_part, hint=ImageComputationHint.MIN_MAX)
        )
        # No partition for unbound stores
        # task.add_constraint(image(C_pos_part_out, C_crd_part))
        # task.add_constraint(image(C_pos_part_out, C_vals_part))

        # For B just taking an image (currently - exact) for the column indices of A partition
        # task.add_constraint(image(A_crd_part, B_pos_part))
        # TODO (marsaev): we replaced custom image functor with separate task.
        # Array class should provide this functionality
        task.add_constraint(align(A_pos_part, B_pos_image_part))
        task.add_constraint(
            image(B_pos_image_part, B_pos_part, hint=ImageComputationHint.MIN_MAX)
        )

        task.add_constraint(
            image(B_pos_part, B_crd_part, hint=ImageComputationHint.MIN_MAX)
        )
        task.add_constraint(
            image(B_pos_part, B_vals_part, hint=ImageComputationHint.MIN_MAX)
        )
        # num columns in output
        task.add_scalar_arg(B.shape[1], types.uint64)
        # folded dimension
        task.add_scalar_arg(B.shape[0], types.uint64)
        # 1 if we want to try faster algorithm but that
        # might need more available eager GPU scratch space
        # TODO (marsaev): it might make sense to add this as parameter to dot()
        task.add_scalar_arg(1 if settings.fast_spgemm() else 0, types.uint64)

        task.execute()

        # we can keep new stores in the new csr_array
        return csr_array(
            (vals, crd, pos),
            shape=(A.shape[0], B.shape[1]),
            dtype=A.dtype,
            copy=False,
        )
    else:
        # Create the query result.
        q_nnz = runtime.create_store(nnz_ty, shape=(A.shape[0],))
        task = runtime.create_auto_task(SparseOpCode.SPGEMM_CSR_CSR_CSR_NNZ)
        nnz_per_row_part = task.add_output(q_nnz)
        A_pos_part = task.add_input(A.pos)
        A_crd_part = task.add_input(A.crd)
        B_pos_part = task.add_input(B.pos)
        B_crd_part = task.add_input(B.crd)
        task.add_constraint(align(A_pos_part, nnz_per_row_part))
        task.add_constraint(image(A_pos_part, A_crd_part))

        # We'll only ask for the rows used by each partition by
        # following an image of pos through crd. We'll then use that
        # partition to declare the pieces of crd and vals of other that
        # are needed by the matmul. The resulting image of coordinates
        # into rows of other is not necessarily complete or disjoint.
        task.add_constraint(image(A_crd_part, B_pos_part))
        # Since the target partition of pos is likely not contiguous,
        # we can't use the CompressedImagePartition functor and have to
        # fall back to a standard functor. Since the source partition
        # of the rows is not complete or disjoint, the images into crd
        # and vals are not disjoint either.
        task.add_constraint(image(B_pos_part, B_crd_part))

        task.execute()

        pos, nnz = CompressedBase.nnz_to_pos_cls(q_nnz)
        # Block and convert the nnz future into an int.
        nnz = int(nnz)
        crd = runtime.create_store(coord_ty, shape=(nnz,))
        vals = runtime.create_store(A.dtype, shape=(nnz,))

        task = runtime.create_auto_task(SparseOpCode.SPGEMM_CSR_CSR_CSR)
        C_pos_part_out = task.add_output(pos)
        C_crd_part = task.add_output(crd)
        C_vals_part = task.add_output(vals)
        A_pos_part = task.add_input(A.pos)
        A_crd_part = task.add_input(A.crd)
        A_vals_part = task.add_input(A.vals)
        B_pos_part = task.add_input(B.pos)
        B_crd_part = task.add_input(B.crd)
        B_vals_part = task.add_input(B.vals)
        # Add pos to the inputs as well so that we get READ_WRITE
        # privileges.
        C_pos_part_in = task.add_input(pos)
        task.add_constraint(align(A_pos_part, C_pos_part_in))
        # Constraints
        # By-row split - same way for A and C
        task.add_constraint(align(A_pos_part, C_pos_part_out))
        task.add_constraint(image(A_pos_part, A_crd_part))
        task.add_constraint(image(A_pos_part, A_vals_part))
        task.add_constraint(image(C_pos_part_out, C_crd_part))
        task.add_constraint(image(C_pos_part_out, C_vals_part))
        # For B just taking an image (currently - exact) for the column indices of A partition
        task.add_constraint(image(A_crd_part, B_pos_part))
        task.add_constraint(image(B_pos_part, B_crd_part))
        task.add_constraint(image(B_pos_part, B_vals_part))

        task.execute()
        return csr_array(
            (vals, crd, pos),
            shape=Shape((A.shape[0], B.shape[1])),
        )
