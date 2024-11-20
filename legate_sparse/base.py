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

import cupynumeric
from legate.core import LogicalStore  # type: ignore[attr-defined]
from legate.core import align  # type: ignore[attr-defined]

from .config import SparseOpCode, rect1
from .runtime import runtime
from .types import int64
from .utils import (
    copy_store,
    get_storage_type,
    get_store_from_cupynumeric_array,
    store_to_cupynumeric_array,
)


# CompressedBase is a base class for several different kinds of sparse
# matrices, such as CSR, CSC, COO and DIA.
class CompressedBase:
    @classmethod
    def nnz_to_pos_cls(cls, q_nnz: LogicalStore):
        q_nnz_arr = store_to_cupynumeric_array(q_nnz)
        cs = cupynumeric.cumsum(q_nnz_arr)
        cs_shifted = cs - q_nnz_arr
        cs_store = get_store_from_cupynumeric_array(cs)
        cs_shifted_store = get_store_from_cupynumeric_array(cs_shifted)
        # Zip the scan result into a rect1 region for the pos.
        pos = runtime.create_store(
            rect1,  # type: ignore
            shape=(q_nnz.shape[0],),
            optimize_scalar=False,
        )
        task = runtime.create_auto_task(SparseOpCode.ZIP_TO_RECT1)
        pos_var = task.add_output(pos)
        cs_shifted_var = task.add_input(cs_shifted_store)
        cs_var = task.add_input(cs_store)
        task.add_constraint(align(pos_var, cs_shifted_var))
        task.add_constraint(align(cs_shifted_var, cs_var))
        task.execute()
        # Don't convert cs[-1] to an int to avoid blocking.
        return pos, cs[-1]

    def nnz_to_pos(self, q_nnz: LogicalStore):
        return CompressedBase.nnz_to_pos_cls(q_nnz)

    def asformat(self, format, copy=False):
        if format is None or format == self.format:
            if copy:
                raise NotImplementedError
            else:
                return self
        else:
            try:
                convert_method = getattr(self, "to" + format)
            except AttributeError as e:
                raise ValueError("Format {} is unknown.".format(format)) from e

            # Forward the copy kwarg, if it's accepted.
            try:
                return convert_method(copy=copy)
            except TypeError:
                return convert_method()

    # The implementation of sum is mostly lifted from scipy.sparse.
    def sum(self, axis=None, dtype=None, out=None):
        """
        Sum the matrix elements over a given axis.
        Parameters
        ----------
        axis : {-2, -1, 0, 1, None} optional
            Axis along which the sum is computed. The default is to
            compute the sum of all the matrix elements, returning a scalar
            (i.e., `axis` = `None`).
        dtype : dtype, optional
            The type of the returned matrix and of the accumulator in which
            the elements are summed.  The dtype of `a` is used by default
            unless `a` has an integer dtype of less precision than the default
            platform integer.  In that case, if `a` is signed then the platform
            integer is used while if `a` is unsigned then an unsigned integer
            of the same precision as the platform integer is used.
            .. versionadded:: 0.18.0
        out : np.matrix, optional
            Alternative output matrix in which to place the result. It must
            have the same shape as the expected output, but the type of the
            output values will be cast if necessary.
            .. versionadded:: 0.18.0
        Returns
        -------
        sum_along_axis : np.matrix
            A matrix with the same shape as `self`, with the specified
            axis removed.
        See Also
        --------
        numpy.matrix.sum : NumPy's implementation of 'sum' for matrices
        """

        # We use multiplication by a matrix of ones to achieve this.
        # For some sparse matrix formats more efficient methods are
        # possible -- these should override this function.
        m, n = self.shape

        # Mimic numpy's casting.
        res_dtype = self.dtype

        if axis is None:
            return self.data.sum(dtype=res_dtype, out=out)

        if axis < 0:
            axis += 2

        # axis = 0 or 1 now
        if axis == 0:
            # sum over columns
            # TODO: (marsaev) currently not supported as we don't have rmatmul yet
            # (need CSC to have easier sum over columns)
            raise NotImplementedError
            ret = self.__rmatmul__(cupynumeric.ones((1, m), dtype=res_dtype))
        else:
            # sum over rows
            ret = self @ cupynumeric.ones((n, 1), dtype=res_dtype)

        if out is not None and out.shape != ret.shape:
            raise ValueError("dimensions do not match")

        return ret.sum(axis=axis, dtype=dtype, out=out)

    # needed by _data_matrix
    def _with_data(self, data, copy=True):
        """Returns a _different_ matrix object with the same sparsity structure as self,
        but with different data.  By default the structure arrays
        (i.e. .indptr and .indices) are copied. 'data' parameter is never copied.
        """

        # For CSR and CSC compressed base we can just reuse compressed stores,
        # Create copy if needed
        if copy:
            return self.__class__(
                (data, copy_store(self.crd), copy_store(self.pos)),
                shape=self.shape,
                dtype=get_storage_type(data),
                # we already made copies where needed
                copy=False,
            )
        else:
            return self.__class__(
                (data, self.crd, self.pos),
                shape=self.shape,
                dtype=get_storage_type(data),
                copy=False,
            )

    def astype(self, dtype, casting="unsafe", copy=True):
        dtype = cupynumeric.dtype(dtype)
        # if type doesn't match, create a matrix copy with casted data array
        if self.dtype != dtype:
            return self._with_data(
                self.data.astype(dtype, casting=casting, copy=True), copy=copy
            )
        else:
            return self.copy() if copy else self


# These univariate ufuncs preserve zeros.
_ufuncs_with_fixed_point_at_zero = frozenset(
    [
        cupynumeric.sin,
        cupynumeric.tan,
        cupynumeric.arcsin,
        cupynumeric.arctan,
        cupynumeric.sinh,
        cupynumeric.tanh,
        cupynumeric.arcsinh,
        cupynumeric.arctanh,
        cupynumeric.rint,
        cupynumeric.sign,
        cupynumeric.expm1,
        cupynumeric.log1p,
        cupynumeric.deg2rad,
        cupynumeric.rad2deg,
        cupynumeric.floor,
        cupynumeric.ceil,
        cupynumeric.trunc,
        cupynumeric.sqrt,
    ]
)

# Add the numpy unary ufuncs for which func(0) = 0 to _data_matrix.
for npfunc in _ufuncs_with_fixed_point_at_zero:
    name = npfunc.__name__

    def _create_method(op):
        def method(self):
            result = op(self.data)
            return self._with_data(result)

        method.__doc__ = "Element-wise %s.\n\nSee `numpy.%s` for more information." % (
            name,
            name,
        )
        method.__name__ = name

        return method

    setattr(CompressedBase, name, _create_method(npfunc))


# DenseSparseBase is a base class for sparse matrices that have a TACO
# format of {Dense, Sparse}. For our purposes, that means CSC and CSR
# matrices.
class DenseSparseBase:
    def __init__(self):
        self._balanced_pos_partition = None

    # consider using _with_data() here
    @classmethod
    def make_with_same_nnz_structure(cls, mat, arg, shape=None, dtype=None):
        if shape is None:
            shape = mat.shape
        if dtype is None:
            dtype = mat.dtype
        result = cls(arg, shape=shape, dtype=dtype)
        return result


# unpack_rect1_store unpacks a rect1 store into two int64 stores.
def unpack_rect1_store(pos):
    out1 = runtime.create_store(int64, shape=pos.shape)
    out2 = runtime.create_store(int64, shape=pos.shape)
    task = runtime.create_auto_task(SparseOpCode.UNZIP_RECT1)
    lo_var = task.add_output(out1)
    hi_var = task.add_output(out2)
    src_var = task.add_input(pos)
    task.add_constraint(align(lo_var, hi_var))
    task.add_constraint(align(hi_var, src_var))
    task.execute()
    return out1, out2


# pack_to_rect1_store packs two int64 stores into a rect1 store.
def pack_to_rect1_store(lo, hi, output=None):
    if output is None:
        output = runtime.create_store(rect1, shape=(lo.shape[0],))
    task = runtime.create_auto_task(SparseOpCode.ZIP_TO_RECT1)
    out_var = task.add_output(output)
    lo_var = task.add_input(lo)
    hi_var = task.add_input(hi)
    task.add_constraint(align(lo_var, hi_var))
    task.add_constraint(align(hi_var, out_var))
    task.execute()
    return output
