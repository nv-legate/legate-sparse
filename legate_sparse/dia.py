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
import numpy
import scipy  # type: ignore

from .base import CompressedBase
from .coverage import clone_scipy_arr_kind
from .csr import csr_array
from .types import coord_ty
from .utils import (
    cast_arr,
    get_store_from_cupynumeric_array,
    store_to_cupynumeric_array,
)


# Temporary implementation for matrix generation in examples
@clone_scipy_arr_kind(scipy.sparse.dia_array)
class dia_array(CompressedBase):
    def __init__(self, arg, shape=None, dtype=None, copy=False):
        if shape is None:
            raise NotImplementedError
        assert isinstance(arg, tuple)
        data, offsets = arg
        if isinstance(offsets, int):
            offsets = cupynumeric.full((1,), offsets)
        data, offsets = cast_arr(data), cast_arr(offsets)
        if dtype is not None:
            data = data.astype(dtype)
        dtype = data.dtype
        assert dtype is not None
        if not isinstance(dtype, numpy.dtype):
            dtype = numpy.dtype(dtype)

        self.dtype = dtype
        # Ensure that we don't accidentally include ndarray
        # objects as the elements of our shapes, as that can
        # lead to reference cycles or issues when talking to
        # legate under the hood.
        self.shape = tuple(int(i) for i in shape)
        self._offsets = get_store_from_cupynumeric_array(offsets, copy=copy)
        self._data = get_store_from_cupynumeric_array(data, copy=copy)

    @property
    def nnz(self):
        M, N = self.shape
        nnz = 0
        for k in self.offsets:
            if k > 0:
                nnz += min(M, N - k)
            else:
                nnz += min(M + k, N)
        return int(nnz)

    @property
    def data(self):
        return store_to_cupynumeric_array(self._data)

    @property
    def offsets(self):
        return store_to_cupynumeric_array(self._offsets)

    def copy(self):
        data = cupynumeric.array(self.data)
        offsets = cupynumeric.array(self.offsets)
        return dia_array((data, offsets), shape=self.shape, dtype=self.dtype)

    def transpose(self, axes=None, copy=False):
        if axes is not None:
            raise ValueError(
                "Sparse matrices do not support "
                "an 'axes' parameter because swapping "
                "dimensions is the only logical permutation."
            )
        if copy:
            raise AssertionError

        num_rows, num_cols = self.shape
        max_dim = max(self.shape)

        # flip diagonal offsets
        offsets = -self.offsets

        # re-align the data matrix
        r = cupynumeric.arange(len(offsets), dtype=coord_ty)[:, None]
        c = cupynumeric.arange(num_rows, dtype=coord_ty) - (offsets % max_dim)[:, None]
        pad_amount = max(0, max_dim - self.data.shape[1])
        data = cupynumeric.hstack(
            (
                self.data,
                cupynumeric.zeros(
                    (self.data.shape[0], pad_amount), dtype=self.data.dtype
                ),
            )
        )
        data = data[r, c]
        return dia_array(
            (data, offsets),
            shape=(num_cols, num_rows),
            copy=copy,
            dtype=self.dtype,
        )

    T = property(transpose)

    def tocsr(self, copy=False):
        if copy:
            return self.copy().tocsr(copy=False)
        # we don't need secondary copy
        return self.transpose(copy=copy)._tocsr_transposed(copy=False)

    # This routine is lifted from scipy.sparse's converter.
    def _tocsr_transposed(self, copy=False):
        if self.nnz == 0:
            return csr_array(self.shape, self.dtype)

        num_rows, num_cols = self.shape
        num_offsets, offset_len = self.data.shape
        offset_inds = cupynumeric.arange(offset_len)

        row = offset_inds - self.offsets[:, None]
        mask = row >= 0
        mask &= row < num_rows
        mask &= offset_inds < num_cols
        mask &= self.data != 0

        idx_dtype = coord_ty
        indptr = cupynumeric.zeros(num_cols + 1, dtype=idx_dtype)
        # note that the output dtype in a reduction (e.g, sum) determines
        # the dtype of the accumulator that is used in the reduction
        # in cupynumeric, it looks like the output dtype is set to the src
        # dtype if unspecified and that results in the output not performing
        # an integer sum. But we want the integer sum, so specify
        # dtype as idx_dtype to mask.sum()
        indptr[1 : offset_len + 1] = cupynumeric.cumsum(
            mask.sum(axis=0, dtype=idx_dtype)[:num_cols]
        )
        if offset_len < num_cols:
            indptr[offset_len + 1 :] = indptr[offset_len]
        indices = row.T[mask.T].astype(idx_dtype, copy=False)
        data = self.data.T[mask.T]
        return csr_array(
            (data, indices, indptr), shape=self.shape, dtype=self.dtype, copy=False
        )


# Declare an alias for this type.
dia_matrix = dia_array
