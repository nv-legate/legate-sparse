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
from __future__ import annotations

from typing import TYPE_CHECKING

import cupynumeric as cn
import numpy as np
from legate.core import LogicalStore, align

from .config import SparseOpCode, rect1
from .runtime import runtime
from .types import int64
from .utils import (
    copy_store,
    get_storage_type,
    get_store_from_cupynumeric_array,
    store_to_cupynumeric_array,
)

if TYPE_CHECKING:
    from typing import Any, Callable

    import numpy.typing as npt

    from cupynumeric.types import CastingKind


# CompressedBase is a base class for several different kinds of sparse
# matrices, such as CSR, CSC, COO and DIA.
class CompressedBase:
    """Base class for compressed sparse matrix formats.

    This class provides common functionality for compressed sparse matrix
    formats like CSR, CSC, COO, and DIA. It handles the conversion from
    non-zero counts to position arrays and provides common operations.

    Notes
    -----
    This is an internal base class and should not be instantiated directly.
    Use specific format classes like csr_array instead.
    """

    shape: tuple[int, ...]
    pos: LogicalStore
    dtype: npt.dtype[Any]
    format: str
    crd: LogicalStore
    _data: cn.ndarray

    def __init__(self, *args: Any, **kw: Any) -> None:
        super().__init__(*args, **kw)

    @property
    def data(self) -> cn.ndarray:
        return self._data

    @property
    def size(self) -> int:
        raise NotImplementedError

    @classmethod
    def nnz_to_pos_cls(
        cls, q_nnz: LogicalStore
    ) -> tuple[LogicalStore, cn.ndarray]:
        """Convert non-zero counts to position arrays.

        This class method converts an array of non-zero counts per row/column
        into the position array used in compressed sparse formats.

        Parameters
        ----------
        q_nnz : LogicalStore
            Store containing the number of non-zeros per row/column.

        Returns
        -------
        tuple
            (pos, total_nnz) where pos is the position array and total_nnz
            is the total number of non-zeros.
        """
        q_nnz_arr = store_to_cupynumeric_array(q_nnz)
        cs = cn.cumsum(q_nnz_arr)
        cs_shifted = cs - q_nnz_arr
        cs_store = get_store_from_cupynumeric_array(cs)
        cs_shifted_store = get_store_from_cupynumeric_array(cs_shifted)
        # Zip the scan result into a rect1 region for the pos.
        pos = runtime.create_store(
            rect1, shape=(q_nnz.shape[0],), optimize_scalar=False
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

    def nnz_to_pos(
        self, q_nnz: LogicalStore
    ) -> tuple[LogicalStore, cn.ndarray]:
        """Convert non-zero counts to position arrays for this instance.

        Parameters
        ----------
        q_nnz : LogicalStore
            Store containing the number of non-zeros per row/column.

        Returns
        -------
        tuple
            (pos, total_nnz) where pos is the position array and total_nnz
            is the total number of non-zeros.
        """
        return CompressedBase.nnz_to_pos_cls(q_nnz)

    def copy(self) -> CompressedBase:
        raise NotImplementedError()

    def asformat(
        self, format: str | None, copy: bool = False
    ) -> CompressedBase:
        """Convert the matrix to a specified format.

        Parameters
        ----------
        format : str
            The desired format ('csr', 'csc', 'coo', etc.).
        copy : bool, optional
            Whether to create a copy. Default is False.

        Returns
        -------
        sparse matrix
            Matrix in the requested format.

        Raises
        ------
        ValueError
            If the format is unknown.
        NotImplementedError
            If conversion to the requested format is not implemented.
        """
        if format is None or format == self.format:
            if copy:
                raise NotImplementedError
            else:
                return self
        else:
            try:
                convert_method: Callable[..., CompressedBase] = getattr(
                    self, "to" + format
                )
            except AttributeError as e:
                raise ValueError("Format {} is unknown.".format(format)) from e

            # Forward the copy kwarg, if it's accepted.
            try:
                return convert_method(copy=copy)
            except TypeError:
                return convert_method()

    # The implementation of sum is mostly lifted from scipy.sparse.
    def sum(
        self,
        axis: int | None = None,
        dtype: npt.dtype[Any] | None = None,
        out: cn.ndarray | None = None,
    ) -> cn.ndarray:
        """Sum the matrix elements over a given axis.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None}, optional
            Axis along which the sum is computed. The default is to
            compute the sum of all the matrix elements, returning a scalar
            (i.e., `axis` = `None`).
        dtype : dtype, optional
            The type of the returned matrix and of the accumulator in which
            the elements are summed. The dtype of `a` is used by default
            unless `a` has an integer dtype of less precision than the default
            platform integer. In that case, if `a` is signed then the platform
            integer is used while if `a` is unsigned then an unsigned integer
            of the same precision as the platform integer is used.
        out : cupynumeric.ndarray, optional
            Alternative output array in which to place the result. It must
            have the same shape as the expected output, but the type of the
            output values will be cast if necessary.

        Returns
        -------
        sum_along_axis : cupynumeric.ndarray or scalar
            A matrix with the same shape as `self`, with the specified
            axis removed, or a scalar if axis=None.

        Raises
        ------
        NotImplementedError
            If axis=0 (sum over columns) is requested.
        ValueError
            If out is provided but has incompatible shape.

        Notes
        -----
        The implementation uses multiplication by a matrix of ones to achieve
        the sum. For some sparse matrix formats more efficient methods are
        possible and should override this function.

        Currently, summing over columns (axis=0) is not implemented due to
        the lack of right matrix multiplication support.

        See Also
        --------
        cupynumeric.matrix.sum : NumPy's implementation of 'sum' for matrices
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
            # ret = self.__rmatmul__(cn.ones((1, m), dtype=res_dtype))
        else:
            # sum over rows
            ret = self @ cn.ones((n, 1), dtype=res_dtype)

        if out is not None and out.shape != ret.shape:
            raise ValueError("dimensions do not match")

        return ret.sum(axis=axis, dtype=dtype, out=out)

    # needed by _data_matrix
    def _with_data(self, data: Any, copy: bool = True) -> CompressedBase:
        """Returns a matrix object with the same sparsity structure as self,
        but with different data.

        Parameters
        ----------
        data : array_like
            The new data array. This parameter is never copied.
        copy : bool, optional
            Whether to copy the structure arrays (indptr and indices).
            Default is True.

        Returns
        -------
        sparse matrix
            A new matrix with the same sparsity structure but different data.

        Notes
        -----
        This method creates a new matrix object with the same sparsity pattern
        but replaces the data array. The structure arrays (indptr and indices)
        are copied by default to avoid modifying the original matrix.
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

    def astype(
        self,
        dtype: npt.dtype[Any],
        casting: CastingKind = "unsafe",
        copy: bool = True,
    ) -> CompressedBase:
        dtype = np.dtype(dtype)
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
        cn.sin,
        cn.tan,
        cn.arcsin,
        cn.arctan,
        cn.sinh,
        cn.tanh,
        cn.arcsinh,
        cn.arctanh,
        cn.rint,
        cn.sign,
        cn.expm1,
        cn.log1p,
        cn.deg2rad,
        cn.rad2deg,
        cn.floor,
        cn.ceil,
        cn.trunc,
        cn.sqrt,
    ]
)

# Add the numpy unary ufuncs for which func(0) = 0 to _data_matrix.
for npfunc in _ufuncs_with_fixed_point_at_zero:
    name = npfunc.__name__

    def _create_method(op: Callable[[Any], Any]) -> Callable[[Any], Any]:
        def method(self: Any) -> Any:
            result = op(self.data)
            return self._with_data(result)

        method.__doc__ = (
            "Element-wise %s.\n\nSee `numpy.%s` for more information."
            % (name, name)
        )
        method.__name__ = name

        return method

    setattr(CompressedBase, name, _create_method(npfunc))


# unpack_rect1_store unpacks a rect1 store into two int64 stores.
def unpack_rect1_store(pos: LogicalStore) -> tuple[LogicalStore, LogicalStore]:
    """Unpack a rect1 store into two int64 stores.

    This function unpacks the compressed position array used in CSR/CSC
    formats into separate start and end position arrays.

    Parameters
    ----------
    pos : LogicalStore
        The rect1 store containing packed position information.

    Returns
    -------
    tuple
        (lo, hi) where lo contains start positions and hi contains end positions.
    """
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
def pack_to_rect1_store(
    lo: LogicalStore, hi: LogicalStore, output: LogicalStore | None = None
) -> LogicalStore:
    """Pack two int64 stores into a rect1 store.

    This function packs separate start and end position arrays into the
    compressed rect1 format used in CSR/CSC formats.

    Parameters
    ----------
    lo : LogicalStore
        Store containing start positions.
    hi : LogicalStore
        Store containing end positions.
    output : LogicalStore, optional
        Output store for the packed result. If None, creates a new store.

    Returns
    -------
    LogicalStore
        The packed rect1 store.
    """
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
