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
from legate.core import (
    ImageComputationHint,
    Scalar,
    Shape,
    align,
    broadcast,
    image,
    types,
)

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
    is_scalar_like,
    sort_by_rows_then_cols,
    store_from_store_or_array,
    store_to_cupynumeric_array,
)


@clone_scipy_arr_kind(scipy.sparse.csr_array)
class csr_array(CompressedBase, DenseSparseBase):
    """Compressed Sparse Row array.

    This can be instantiated in several ways:
        csr_array(D)
            where D is a 2-D ndarray or cupynumeric.ndarray

        csr_array(S)
            with another sparse array or matrix S (equivalent to S.tocsr())

        csr_array((M, N), [dtype])
            to construct an empty array with shape (M, N)
            dtype is optional, defaulting to dtype='d'.

        csr_array((data, (row_ind, col_ind)), [shape=(M, N)])
            where ``data``, ``row_ind`` and ``col_ind`` satisfy the
            relationship ``a[row_ind[k], col_ind[k]] = data[k]``.

        csr_array((data, indices, indptr), [shape=(M, N)])
            is the standard CSR representation where the column indices for
            row i are stored in ``indices[indptr[i]:indptr[i+1]]`` and their
            corresponding values are stored in ``data[indptr[i]:indptr[i+1]]``.
            If the shape parameter is not supplied, the array dimensions
            are inferred from the index arrays.

    Attributes
    ----------
    dtype : dtype
        Data type of the array
    shape : 2-tuple
        Shape of the array
    ndim : int
        Number of dimensions (this is always 2)
    nnz : int
        Number of stored values, including explicit zeros
    data : cupynumeric.ndarray
        CSR format data array of the array
    indices : cupynumeric.ndarray
        CSR format index array of the array
    indptr : cupynumeric.ndarray
        CSR format index pointer array of the array
    has_sorted_indices : bool
        Whether the indices are sorted
    has_canonical_format : bool
        Whether the matrix is in canonical format
    T : csr_array
        Transpose of the matrix

    Notes
    -----
    Sparse arrays can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Advantages of the CSR format:
        - fast matrix vector products

    Disadvantages of the CSR format:
        - changes to the sparsity structure are expensive (consider LIL or DOK)

    Canonical Format:
        - Within each row, indices are sorted by column.
        - There are no duplicate entries.

    Differences from SciPy:
        - Uses cupynumeric arrays instead of numpy arrays
        - GPU acceleration via cuSPARSE when available
        - Limited to supported datatypes on GPU: float32, float64, complex64, complex128
        - Some operations may create implicit copies due to transformed arrays
        - Element-wise operations with scalars only operate on existing non-zero elements
        - Indexing with boolean masks only updates existing non-zero elements

    Examples
    --------
    >>> import cupynumeric as np
    >>> from legate_sparse import csr_array
    >>> csr_array((3, 4), dtype=np.int8).todense()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> row = np.array([0, 0, 1, 2, 2, 2])
    >>> col = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> csr_array((data, (row, col)), shape=(3, 3)).todense()
    array([[1, 0, 2],
           [0, 0, 3],
           [4, 5, 6]])

    >>> indptr = np.array([0, 2, 3, 6])
    >>> indices = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> csr_array((data, indices, indptr), shape=(3, 3)).todense()
    array([[1, 0, 2],
           [0, 0, 3],
           [4, 5, 6]])
    """

    def __init__(self, arg, shape=None, dtype=None, copy=False):
        """Initialize a CSR array.

        Parameters
        ----------
        arg : array_like, tuple, or csr_array
            The input data. Can be:
            - A 2-D dense array (numpy.ndarray or cupynumeric.ndarray)
            - A sparse array/matrix to convert to CSR format
            - A tuple (M, N) for an empty array of shape (M, N)
            - A tuple (data, (row_ind, col_ind)) for COO format data
            - A tuple (data, indices, indptr) for CSR format data
        shape : tuple, optional
            Shape of the array (M, N). Required if not inferrable from input.
        dtype : dtype, optional
            Data type of the array. If None, inferred from input data.
            Defaults to float64 if not specified.
        copy : bool, optional
            Whether to copy the input data. Default is False.

        Raises
        ------
        NotImplementedError
            If the input type is not supported for conversion to CSR.
        AssertionError
            If shape cannot be inferred and is not provided.
        ValueError
            If input data is inconsistent or invalid.

        Notes
        -----
        When converting from dense arrays, the implementation uses a two-pass
        algorithm that first counts non-zeros per row, then fills them in.
        This may not scale well on distributed systems due to alignment constraints.

        When converting from COO format, the data is automatically sorted by
        rows and then by columns to ensure canonical format.
        """
        self.ndim = 2
        self.indices_sorted = False
        self.canonical_format = False
        super().__init__()

        # Note that cupynumeric.dtype(None) returns float64, so make
        # sure dtype is passed to csr_array if it is known apriori,
        # especially when copying the matrix
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
            dtype, shape = self._init_from_tuple_inputs(arg, dtype, shape, copy)
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

    def _init_from_tuple_inputs(self, arg, dtype, shape, copy):
        """Initialize CSR array from tuple inputs.

        This internal method handles the various tuple-based constructor formats:
        - (M, N) for empty arrays
        - (data, (row_ind, col_ind)) for COO format
        - (data, indices, indptr) for CSR format

        Parameters
        ----------
        arg : tuple
            The input tuple in one of the supported formats.
        dtype : dtype, optional
            The desired data type.
        shape : tuple, optional
            The shape of the array.
        copy : bool
            Whether to copy the input data.

        Returns
        -------
        tuple
            (dtype, shape) for the constructed array.

        Raises
        ------
        AssertionError
            If shape cannot be inferred or input is invalid.
        NotImplementedError
            If the tuple format is not supported.
        """

        def _get_empty_csr(dtype, nrows_plus_one):
            """Helper function to create empty CSR arrays."""
            return (
                cupynumeric.zeros(0, dtype=dtype),
                cupynumeric.zeros(0, dtype=coord_ty),
                cupynumeric.zeros(nrows_plus_one, dtype=coord_ty),
            )

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
                dtype = (
                    cupynumeric.float64 if dtype is None else cupynumeric.dtype(dtype)
                )

                # and pass this to next ctor
                arg = _get_empty_csr(dtype, M + 1)

            # Otherwise assume arg is COO data : (data, (row_ind, col_ind))
            else:
                if shape is None:
                    raise AssertionError("Cannot infer shape in this case.")

                st_data, (st_row, st_col) = arg

                # issue 209: handle the case where we have empty CSR array
                if st_data.size == st_row.size == st_col.size == 0:
                    arg = _get_empty_csr(dtype, shape[0] + 1)
                    copy = False
                else:
                    # if passed numpy arrays - convert them
                    if isinstance(st_row, numpy.ndarray):
                        st_row = cupynumeric.array(st_row)
                    if isinstance(st_col, numpy.ndarray):
                        st_col = cupynumeric.array(st_col)
                    if isinstance(st_data, numpy.ndarray):
                        st_data = cupynumeric.array(st_data)

                    if not self.indices_sorted:
                        # NOTE that CSR format does not require sorting the data
                        # by columns but in setitem, we assume that the data is
                        # sorted by rows and then by columns, so we sort the data
                        # by columns as well

                        row_array = array_from_store_or_array(st_row, copy=copy)
                        col_array = array_from_store_or_array(st_col, copy=copy)
                        new_data = array_from_store_or_array(st_data, copy=copy)

                        indices = sort_by_rows_then_cols(row_array, col_array)

                        new_data = new_data[indices]
                        row_array = row_array[indices]
                        col_array = col_array[indices]

                        row_offsets = cupynumeric.append(
                            cupynumeric.array([0]),
                            cupynumeric.cumsum(
                                cupynumeric.bincount(row_array, minlength=shape[0])
                            ),
                        )

                        # pass to next ctor
                        arg = (new_data, col_array, row_offsets)

                        self.indices_sorted = True
                        self.canonical_format = True
                    else:
                        # we need to convert row indices to row offsets/indptr
                        row_array = array_from_store_or_array(st_row)
                        row_offsets = cupynumeric.append(
                            cupynumeric.array([0]),
                            cupynumeric.cumsum(
                                cupynumeric.bincount(row_array, minlength=shape[0])
                            ),
                        )
                        if copy:
                            arg = (st_data.copy(), st_col.copy(), row_offsets)
                        else:
                            arg = (st_data, st_col, row_offsets)

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
                self.crd = store_from_store_or_array(cast_arr(indices, coord_ty), copy)
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

        return dtype, shape

    @property
    def dim(self):
        """Number of dimensions (always 2 for CSR arrays)."""
        return self.ndim

    @property
    def nnz(self):
        """Number of stored values, including explicit zeros.

        Returns
        -------
        int
            The number of non-zero elements in the matrix.
        """
        return self.vals.shape[0]

    @property
    def dtype(self):
        """Data type of the array.

        Returns
        -------
        dtype
            The data type of the array elements.
        """
        # We can just return self.vals.type, but bookkeep type separately now
        return self._dtype

    # Enable direct operation on the values array.
    def get_data(self):
        """Get the data array of the CSR matrix.

        Returns
        -------
        cupynumeric.ndarray
            The data array containing the non-zero values.
        """
        return store_to_cupynumeric_array(self.vals)

    # From array,
    def set_data(self, data):
        """Set the data array of the CSR matrix.

        Parameters
        ----------
        data : cupynumeric.ndarray
            The new data array. Must have the same length as the current data array.

        Raises
        ------
        AssertionError
            If data is not a cupynumeric.ndarray.
        """
        if isinstance(data, numpy.ndarray):
            data = cupynumeric.array(data)
        assert isinstance(data, cupynumeric.ndarray)
        self.vals = get_store_from_cupynumeric_array(data)
        self._dtype = data.dtype

    data = property(
        fget=get_data, fset=set_data, doc="CSR format data array of the matrix"
    )

    # Enable direct operation on the indices array.
    def get_indices(self):
        """Get the column indices array of the CSR matrix.

        Returns
        -------
        cupynumeric.ndarray
            The column indices array.
        """
        return store_to_cupynumeric_array(self.crd)

    def set_indices(self, indices):
        """Set the column indices array of the CSR matrix.

        Parameters
        ----------
        indices : cupynumeric.ndarray
            The new column indices array. Must have the same length as the current indices array.

        Raises
        ------
        AssertionError
            If indices is not a cupynumeric.ndarray.

        Notes
        -----
        Setting new indices will mark the matrix as not having sorted indices
        and not being in canonical format.
        """
        if isinstance(indices, numpy.ndarray):
            indices = cupynumeric.array(indices)
        assert isinstance(indices, cupynumeric.ndarray)
        self.crd = get_store_from_cupynumeric_array(indices)
        # we can't guarantee new indices are sorted
        self.canonical_format = False
        self.indices_sorted = False

    indices = property(
        fget=get_indices, fset=set_indices, doc="CSR format index array of the matrix"
    )

    def get_indptr(self):
        """Get the index pointer array of the CSR matrix.

        Returns
        -------
        cupynumeric.ndarray
            The index pointer array. For row i, the column indices are stored in
            indices[indptr[i]:indptr[i+1]] and their corresponding values are
            stored in data[indptr[i]:indptr[i+1]].
        """
        row_start_st, row_end_st = unpack_rect1_store(self.pos)
        row_start = store_to_cupynumeric_array(row_start_st)
        return cupynumeric.append(row_start, [self.nnz])

    # Disallow changing intptrs directly
    indptr = property(
        fget=get_indptr, doc="CSR format index pointer array of the matrix"
    )

    def _get_row_indices(self):
        """Helper routine that converts pos to row indices.

        This internal method expands the compressed row storage format's position
        array into explicit row indices for each non-zero element.

        Returns
        -------
        cupynumeric.ndarray
            Array of row indices corresponding to each non-zero element.

        Notes
        -----
        This method is used internally by comparison operations and other
        methods that need explicit row indices. The result could be cached
        for performance, but currently is recomputed each time.
        """
        row_indices = runtime.create_store(coord_ty, shape=self.crd.shape)
        task = runtime.create_auto_task(SparseOpCode.EXPAND_POS_TO_COORDINATES)
        src_part = task.add_input(self.pos)
        dst_part = task.add_output(row_indices)
        task.add_constraint(image(src_part, dst_part))

        task.execute()
        return store_to_cupynumeric_array(row_indices)

    def has_sorted_indices(self):
        """Determine whether the matrix has sorted indices.

        Returns
        -------
        bool
            True if the indices are sorted, False otherwise.
        """
        return self.indices_sorted

    def has_canonical_format(self):
        """Determine whether the matrix is in canonical format.

        Returns
        -------
        bool
            True if the matrix is in canonical format, False otherwise.

        Notes
        -----
        A matrix is in canonical format if:
        - Within each row, indices are sorted by column
        - There are no duplicate entries
        """
        return self.canonical_format

    # The rest of the methods
    def diagonal(self, k=0):
        """Return the k-th diagonal of the matrix.

        Parameters
        ----------
        k : int, optional
            Which diagonal to retrieve. Default is 0 (main diagonal).
            k > 0 for upper diagonals, k < 0 for lower diagonals.

        Returns
        -------
        cupynumeric.ndarray
            The k-th diagonal of the matrix.

        Raises
        ------
        NotImplementedError
            If k != 0 (only main diagonal is currently supported).

        Notes
        -----
        Currently only supports k=0 (main diagonal). Other diagonals
        are not implemented.
        """
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
        """Return a dense matrix representation of this matrix.

        Parameters
        ----------
        order : str, optional
            Not supported. Must be None.
        out : cupynumeric.ndarray, optional
            Output array for the result. Must have the same shape and dtype
            as the expected output.

        Returns
        -------
        cupynumeric.ndarray
            A dense matrix with the same shape and dtype as this matrix.

        Raises
        ------
        NotImplementedError
            If order is not None.
        ValueError
            If out is provided but has incompatible dtype or shape.

        Notes
        -----
        The order parameter is not supported and must be None.
        If out is provided, it must have the correct shape and dtype.
        """
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
        """Point-wise multiplication by another matrix, vector, or scalar.

        Parameters
        ----------
        other : csr_array, cupynumeric.ndarray, or scalar
            The object to multiply with.

        Returns
        -------
        csr_array or cupynumeric.ndarray
            The result of the multiplication.

        Notes
        -----
        This is equivalent to the * operator.
        """
        return self * other

    def __rmul__(self, other):
        """Right multiplication by a scalar.

        Parameters
        ----------
        other : scalar
            The scalar to multiply with.

        Returns
        -------
        csr_array
            The result of the multiplication.
        """
        return self * other

    # This is an element-wise operation now.
    def __mul__(self, other):
        """Element-wise multiplication.

        Parameters
        ----------
        other : scalar or array_like
            The object to multiply with.

        Returns
        -------
        csr_array
            The result of the multiplication.

        Raises
        ------
        NotImplementedError
            If other is not a scalar.

        Notes
        -----
        Currently only supports scalar multiplication. Array multiplication
        is not implemented.
        """
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
        """Right matrix multiplication (other @ self).

        Parameters
        ----------
        other : array_like
            The left operand for matrix multiplication.

        Returns
        -------
        cupynumeric.ndarray or csr_array
            The result of the matrix multiplication.

        Raises
        ------
        NotImplementedError
            Currently not implemented.

        Notes
        -----
        This method handles the case where a dense matrix is multiplied
        with a CSR matrix from the left. Currently not implemented.
        """
        # Handle dense @ CSR
        raise NotImplementedError

    def __matmul__(self, other):
        """Matrix multiplication (self @ other).

        Parameters
        ----------
        other : array_like or csr_array
            The right operand for matrix multiplication.

        Returns
        -------
        cupynumeric.ndarray or csr_array
            The result of the matrix multiplication.

        Notes
        -----
        This is equivalent to the dot method.
        """
        return self.dot(other)

    def _compare_scalar(self, other, op):
        """Helper method for element-wise comparison operations with scalars.
        This methods returns a boolean CSR array with True values where
        the comparison for op returns True.

        Parameters
        ----------
        other : scalar
            The scalar value to compare against
        op : callable
            The comparison operator to use (e.g. cupynumeric.greater)

        Returns
        -------
        csr_array
            A boolean CSR array with True values where the comparison is True
        """
        assert is_scalar_like(other)
        mask = op(store_to_cupynumeric_array(self.vals), other)
        col_indices = store_to_cupynumeric_array(self.crd)[mask]
        row_indices = self._get_row_indices()[mask]
        vals = cupynumeric.ones(row_indices.size, dtype=bool)

        # NOTE:
        # If the data was already sorted by rows and cols in self,
        # then we don't have to sort again in the constructor of csr_array,
        # but there's no clean way to pass to the class that the data
        # is already sorted
        return csr_array(
            (vals, (row_indices, col_indices)),
            shape=self.shape,
            dtype=bool,
        )

    def __gt__(self, other):
        """Element-wise greater than comparison with a scalar value.
        This operates only on the existing non-zero elements of the matrix.

        Parameters
        ----------
        other : scalar
            The scalar value to compare against.

        Returns
        -------
        csr_array
            A boolean CSR array with True values where elements are greater
            than the scalar.

        Raises
        ------
        AssertionError
            If the input is not scalar-like.

        Examples
        --------
        >>> A = csr_array(...)
        >>> mask = A > 0.5  # Returns boolean CSR array
        """
        return self._compare_scalar(other, cupynumeric.greater)

    def __lt__(self, other):
        """Element-wise less than comparison with a scalar value.
        This operates only on the existing non-zero elements of the matrix.

        Parameters
        ----------
        other : scalar
            The scalar value to compare against.

        Returns
        -------
        csr_array
            A boolean CSR array with True values where elements are less
            than the scalar.

        Raises
        ------
        AssertionError
            If the input is not scalar-like.

        Examples
        --------
        >>> A = csr_array(...)
        >>> mask = A < 0.5  # Returns boolean CSR array
        """
        return self._compare_scalar(other, cupynumeric.less)

    def __ge__(self, other):
        """Element-wise greater than or equal comparison with a scalar value.
        This operates only on the existing non-zero elements of the matrix.

        Parameters
        ----------
        other : scalar
            The scalar value to compare against.

        Returns
        -------
        csr_array
            A boolean CSR array with True values where elements are greater
            than or equal to the scalar.

        Raises
        ------
        AssertionError
            If the input is not scalar-like.

        Examples
        --------
        >>> A = csr_array(...)
        >>> mask = A >= 0.5  # Returns boolean CSR array
        """
        return self._compare_scalar(other, cupynumeric.greater_equal)

    def __le__(self, other):
        """Element-wise less than or equal comparison with a scalar value.
        This operates only on the existing non-zero elements of the matrix.

        Parameters
        ----------
        other : scalar
            The scalar value to compare against.

        Returns
        -------
        csr_array
            A boolean CSR array with True values where elements are less
            than or equal to the scalar.

        Raises
        ------
        AssertionError
            If the input is not a scalar or a zero-dimensional array.

        Examples
        --------
        >>> A = csr_array(...)
        >>> mask = A <= 0.5  # Returns boolean CSR array
        """
        return self._compare_scalar(other, cupynumeric.less_equal)

    def __eq__(self, other):
        """Element-wise equality comparison with a scalar value.
        This operates only on the existing non-zero elements of the matrix.

        Parameters
        ----------
        other : scalar
            The scalar value to compare against.

        Returns
        -------
        csr_array
            A boolean CSR array with True values where elements are equal
            to the scalar.

        Raises
        ------
        AssertionError
            If the input is not scalar-like.

        Examples
        --------
        >>> A = csr_array(...)
        >>> mask = A == 0.5  # Returns boolean CSR array
        """
        return self._compare_scalar(other, cupynumeric.equal)

    def __ne__(self, other):
        """Element-wise not equal comparison with a scalar value.
        This operates only on the existing non-zero elements of the matrix.

        Parameters
        ----------
        other : scalar
            The scalar value to compare against.

        Returns
        -------
        csr_array
            A boolean CSR array with True values where elements are not equal
            to the scalar.

        Raises
        ------
        AssertionError
            If the input is not scalar-like.

        Examples
        --------
        >>> A = csr_array(...)
        >>> mask = A != 0.5  # Returns boolean CSR array
        """
        return self._compare_scalar(other, cupynumeric.not_equal)

    def __setitem__(self, key, value):
        """Set values in the matrix using a boolean CSR mask.

        Parameters
        ----------
        key : csr_array or csr_matrix
            A boolean CSR matrix of the same shape as self that indicates which
            elements to modify. Must have dtype=bool and same shape as the matrix
        value : scalar
            Value to assign at the positions indicated by key. Value gets
            converted to the datatype of CSR matrix before assignment.

        Returns
        -------
        csr_array
            The modified matrix (self).

        Raises
        ------
        NotImplementedError
            If key is not a CSR matrix.

        Examples
        --------
        >>> A = csr_array([[1, 2, 0], [3, 0, 4]])
        >>> mask = A > 2  # Create mask from A
        >>> A[mask] = 10
        >>> A.todense()
        array([[ 1,  2,  0],
               [10,  0, 10]])

        Notes
        -----
        This operation only updates entries that are
        non-zero in both the original matrix and the mask. Elements that are zero
        in the original matrix will remain zero even if they are True in the mask.
        """
        allowed_types = (csr_matrix, csr_array)
        if not isinstance(key, allowed_types):
            msg = "setting item is only supported for bool csr matrices"
            raise NotImplementedError(msg)

        assert key.shape == self.shape
        assert key.dtype == bool

        value_store = runtime.legate_runtime.create_store_from_scalar(Scalar(value))

        # launch c++ task
        task = runtime.create_auto_task(SparseOpCode.CSR_INDEXING_CSR)
        A_vals_part = task.add_output(self.vals)
        A_pos_part = task.add_input(self.pos)
        A_crd_part = task.add_input(self.crd)
        mask_pos_part = task.add_input(key.pos)
        mask_crd_part = task.add_input(key.crd)
        task.add_input(value_store)

        # The elements that get updated are the ones where the mask
        # and the current matrix have a non-zero value, so the coordinates
        # that get updated in this operation is same as that from
        # an AND operation of the coordinates of mask and self/matrix

        # add partitioning constraints
        task.add_constraint(image(A_pos_part, A_crd_part))
        task.add_constraint(image(A_pos_part, A_vals_part))
        task.add_constraint(image(mask_pos_part, mask_crd_part))
        task.add_constraint(align(A_pos_part, mask_pos_part))

        task.execute()

        return self

    def dot(self, other, out=None):
        """Ordinary dot product.

        Parameters
        ----------
        other : array_like or csr_array
            The object to compute dot product with. Can be:
            - A dense vector (1-D array) for sparse matrix-vector multiplication (SpMV)
            - A dense matrix (2-D array) for sparse matrix-matrix multiplication (SpMM)
            - A CSR matrix for sparse matrix-sparse matrix multiplication (SpGEMM)
        out : cupynumeric.ndarray, optional
            Output array for the result. Only supported for SpMV operations.
            Must have the correct shape and dtype.

        Returns
        -------
        cupynumeric.ndarray or csr_array
            The result of the dot product:
            - For SpMV: dense vector
            - For SpMM: dense matrix
            - For SpGEMM: CSR matrix

        Raises
        ------
        NotImplementedError
            If the operation is not supported or datatypes are not supported on GPU.
        ValueError
            If out is provided for SpGEMM operations or has incompatible dtype/shape.
        RuntimeWarning
            If an implicit copy is created due to transformed input arrays.

        Notes
        -----
        Supported operations:
        - SpMV (sparse matrix-vector): A @ x where x is a dense vector
        - SpGEMM (sparse-sparse): A @ B where B is a CSR matrix

        GPU limitations:
        - Only floating point datatypes are supported: float32, float64, complex64, complex128
        - Some operations may create implicit copies due to transformed arrays

        The implementation automatically chooses the appropriate algorithm:
        - For vectors: uses cuSPARSE SpMV when available
        - For CSR matrices: uses cuSPARSE SpGEMM on GPU, custom implementation on CPU

        Examples
        --------
        >>> import cupynumeric as np
        >>> from legate_sparse import csr_array
        >>> A = csr_array([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
        >>> v = np.array([1, 0, -1])
        >>> A.dot(v)
        array([ 1, -3, -1])
        """
        # If output specified - it should be cupynumeric array
        if out is not None:
            assert isinstance(out, cupynumeric.ndarray)

        # only floating point operations are supported by cusparse at the moment
        if runtime.num_gpus > 0:
            if not is_dtype_supported(self.dtype) or not is_dtype_supported(
                other.dtype
            ):
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
        """Helper method to get row start and end positions.

        This internal method unpacks the compressed row storage format's position array
        into start and end positions for each row.

        Returns
        -------
        list of tuple
            List of (start, end) position tuples for each row in the matrix.
            For row i, the non-zero elements are stored in positions
            [start, end) in the data and indices arrays.
        """
        row_start_st, row_end_st = unpack_rect1_store(self.pos)
        row_start = store_to_cupynumeric_array(row_start_st)
        row_end = store_to_cupynumeric_array(row_end_st)
        return [(i, j) for (i, j) in zip(row_start, row_end)]

    def copy(self):
        """Returns a copy of this matrix.

        Returns
        -------
        csr_array
            A copy of the matrix with the same data and structure.
        """
        return csr_array(self, dtype=self.dtype)

    def conj(self, copy=True):
        """Element-wise complex conjugate.

        Parameters
        ----------
        copy : bool, optional
            Whether to create a new matrix or modify in-place. Default is True.

        Returns
        -------
        csr_array
            The conjugate matrix.

        Notes
        -----
        If copy=True, returns a new matrix. If copy=False, modifies the
        current matrix in-place.
        """
        if copy:
            return self.copy().conj(copy=False)
        return self._with_data(
            get_store_from_cupynumeric_array(self.data.conj()), copy=False
        )

    def transpose(self, axes=None, copy=False):
        """Reverses the dimensions of the sparse matrix.

        Parameters
        ----------
        axes : None, optional
            This argument is not supported and must be None.
        copy : bool, optional
            Whether to create a copy. Ignored - CSR transpose always creates a copy.

        Returns
        -------
        csr_array
            Transposed matrix with shape (N, M) where the original shape was (M, N).

        Raises
        ------
        AssertionError
            If axes is not None.

        Notes
        -----
        The axes parameter is not supported and must be None.
        CSR transpose always creates a copy due to the format conversion.
        The implementation sorts the data by columns to maintain canonical format.
        """
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

    T = property(transpose, doc="Transpose of the matrix")

    def asformat(self, format, copy=False):
        """Convert this matrix to a specified format.

        Parameters
        ----------
        format : str
            Desired sparse format. Currently only 'csr' is supported.
        copy : bool, optional
            Whether to create a copy. Default is False.

        Returns
        -------
        csr_array
            Matrix in the requested format.

        Raises
        ------
        NotImplementedError
            If format is not 'csr'.

        Notes
        -----
        Currently only CSR format is supported. Other formats are not implemented.
        """
        if format == "csr":
            return self.copy() if copy else self
        else:
            raise NotImplementedError("Only CSR format is supported right now")

    def tocsr(self, copy=False):
        """Convert this matrix to a CSR matrix.

        Parameters
        ----------
        copy : bool, optional
            Whether to create a copy. Default is False.

        Returns
        -------
        csr_array
            The converted CSR matrix.

        Notes
        -----
        Since this matrix is already in CSR format, this method simply
        returns a copy if requested, or the matrix itself otherwise.
        """
        if copy:
            return self.copy().tocsr(copy=False)
        return self

    def nonzero(self):
        """Return the indices of the non-zero elements.

        Returns
        -------
        (row, col) : tuple of cupynumeric.ndarrays
            Row and column indices of non-zeros. Only returns indices
            where the values are actually non-zero (not just stored).

        Notes
        -----
        This method filters out explicit zeros that may be stored in the
        sparse matrix structure.
        """
        task = runtime.create_auto_task(SparseOpCode.EXPAND_POS_TO_COORDINATES)

        row_indices = runtime.create_store(coord_ty, shape=self.crd.shape)
        row_indices_part = task.add_output(row_indices)
        pos_part = task.add_input(self.pos)
        task.add_constraint(image(pos_part, row_indices_part))
        task.execute()

        row_indices = store_to_cupynumeric_array(row_indices)
        col_indices = store_to_cupynumeric_array(self.crd)
        vals_array = store_to_cupynumeric_array(self.vals)
        mask = vals_array != 0.0

        return (row_indices[mask], col_indices[mask])


csr_matrix = csr_array
"""Alias for csr_array for backward compatibility with SciPy naming conventions."""


# spmv computes y = A @ x.
def spmv(A: csr_array, x: cupynumeric.ndarray, y: cupynumeric.ndarray):
    """Perform sparse matrix vector product y = A @ x.

    Parameters
    ----------
    A : csr_array
        Input sparse matrix of shape (M, N).
    x : cupynumeric.ndarray
        Dense vector of shape (N,) for the dot product.
    y : cupynumeric.ndarray
        Output array of shape (M,) to store the result.

    Notes
    -----
    This function computes the sparse matrix-vector multiplication y = A @ x.
    The implementation uses an auto-parallelized kernel that distributes
    the computation across available processors.

    The function modifies y in-place to store the result.
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
    """Perform sparse matrix multiplication C = A @ B.

    Parameters
    ----------
    A : csr_array
        Input sparse matrix A of shape (M, K).
    B : csr_array
        Input sparse matrix B of shape (K, N).

    Returns
    -------
    csr_array
        The result of the sparse matrix multiplication with shape (M, N).

    Notes
    -----
    This function computes the sparse matrix-sparse matrix multiplication C = A @ B.

    The implementation differs based on the available hardware:
    - On GPU: Uses cuSPARSE SpGEMM with local CSR matrices that are aggregated
    - On CPU: Uses a custom implementation with two-pass algorithm

    The GPU implementation creates a set of local CSR matrices that are
    aggregated into a global CSR matrix. The CPU implementation uses a
    query phase to determine the number of non-zeros per row, followed
    by the actual computation phase.

    Both implementations maintain the CSR format throughout the computation.
    """
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
