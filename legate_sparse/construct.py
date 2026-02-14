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

import cupynumeric as cn

from .csr import csr_array


def _block(blocks, format="csr", dtype=None):
    """Build a sparse CSR array from sparse sub-blocks using COO intermediate.

    1. Extracts (row, col, data) from each block
    2. Adjusts indices by block offsets
    3. Concatenates all coordinates
    4. Builds CSR from COO format
    """
    if format != "csr":
        raise ValueError("Only 'csr' format is supported for block_array")

    if not isinstance(blocks, (list, tuple)):
        blocks = list(blocks)

    blocks = [
        list(row) if isinstance(row, (list, tuple)) else [row]
        for row in blocks
    ]

    n_block_rows = len(blocks)
    if n_block_rows == 0:
        raise ValueError("blocks cannot be empty")

    n_block_cols = len(blocks[0])
    if n_block_cols == 0:
        raise ValueError("blocks cannot be empty")

    # Row height and col width for a sub-block looks like this.
    # +--------------+
    # | ^            |
    # | | row height |
    # | v            |
    # +--------------+
    # <- col width ->

    # store row heights and col widths of each sub-block
    row_heights = [None] * n_block_rows
    col_widths = [None] * n_block_cols

    for i in range(n_block_rows):
        for j in range(n_block_cols):
            block = blocks[i][j]
            if block is None:
                continue

            if not isinstance(block, csr_array):
                raise TypeError(
                    f"blocks[{i}][{j}] must be a csr_array or None, "
                    f"got {type(block).__name__}"
                )

            block_nrows, block_ncols = block.shape

            # Check/set row height for this block row.
            # The row heights of all the sub-blocks in a row of the input
            # should be the same, else we can't concatenate horizontally
            if row_heights[i] is None:
                row_heights[i] = block_nrows
            elif row_heights[i] != block_nrows:
                raise ValueError(
                    f"blocks[{i}][{j}] has {block_nrows} rows, "
                    f"expected {row_heights[i]}"
                )

            # Check/set column width for this block column.
            # The col widths of all the sub-blocks in a col of the input
            # should be the same, else we can't concatenate vertically
            if col_widths[j] is None:
                col_widths[j] = block_ncols
            elif col_widths[j] != block_ncols:
                raise ValueError(
                    f"blocks[{i}][{j}] has {block_ncols} columns, "
                    f"expected {col_widths[j]}"
                )

    # The input can have None instead of a csr matrix. To correctly compute
    # the row offsets for those cases, we set the row height to 0 if the
    # input is None.
    row_heights = cn.array([h if h is not None else 0 for h in row_heights])
    col_widths = cn.array([w if w is not None else 0 for w in col_widths])

    # Compute the no. or rows and cols in the output matrix.
    total_nrows = cn.sum(row_heights).item()
    total_ncols = cn.sum(col_widths).item()

    # When the output matrix is empty, we don't need to concatenate.
    if total_nrows == 0 or total_ncols == 0:
        result_dtype = dtype if dtype is not None else cn.float64
        return csr_array((total_nrows, total_ncols), dtype=result_dtype)

    row_offsets = cn.concatenate([cn.array([0]), cn.cumsum(row_heights)])
    col_offsets = cn.concatenate([cn.array([0]), cn.cumsum(col_widths)])

    if dtype is None:
        dtypes = [b.dtype for row in blocks for b in row if b is not None]
        dtype = cn.result_type(*dtypes) if dtypes else cn.float64

    all_rows = []
    all_cols = []
    all_data = []

    # Populate the concatenated (rows, cols, data) arrays for the
    # output matrix. The outer loop concatenates the sub-blocks vertically
    # while the inner loop concatenates them horizontally. This is done
    # without creating any intermediate csr representation.
    for i in range(n_block_rows):
        row_offset = row_offsets[i].item()

        for j in range(n_block_cols):
            block = blocks[i][j]

            # If block is empty, the (rows, cols, data) of the output matrix
            # doesn't get modified, so we continue with the loop.
            if block is None:
                continue

            col_offset = col_offsets[j].item()
            block_nrows = block.shape[0]

            indptr = block.indptr
            indices = block.indices
            data = block.data

            # Empty csr matrices don't modify the output matrix either, so we
            # continue with the loop.
            if data.size == 0:
                continue

            # Expand the indptr array to store the row indices.
            # For each row r, repeating r by (indptr[r+1] - indptr[r]) times
            # the needed storage to store non-zero entries.
            nnz_per_row = cn.diff(indptr)
            block_rows = cn.repeat(cn.arange(block_nrows), nnz_per_row)

            # After concatenating the matrices, we get one block matrix that
            # can be represented by (rows, cols, data) arrays. Note that
            # we have to add the offsets for both the row and col indices
            # that correspond to the non-zero in the previous sub-block as
            # concatenate them horizontally. This is because the output matrix
            # is going to be represented as one giant CSR matrix.
            all_rows.append(block_rows + row_offset)
            all_cols.append(indices + col_offset)
            all_data.append(data)

    if not all_data:
        result_dtype = dtype if dtype is not None else cn.float64
        return csr_array((total_nrows, total_ncols), dtype=result_dtype)

    concatenated_rows = cn.concatenate(all_rows)
    concatenated_cols = cn.concatenate(all_cols)
    concatenated_data = cn.concatenate(all_data).astype(dtype)

    return csr_array(
        (concatenated_data, (concatenated_rows, concatenated_cols)),
        shape=(total_nrows, total_ncols),
        dtype=dtype,
    )


def block_array(blocks, format="csr", dtype=None):
    """Build a sparse array from sparse sub-blocks.

    Parameters
    ----------
    blocks : array_like
        A 2-D array-like of shape (M, N) where each element is a sparse
        CSR array or None. None elements are treated as zero matrices.
    format : str, optional
        Output format. Currently only 'csr' is supported. Default is 'csr'.
    dtype : dtype, optional
        Data type of the output array. If None, inferred from the blocks.

    Returns
    -------
    csr_array
        A sparse CSR array formed by combining the sub-blocks.

    Raises
    ------
    ValueError
        - If `format` is not 'csr'.
        - If `blocks` is empty (has zero rows or zero columns).
        - If sub-blocks in the same row have different numbers of rows.
        - If sub-blocks in the same column have different numbers of columns.
    TypeError
        - If any non-None block is not a csr_array.

    Notes
    -----
    This function may not be performant when the number of sub-blocks is large,
    as it iterates over all blocks sequentially to extract and concatenate their
    COO coordinates.

    Examples
    --------
    >>> import legate_sparse as sparse
    >>> A = sparse.csr_array([[1, 2], [3, 4]])
    >>> B = sparse.csr_array([[5], [6]])
    >>> C = sparse.csr_array([[7, 8, 9]])
    >>> result = sparse.block_array([[A, B], [C, None]])
    >>> result.todense()
    array([[1, 2, 5],
           [3, 4, 6],
           [7, 8, 9]])
    """
    return _block(blocks, format, dtype)
