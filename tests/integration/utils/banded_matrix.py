# Copyright 2023-2024 NVIDIA Corporation
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

import cupynumeric as np

import legate_sparse as sparse


def banded_matrix(
    N: int,
    nnz_per_row: int,
    from_diags: bool = True,
    init_with_ones: bool = True,
    verbose: bool = False,
):
    """Create a banded sparse matrix for testing purposes.

    Parameters
    ----------
    N : int
        Size of the NxN sparse matrix.
    nnz_per_row : int
        Number of non-zero elements per row (must be odd).
    from_diags : bool, optional
        Use sparse.diags to generate the banded matrix. Default is True.
    init_with_ones : bool, optional
        Initialize the matrix with ones instead of arange. Default is True.
    verbose : bool, optional
        Print detailed information about the matrix construction. Default is False.

    Returns
    -------
    csr_array
        A banded sparse matrix in CSR format.

    Raises
    ------
    AssertionError
        If N <= nnz_per_row or nnz_per_row is not odd.

    Notes
    -----
    This function creates a banded matrix with a specific sparsity pattern.
    When from_diags=True, it uses the sparse.diags function which is simpler
    but may be slower. When from_diags=False, it constructs the CSR matrix
    directly for better performance.

    The matrix has a banded structure with nnz_per_row non-zeros per row,
    centered around the main diagonal. The function handles the boundary
    conditions by masking out-of-bounds indices.

    Examples
    --------
    >>> A = banded_matrix(5, 3, from_diags=True)
    >>> print(A.toarray())
    [[1. 1. 0. 0. 0.]
     [1. 1. 1. 0. 0.]
     [0. 1. 1. 1. 0.]
     [0. 0. 1. 1. 1.]
     [0. 0. 0. 1. 1.]]
    """

    if from_diags:
        return sparse.diags(
            np.array([1] * nnz_per_row),
            np.array([x - (nnz_per_row // 2) for x in range(nnz_per_row)]),
            shape=(N, N),
            format="csr",
            dtype=np.float64,
        )
    else:
        assert N > nnz_per_row
        assert nnz_per_row % 2 == 1
        half_nnz = nnz_per_row // 2

        pred_nrows = nnz_per_row - half_nnz
        post_nrows = pred_nrows
        main_rows = N - pred_nrows - post_nrows

        pred = np.arange(nnz_per_row - half_nnz, nnz_per_row + 1)
        post = np.flip(pred)
        nnz_arr = np.concatenate(
            (pred, np.ones(main_rows) * nnz_per_row, post)
        )

        if sparse.__name__ == "legate_sparse":
            row_offsets = np.zeros(N + 1).astype(sparse.coord_ty)
        else:
            row_offsets = np.zeros(N + 1).astype(int)

        row_offsets[1 : N + 1] = np.cumsum(nnz_arr)
        nnz = row_offsets[-1]

        col_indices = np.tile(
            np.arange(-half_nnz, nnz_per_row - half_nnz), (N,)
        ) + np.repeat(np.arange(N), nnz_per_row)

        if init_with_ones:
            data = np.ones(N * nnz_per_row).astype(np.float64)
        else:
            data = np.arange(N * nnz_per_row).astype(np.float64) / N

        mask = col_indices >= 0
        mask &= col_indices < N

        col_indices = col_indices[mask]
        data = data[mask]
        assert data.shape[0] == nnz
        assert col_indices.shape[0] == nnz

        if verbose:
            np.set_printoptions(linewidth=1000)
            print(f"data       : {data}")
            print(f"col_indices: {col_indices}")
            print(f"row_offsets: {row_offsets}")

        return sparse.csr_array(
            (data, col_indices.astype(np.int64), row_offsets.astype(np.int64)),
            shape=(N, N),
            copy=False,
        )
