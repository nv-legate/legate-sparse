import numpy
import pytest
from scipy import sparse as scipy_sparse
from utils.sample import simple_system_gen

import legate_sparse as sparse


@pytest.fixture
def create_mask():
    """Create a boolean mask matrix with a random sparsity pattern.

    This fixture creates equivalent boolean mask matrices in both SciPy and
    Legate Sparse formats for testing purposes.

    Parameters
    ----------
    rows : int
        Number of rows (and columns) in the square matrix.
    density : float, optional
        Density of non-zero elements. Default is 0.3.

    Returns
    -------
    tuple
        (A_scipy, A_sparse) - Equivalent boolean matrices in SciPy and
        Legate Sparse formats.

    Notes
    -----
    The fixture ensures that both matrices have identical sparsity patterns
    and values. It verifies equivalence by converting both to dense format
    and checking that they are numerically close.

    """

    def _create_mask(rows, density=0.3):
        cols = rows
        nnz = int(rows * cols * density)

        # SciPy
        row_idx = numpy.random.randint(0, rows, size=nnz)
        col_idx = numpy.random.randint(0, cols, size=nnz)
        data = numpy.ones(nnz, dtype=bool)
        A_scipy = scipy_sparse.csr_array((data, (row_idx, col_idx)), shape=(rows, cols))

        # Sparse
        A_sparse = sparse.csr_array(A_scipy.todense())

        # Verify matrices are equivalent
        A_scipy_dense = numpy.asarray(A_scipy.todense())
        A_sparse_dense = numpy.asarray(A_sparse.todense())
        assert numpy.all(
            numpy.allclose(A_scipy_dense, A_sparse_dense, rtol=1e-5, atol=1e-6)
        )

        return A_scipy, A_sparse

    return _create_mask


@pytest.fixture
def create_matrix():
    """Create matrices in SciPy and Legate Sparse that are equivalent.

    This fixture creates equivalent sparse matrices in both SciPy and
    Legate Sparse formats for testing purposes.

    Parameters
    ----------
    N : int
        Number of rows (and columns) in the square matrix.
    tol : float, optional
        Threshold for sparsity. Values below this threshold are set to zero.
        Default is 0.5.

    Returns
    -------
    tuple
        (A_scipy, A_sparse) - Equivalent sparse matrices in SciPy and
        Legate Sparse formats.

    Notes
    -----
    The fixture uses simple_system_gen to create a dense matrix, then
    converts it to sparse format in both libraries. It verifies equivalence
    by converting both to dense format and checking that they are numerically
    close.

    """

    def _create_matrix(N, tol=0.5):
        _, A_scipy, _ = simple_system_gen(N, N, scipy_sparse.csr_array, tol=tol)
        A_sparse = sparse.csr_array(A_scipy)

        # Verify matrices are equivalent
        A_scipy_dense = numpy.asarray(A_scipy.todense())
        A_sparse_dense = numpy.asarray(A_sparse.todense())
        assert numpy.all(
            numpy.allclose(A_scipy_dense, A_sparse_dense, rtol=1e-5, atol=1e-6)
        )

        return A_scipy, A_sparse

    return _create_matrix
