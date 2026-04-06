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
        A_scipy = scipy_sparse.csr_array(
            (data, (row_idx, col_idx)), shape=(rows, cols)
        )

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
        _, A_scipy, _ = simple_system_gen(
            N, N, scipy_sparse.csr_array, tol=tol
        )
        A_sparse = sparse.csr_array(A_scipy)

        # Verify matrices are equivalent
        A_scipy_dense = numpy.asarray(A_scipy.todense())
        A_sparse_dense = numpy.asarray(A_sparse.todense())
        assert numpy.all(
            numpy.allclose(A_scipy_dense, A_sparse_dense, rtol=1e-5, atol=1e-6)
        )

        return A_scipy, A_sparse

    return _create_matrix


@pytest.fixture
def create_tridiagonal_complex_hermitian_matrix():
    """Create a tridiagonal complex Hermitian sparse matrix.

    This fixture creates a tridiagonal complex Hermitian sparse matrix suitable
    for eigenvalue computations. The matrix has a real main diagonal and complex
    conjugate off-diagonals.

    Parameters
    ----------
    N : int
        Number of rows (and columns) in the square matrix.

    Returns
    -------
    scipy.sparse.csr_array
        A tridiagonal complex Hermitian sparse matrix in SciPy CSR format.

    Notes
    -----
    The matrix is constructed with:
    - Main diagonal: 4.0
    - Upper diagonal: -(1.0 + 1.0j)
    - Lower diagonal: -(1.0 - 1.0j) (complex conjugate)

    """

    def _create_tridiagonal_complex_hermitian_matrix(N: int):
        """Returns a scipy.sparse csr_array that is tridiagonal Hermitian"""
        main_diag_val = 4.0
        off_diag_val = -(1.0 + 1.0j)

        main_diag = numpy.full(N, main_diag_val)
        upper_diag = numpy.full(N - 1, off_diag_val)
        lower_diag = numpy.full(N - 1, numpy.conjugate(off_diag_val))

        diagonals = [lower_diag, main_diag, upper_diag]
        offsets = [-1, 0, 1]

        A = scipy_sparse.diags(
            diagonals,
            offsets,
            shape=(N, N),
            format="csr",
            dtype=numpy.complex128,
        )

        return A

    return _create_tridiagonal_complex_hermitian_matrix


@pytest.fixture
def create_tridiagonal_real_symmetric_matrix():
    """Create a tridiagonal real symmetric sparse matrix.

    This fixture creates a tridiagonal real symmetric sparse matrix suitable
    for eigenvalue computations. The matrix has a constant main diagonal and
    constant off-diagonals.

    Parameters
    ----------
    N : int
        Number of rows (and columns) in the square matrix.

    Returns
    -------
    scipy.sparse.csr_array
        A tridiagonal real symmetric sparse matrix in SciPy CSR format.

    Notes
    -----
    The matrix is constructed with:
    - Main diagonal: 4.0
    - Upper diagonal: -1.0
    - Lower diagonal: -1.0

    """

    def _create_tridiagonal_real_symmetric_matrix(N: int):
        """Returns a scipy.sparse csr_array that is tridiagonal symmetric"""
        main_diag_val = 4.0
        off_diag_val = -1.0

        main_diag = numpy.full(N, main_diag_val)
        upper_diag = numpy.full(N - 1, off_diag_val)
        lower_diag = numpy.full(N - 1, numpy.conjugate(off_diag_val))

        diagonals = [lower_diag, main_diag, upper_diag]
        offsets = [-1, 0, 1]

        A = scipy_sparse.diags(
            diagonals, offsets, shape=(N, N), format="csr", dtype=numpy.float64
        )

        return A

    return _create_tridiagonal_real_symmetric_matrix


@pytest.fixture
def create_sparse_real_symmetric_matrix():
    """Create a generic real symmetric sparse matrix with random sparsity.

    This fixture creates a real symmetric sparse matrix suitable for eigenvalue
    computations. The sparsity pattern changes with N, making it suitable for
    testing across different matrix sizes.

    Parameters
    ----------
    N : int
        Number of rows (and columns) in the square matrix.
    density : float, optional
        Approximate density of non-zero elements. Default is 0.3.
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    scipy.sparse.csr_array
        A real symmetric sparse matrix in SciPy CSR format.

    Notes
    -----
    The matrix is constructed by:
    1. Generating a random sparse matrix
    2. Making it symmetric: A = (A + A.T) / 2
    3. Adding a diagonal component to ensure positive definiteness

    """

    def _create_sparse_real_symmetric_matrix(N: int, density=0.3, seed=42):
        """Returns a scipy.sparse csr_array that is symmetric with random sparsity"""
        numpy.random.seed(seed)

        # Generate random sparse matrix
        nnz = int(N * N * density)
        row_idx = numpy.random.randint(0, N, size=nnz)
        col_idx = numpy.random.randint(0, N, size=nnz)
        data = numpy.random.randn(nnz)

        A = scipy_sparse.csr_array((data, (row_idx, col_idx)), shape=(N, N))

        # Make it symmetric: A = (A + A.T) / 2
        A = (A + A.T) / 2

        # Add diagonal dominance to ensure well-conditioned matrix
        # This helps with convergence in eigenvalue computations
        A = A + scipy_sparse.eye(N, format="csr") * N

        return A

    return _create_sparse_real_symmetric_matrix


@pytest.fixture
def create_sparse_complex_hermitian_matrix():
    """Create a generic complex Hermitian sparse matrix with random sparsity.

    This fixture creates a complex Hermitian sparse matrix suitable for
    eigenvalue computations. The sparsity pattern changes with N, making it
    suitable for testing across different matrix sizes.

    Parameters
    ----------
    N : int
        Number of rows (and columns) in the square matrix.
    density : float, optional
        Approximate density of non-zero elements. Default is 0.3.
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    scipy.sparse.csr_array
        A complex Hermitian sparse matrix in SciPy CSR format.

    Notes
    -----
    The matrix is constructed by:
    1. Generating a random complex sparse matrix
    2. Making it Hermitian: A = (A + A.H) / 2
    3. Adding a diagonal component to ensure positive definiteness

    """

    def _create_sparse_complex_hermitian_matrix(N: int, density=0.3, seed=42):
        """Returns a scipy.sparse csr_array that is Hermitian with random sparsity"""
        numpy.random.seed(seed)

        # Generate random complex sparse matrix
        nnz = int(N * N * density)
        row_idx = numpy.random.randint(0, N, size=nnz)
        col_idx = numpy.random.randint(0, N, size=nnz)
        data_real = numpy.random.randn(nnz)
        data_imag = numpy.random.randn(nnz)
        data = data_real + 1j * data_imag

        A = scipy_sparse.csr_array(
            (data, (row_idx, col_idx)), shape=(N, N), dtype=numpy.complex128
        )

        # Make it Hermitian: A = (A + A.H) / 2
        A = (A + A.conjugate().T) / 2

        # Add diagonal dominance to ensure well-conditioned matrix
        # This helps with convergence in eigenvalue computations
        A = A + scipy_sparse.eye(N, format="csr", dtype=numpy.complex128) * N

        return A

    return _create_sparse_complex_hermitian_matrix


@pytest.fixture
def create_matrix_with_zero_diagonal():
    """Create a symmetric/Hermitian matrix with at least one zero diagonal entry.

    This fixture creates a sparse matrix with a missing diagonal element
    to test error handling in eigenvalue computations.

    Parameters
    ----------
    N : int
        Number of rows (and columns) in the square matrix.
    dtype : numpy.dtype
        Data type of the matrix (numpy.float64 or numpy.complex128).
    zero_index : int, optional
        Index of the diagonal element to set to zero. Default is N//2.
    density : float, optional
        Approximate density of non-zero elements. Default is 0.3.
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    scipy.sparse.csr_array
        A sparse matrix with a zero diagonal entry.

    """

    def _create_matrix_with_zero_diagonal(
        N: int, dtype=numpy.float64, zero_index=None, density=0.3, seed=42
    ):
        """Returns a scipy.sparse csr_array with a zero diagonal entry"""
        if zero_index is None:
            zero_index = N // 2

        numpy.random.seed(seed)

        # Generate random sparse matrix
        nnz = int(N * N * density)
        row_idx = numpy.random.randint(0, N, size=nnz)
        col_idx = numpy.random.randint(0, N, size=nnz)

        if dtype == numpy.complex128:
            data_real = numpy.random.randn(nnz)
            data_imag = numpy.random.randn(nnz)
            data = data_real + 1j * data_imag
            A = scipy_sparse.csr_array(
                (data, (row_idx, col_idx)), shape=(N, N), dtype=dtype
            )
            # Make it Hermitian
            A = (A + A.conjugate().T) / 2
            # Add diagonal dominance except for the zero index
            diag_vals = numpy.full(N, N, dtype=dtype)
            diag_vals[zero_index] = 0.0
            A = A + scipy_sparse.diags(diag_vals, 0, format="csr", dtype=dtype)
        else:
            data = numpy.random.randn(nnz)
            A = scipy_sparse.csr_array(
                (data, (row_idx, col_idx)), shape=(N, N)
            )
            # Make it symmetric
            A = (A + A.T) / 2
            # Add diagonal dominance except for the zero index
            diag_vals = numpy.full(N, N, dtype=dtype)
            diag_vals[zero_index] = 0.0
            A = A + scipy_sparse.diags(diag_vals, 0, format="csr")

        # Remove the zero from the sparse representation
        A.eliminate_zeros()

        return A

    return _create_matrix_with_zero_diagonal


@pytest.fixture
def create_non_square_matrix():
    """Create a non-square matrix for testing error handling.

    Parameters
    ----------
    rows : int
        Number of rows in the matrix.
    cols : int
        Number of columns in the matrix.
    dtype : numpy.dtype
        Data type of the matrix.

    Returns
    -------
    numpy.ndarray
        A non-square dense matrix.

    """

    def _create_non_square_matrix(rows: int, cols: int, dtype=numpy.float64):
        """Returns a non-square matrix"""
        return numpy.random.randn(rows, cols).astype(dtype)

    return _create_non_square_matrix
