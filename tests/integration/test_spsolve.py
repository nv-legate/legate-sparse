# Copyright 2024 NVIDIA Corporation
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
import pytest
import scipy.sparse as scipy_sparse
import scipy.sparse.linalg as scipy_linalg
from utils.sample import sample_dense

import legate_sparse.linalg as linalg
from legate_sparse import csr_array
from legate_sparse.runtime import runtime

# Skip all tests in this module if no GPUs are available
# since spsolve is only supported on GPU
pytestmark = pytest.mark.skipif(
    runtime.num_gpus == 0, reason="spsolve is only supported on GPU backend"
)


@pytest.mark.parametrize("N", [5, 10, 20, 50])
def test_spsolve_identity_matrix(N):
    """Test spsolve with an identity matrix."""
    A = csr_array(np.eye(N))
    b = np.ones(N)
    x = linalg.spsolve(A, b)

    # For identity matrix, x should equal b
    assert np.allclose(x, b, rtol=1e-10, atol=1e-12), (
        f"Identity matrix solution incorrect: max error = {np.max(np.abs(x - b))}"
    )


def test_spsolve_basic_square_matrix():
    """Test spsolve with a basic square matrix."""

    N = 5
    np.random.seed(42)
    A_dense = sample_dense(N, N, 0.3, 42)
    A_dense = A_dense + N * np.eye(N)

    A = csr_array(A_dense)
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x = linalg.spsolve(A, b)

    b_computed = A @ x
    assert np.allclose(b_computed, b, rtol=1e-5, atol=1e-6), (
        f"Solution verification failed: max error = {np.max(np.abs(b_computed - b))}"
    )

    A_scipy = scipy_sparse.csr_matrix(np.array(A.todense()))
    x_scipy = scipy_linalg.spsolve(A_scipy, np.array(b))
    assert np.allclose(x, x_scipy, rtol=1e-5, atol=1e-6), (
        f"Solution differs from SciPy: max error = {np.max(np.abs(x - x_scipy))}"
    )


@pytest.mark.parametrize("N", [5, 10, 20, 50])
def test_spsolve_diagonal_matrix(N):
    """Test spsolve with a diagonal matrix."""
    diag_values = np.arange(1.0, N + 1.0)
    A_dense = np.diag(diag_values)
    A = csr_array(A_dense)
    b = np.ones(N)
    x = linalg.spsolve(A, b)
    x_expected = b / diag_values
    assert np.allclose(x, x_expected, rtol=1e-10, atol=1e-12), (
        f"Diagonal matrix solution incorrect: max error = {np.max(np.abs(x - x_expected))}"
    )


@pytest.mark.parametrize("N", [5, 10, 20, 50])
def test_spsolve_tridiagonal_matrix(N):
    """Test spsolve with a tridiagonal matrix."""
    main_diag = np.full(N, 4.0)
    off_diag = np.full(N - 1, -1.0)
    A_dense = np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
    A = csr_array(A_dense)
    b = np.ones(N)
    x = linalg.spsolve(A, b)

    b_computed = A @ x
    assert np.allclose(b_computed, b, rtol=1e-5, atol=1e-6), (
        f"Tridiagonal solution verification failed: max error = {np.max(np.abs(b_computed - b))}"
    )

    A_scipy = scipy_sparse.csr_matrix(np.array(A.todense()))
    x_scipy = scipy_linalg.spsolve(A_scipy, np.array(b))
    assert np.allclose(x, x_scipy, rtol=1e-5, atol=1e-6), (
        f"Tridiagonal solution differs from SciPy: max error = {np.max(np.abs(x - x_scipy))}"
    )


@pytest.mark.parametrize("N", [5, 10, 20, 50])
def test_spsolve_symmetric_positive_definite(N):
    """Test spsolve with a symmetric positive definite matrix.
    We create an SPD matrix by A = B^T * B + N * I.
    """
    seed = 42
    B_dense = sample_dense(N, N, 0.2, seed)
    A_dense = B_dense.T @ B_dense + N * np.eye(N)
    A = csr_array(A_dense)

    # make sure it's positive definite
    eigenvalues = np.linalg.eigvals(A_dense)
    assert np.all(eigenvalues > 0), "Matrix is not positive definite"

    b = np.random.rand(N)
    x = linalg.spsolve(A, b)

    b_computed = A @ x
    assert np.allclose(b_computed, b, rtol=1e-4, atol=1e-5), (
        f"SPD solution verification failed: max error = {np.max(np.abs(b_computed - b))}"
    )


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.complex64, np.complex128]
)
def test_spsolve_all_dtypes(dtype):
    """Comprehensive test for spsolve with all cuDSS-supported data types.

    Note: cuDSS only supports floating-point and complex types.
    Integer and boolean types are not supported
    """
    N = 10

    # Create a well-conditioned matrix for each dtype
    if dtype in [np.complex64, np.complex128]:
        # For complex types, create a Hermitian positive definite matrix
        seed = 42
        np.random.seed(seed)
        B = np.random.randn(N, N) + 1j * np.random.randn(N, N)
        A_dense = (B @ B.conj().T + N * np.eye(N)).astype(dtype)
        b = np.ones(N, dtype=dtype)
    else:
        seed = 42
        A_dense = sample_dense(N, N, 0.3, seed).astype(dtype)
        A_dense = A_dense + N * np.eye(N, dtype=dtype)
        b = np.ones(N, dtype=dtype)

    # Solve the system
    A = csr_array(A_dense)
    x = linalg.spsolve(A, b)

    b_computed = A @ x
    assert np.allclose(b_computed, b, rtol=1e-4, atol=1e-5), (
        f"Solution verification failed for dtype {dtype}: max error = {np.max(np.abs(b_computed - b))}"
    )

    assert x.dtype == b.dtype, (
        f"Output dtype {x.dtype} doesn't match input dtype {b.dtype} for dtype {dtype}"
    )


@pytest.mark.parametrize("N", [5, 10, 20, 50])
def test_spsolve_upper_triangular(N):
    """Test spsolve with an upper triangular matrix."""
    A_dense = np.triu(np.random.rand(N, N) + np.eye(N))
    A = csr_array(A_dense)
    b = np.ones(N)
    x = linalg.spsolve(A, b)

    b_computed = A @ x
    assert np.allclose(b_computed, b, rtol=1e-5, atol=1e-6), (
        f"Upper triangular solution verification failed: max error = {np.max(np.abs(b_computed - b))}"
    )


@pytest.mark.parametrize("N", [5, 10, 20, 50])
def test_spsolve_lower_triangular(N):
    """Test spsolve with a lower triangular matrix."""
    A_dense = np.tril(np.ones((N, N)) + np.eye(N))
    A = csr_array(A_dense)
    b = np.ones(N)
    x = linalg.spsolve(A, b)

    b_computed = A @ x
    assert np.allclose(b_computed, b, rtol=1e-5, atol=1e-6), (
        f"Lower triangular solution verification failed: max error = {np.max(np.abs(b_computed - b))}"
    )


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
    sys.exit(0)
