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

import cupynumeric as cn
import numpy
import pytest

import legate_sparse.linalg as linalg
from legate_sparse import csr_array


@pytest.fixture
def check_eigsh_result():
    """Checks if the Eigenvalues match Ax = wx.

    Parameters
    ----------
    A : csr_array
        Input sparse matrix
    w: numpy.ndarray
        Eigen values
    x: numpy.ndarray
        Eigen vectors
    res_tol: float, optional
        Acceptable residual
    """

    def _check_eigsh_result(A, w, x, res_tol: float = 1e-3):
        """Verify eigsh results by checking residual, Ax - wx"""
        for i in range(w.size):
            # ||Ax - wx|| / ||w||
            Ax = A @ x[:, i]
            wx = w[i] * x[:, i]
            res = cn.linalg.norm(Ax - wx) / cn.abs(w[i])
            assert res < res_tol, (
                f"Residual {res} exceeds tol of {res_tol} for {i}th eigen value"
            )

    return _check_eigsh_result


class TestEigsh:
    """Test eigsh with various parameters following CuPy's testing approach."""

    # ------ Test arguments: N, k, which

    @pytest.mark.parametrize("N", [10, 16])
    @pytest.mark.parametrize("which", ["LM", "LA", "SA"])
    @pytest.mark.parametrize("k", [1, 3])
    def test_eigsh_real_symmetric(
        self,
        N,
        which,
        k,
        create_tridiagonal_real_symmetric_matrix,
        check_eigsh_result,
    ):
        """Test eigsh with real symmetric tridiagonal matrices."""
        A_scipy = create_tridiagonal_real_symmetric_matrix(N)
        A = csr_array(A_scipy.todense())

        w, x = linalg.eigsh(A, k=k, which=which, return_eigenvectors=True)

        assert cn.allclose(cn.imag(w), 0, atol=1e-6), (
            "Eigenvalues should be real for real symmetric matrices"
        )
        assert w.shape == (k,), f"Expected {k} eigenvalues, found {w.shape}"

        check_eigsh_result(A, w, x)

    @pytest.mark.parametrize("N", [10, 16])
    @pytest.mark.parametrize("which", ["LM", "LA", "SA"])
    @pytest.mark.parametrize("k", [1, 3])
    def test_eigsh_complex_hermitian(
        self,
        N,
        which,
        k,
        create_tridiagonal_complex_hermitian_matrix,
        check_eigsh_result,
    ):
        """Test eigsh with complex Hermitian tridiagonal matrices."""
        A_scipy = create_tridiagonal_complex_hermitian_matrix(N)
        A = csr_array(A_scipy.todense())

        w, x = linalg.eigsh(A, k=k, which=which, return_eigenvectors=True)

        assert cn.allclose(cn.imag(w), 0, atol=1e-6), (
            "Eigenvalues should be real for Hermitian matrices"
        )
        assert w.shape == (k,), f"Expected {k} eigenvalues"

        check_eigsh_result(A, w, x)

    # ------ Test argument return_eigenvector

    def test_eigsh_eigenvalues_only_real(
        self, create_tridiagonal_real_symmetric_matrix
    ):
        """Test eigsh with return_eigenvectors=False for real matrices."""
        N, k = 10, 2
        which = "LM"
        A_scipy = create_tridiagonal_real_symmetric_matrix(N)
        A = csr_array(A_scipy.todense())

        w = linalg.eigsh(A, k=k, which=which, return_eigenvectors=False)

        assert cn.allclose(cn.imag(w), 0, atol=1e-6), (
            "Eigenvalues should be real"
        )
        assert w.shape == (k,), f"Expected {k} eigenvalues"

    def test_eigsh_eigenvalues_only_complex(
        self, create_tridiagonal_complex_hermitian_matrix
    ):
        """Test eigsh with return_eigenvectors=False for complex matrices."""
        N, k = 10, 2
        which = "LM"
        A_scipy = create_tridiagonal_complex_hermitian_matrix(N)
        A = csr_array(A_scipy.todense())

        w = linalg.eigsh(A, k=k, which=which, return_eigenvectors=False)

        assert cn.allclose(cn.imag(w), 0, atol=1e-6), (
            "Eigenvalues should be real"
        )
        assert w.shape == (k,), f"Expected {k} eigenvalues"

    # ------ Test argument v0

    def test_eigsh_with_v0_real(
        self, create_tridiagonal_real_symmetric_matrix, check_eigsh_result
    ):
        """Test eigsh with user-provided initial vector v0 for real matrices."""
        N, k = 10, 2
        A_scipy = create_tridiagonal_real_symmetric_matrix(N)
        A = csr_array(A_scipy.todense())

        v0 = numpy.array(cn.random.randn(N), dtype=numpy.float64)

        w, x = linalg.eigsh(
            A, k=k, which="LM", v0=v0, return_eigenvectors=True
        )

        assert cn.allclose(cn.imag(w), 0, atol=1e-6), (
            "Eigenvalues should be real"
        )
        check_eigsh_result(A, w, x)

    def test_eigsh_with_v0_complex(
        self, create_tridiagonal_complex_hermitian_matrix, check_eigsh_result
    ):
        """Test eigsh with user-provided initial vector v0 for complex matrices."""
        N, k = 10, 2
        A_scipy = create_tridiagonal_complex_hermitian_matrix(N)
        A = csr_array(A_scipy.todense())

        v0 = cn.array(
            numpy.random.randn(N) + 1j * numpy.random.randn(N),
            dtype=numpy.complex128,
        )

        w, x = linalg.eigsh(
            A, k=k, which="LM", v0=v0, return_eigenvectors=True
        )

        assert cn.allclose(cn.imag(w), 0, atol=1e-10), (
            "Eigenvalues should be real"
        )
        check_eigsh_result(A, w, x)

    # ------ Test output sortedness

    def test_eigsh_sorted_eigenvalues(
        self, create_tridiagonal_real_symmetric_matrix
    ):
        """Test that eigenvalues are returned sorted."""
        N, k = 20, 6
        A_scipy = create_tridiagonal_real_symmetric_matrix(N)
        A = csr_array(A_scipy.todense())

        w, _ = linalg.eigsh(A, k=k, which="LM", return_eigenvectors=True)

        # Eigenvalues should be sorted in ascending order
        w_sorted = cn.sort(w)
        assert cn.allclose(w, w_sorted), "Eigenvalues should be sorted"


class TestEigshLargeProblems:
    """Test eigsh with larger problem sizes."""

    @pytest.mark.parametrize("N", [15, 30])
    @pytest.mark.parametrize("k", [3, 6])
    @pytest.mark.parametrize("which", ["LM", "SA"])
    def test_eigsh_large_real_symmetric(
        self,
        N,
        k,
        which,
        create_tridiagonal_real_symmetric_matrix,
        check_eigsh_result,
    ):
        """Test eigsh with large real symmetric tridiagonal matrices."""
        A_scipy = create_tridiagonal_real_symmetric_matrix(N)
        A = csr_array(A_scipy.todense())

        w, x = linalg.eigsh(A, k=k, which=which, return_eigenvectors=True)

        assert cn.allclose(cn.imag(w), 0, atol=1e-6), (
            "Eigenvalues should be real"
        )
        assert w.shape == (k,), f"Expected {k} eigenvalues"
        check_eigsh_result(A, w, x)

    @pytest.mark.parametrize("N", [15, 30])
    @pytest.mark.parametrize("k", [3, 6])
    @pytest.mark.parametrize("which", ["LM", "SA"])
    def test_eigsh_large_complex_hermitian(
        self,
        N,
        k,
        which,
        create_tridiagonal_complex_hermitian_matrix,
        check_eigsh_result,
    ):
        """Test eigsh with large complex Hermitian tridiagonal matrices."""
        A_scipy = create_tridiagonal_complex_hermitian_matrix(N)
        A = csr_array(A_scipy.todense())

        w, x = linalg.eigsh(A, k=k, which=which, return_eigenvectors=True)

        assert cn.allclose(cn.imag(w), 0, atol=1e-6), (
            "Eigenvalues should be real"
        )
        assert w.shape == (k,), f"Expected {k} eigenvalues"
        check_eigsh_result(A, w, x)


class TestEigshRandomSparse:
    """Test eigsh with random sparse symmetric/Hermitian matrices."""

    @pytest.mark.parametrize("N", [15, 30])
    @pytest.mark.parametrize("k", [1, 3])
    @pytest.mark.parametrize("which", ["LM", "SA"])
    def test_eigsh_random_real_symmetric(
        self,
        N,
        k,
        which,
        create_sparse_real_symmetric_matrix,
        check_eigsh_result,
    ):
        """Test eigsh with random sparse real symmetric matrices."""
        A_scipy = create_sparse_real_symmetric_matrix(N, density=0.3, seed=42)
        A = csr_array(A_scipy.todense())

        w, x = linalg.eigsh(A, k=k, which=which, return_eigenvectors=True)

        assert cn.allclose(cn.imag(w), 0, atol=1e-6), (
            "Eigenvalues should be real"
        )
        assert w.shape == (k,), f"Expected {k} eigenvalues"
        check_eigsh_result(A, w, x)

    @pytest.mark.parametrize("N", [15, 30])
    @pytest.mark.parametrize("k", [1, 3])
    @pytest.mark.parametrize("which", ["LM", "SA"])
    def test_eigsh_random_complex_hermitian(
        self,
        N,
        k,
        which,
        create_sparse_complex_hermitian_matrix,
        check_eigsh_result,
    ):
        """Test eigsh with random sparse complex Hermitian matrices."""
        A_scipy = create_sparse_complex_hermitian_matrix(
            N, density=0.3, seed=42
        )
        A = csr_array(A_scipy.todense())

        w, x = linalg.eigsh(A, k=k, which=which, return_eigenvectors=True)

        assert cn.allclose(cn.imag(w), 0, atol=1e-6), (
            "Eigenvalues should be real"
        )
        assert w.shape == (k,), f"Expected {k} eigenvalues"
        check_eigsh_result(A, w, x)


class TestEigshLinearOperator:
    """Test eigsh with LinearOperator input."""

    @pytest.mark.parametrize("N", [10, 20])
    @pytest.mark.parametrize("k", [1, 3])
    @pytest.mark.parametrize("which", ["LM", "SA"])
    def test_eigsh_linear_operator_real(
        self,
        N,
        k,
        which,
        create_tridiagonal_real_symmetric_matrix,
        check_eigsh_result,
    ):
        """Test eigsh with LinearOperator wrapping a real symmetric matrix."""
        A_scipy = create_tridiagonal_real_symmetric_matrix(N)
        A_dense = cn.array(A_scipy.todense())

        A_op = linalg.LinearOperator(
            shape=(N, N), matvec=lambda v: A_dense @ v, dtype=A_dense.dtype
        )

        w, x = linalg.eigsh(A_op, k=k, which=which, return_eigenvectors=True)

        assert cn.allclose(cn.imag(w), 0.0, atol=1e-6), (
            "Eigenvalues should be real"
        )
        assert w.shape == (k,), f"Expected {k} eigenvalues"
        check_eigsh_result(A_dense, w, x)

    @pytest.mark.parametrize("N", [10, 20])
    @pytest.mark.parametrize("k", [1, 3])
    @pytest.mark.parametrize("which", ["LM", "SA"])
    def test_eigsh_linear_operator_complex(
        self,
        N,
        k,
        which,
        create_tridiagonal_complex_hermitian_matrix,
        check_eigsh_result,
    ):
        """Test eigsh with LinearOperator wrapping a complex Hermitian matrix."""
        A_scipy = create_tridiagonal_complex_hermitian_matrix(N)
        A_dense = cn.array(A_scipy.todense())

        A_op = linalg.LinearOperator(
            shape=(N, N), matvec=lambda v: A_dense @ v, dtype=A_dense.dtype
        )

        w, x = linalg.eigsh(A_op, k=k, which=which, return_eigenvectors=True)

        assert cn.allclose(cn.imag(w), 0.0, atol=1e-6), (
            "Eigenvalues should be real"
        )
        assert w.shape == (k,), f"Expected {k} eigenvalues"
        check_eigsh_result(A_dense, w, x)


class TestEigshErrors:
    """Test eigsh error handling."""

    def test_non_square_matrix(self):
        """Test that non-square matrix raises ValueError."""
        A_rect = csr_array(numpy.random.randn(10, 15))
        with pytest.raises(ValueError, match="expected square matrix"):
            linalg.eigsh(A_rect, k=1)

    def test_k_too_large(self):
        """Test that k >= n raises ValueError."""
        n = 10
        A = csr_array(numpy.eye(n))
        with pytest.raises(ValueError, match="k must be smaller than n"):
            linalg.eigsh(A, k=n)

    def test_k_zero_or_negative(self):
        """Test that k <= 0 raises ValueError."""
        A = csr_array(numpy.eye(10))
        with pytest.raises(ValueError, match="k must be greater than 0"):
            linalg.eigsh(A, k=0)

    def test_invalid_which(self):
        """Test that invalid which raises ValueError."""
        A = csr_array(numpy.eye(10))
        with pytest.raises(ValueError, match="which must be"):
            linalg.eigsh(A, k=1, which="INVALID")


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
