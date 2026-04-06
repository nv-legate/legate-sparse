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

"""Tests for GEAM API and sparse matrix arithmetic operations."""

import sys

import cupynumeric as np
import pytest
from utils.banded_matrix import banded_matrix
from utils.sample import simple_system_gen

import legate_sparse as sparse
from legate_sparse.csr import geam


# =============================================================================
# GEAM API Tests - Error Cases
# =============================================================================


def test_geam_sparse_dense_mismatch_A_dense():
    """Test that geam raises error when only one of the arrays is sparse."""
    N = 5
    np.random.seed(42)
    A_dense = np.random.rand(N, N)
    B_sparse = banded_matrix(N, 3)

    with pytest.raises((TypeError, AttributeError)):
        geam(A_dense, B_sparse, 1.0, 2.0)

    with pytest.raises((TypeError, AttributeError)):
        geam(B_sparse, A_dense, 1.0, 2.0)


def test_geam_wrong_sparsity_pattern_for_C():
    """Providing C with incompatible sparsity pattern leads to incorrect results."""
    N = 5
    np.random.seed(42)

    A = banded_matrix(N, 3)  # tri-diagonal
    B = banded_matrix(N, 5)  # penta-diagonal

    C_correct = geam(A, B, 2.0, 3.0)
    C_wrong = banded_matrix(N, 3)  # wrong pattern - too few non-zeros
    C_result = geam(A, B, 2.0, 3.0, C=C_wrong)

    # Results should NOT match due to incompatible sparsity
    assert not np.allclose(C_correct.todense(), C_result.todense())


# =============================================================================
# GEAM API Tests - Success Cases
# =============================================================================


@pytest.mark.parametrize("N", [5, 15, 30])
def test_geam_basic_without_C(N):
    """Test geam without providing C."""
    np.random.seed(42)
    A_dense, A_sparse, _ = simple_system_gen(N, N, sparse.csr_array, tol=0.3)
    B_dense, B_sparse, _ = simple_system_gen(N, N, sparse.csr_array, tol=0.3)

    C_sparse = geam(A_sparse, B_sparse, 2.5, -1.5)
    C_expected = 2.5 * A_dense + (-1.5) * B_dense

    assert np.allclose(C_sparse.todense(), C_expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("N", [5, 15, 30])
def test_geam_basic_with_C(N):
    """Test geam with pre-allocated C, then reuse it."""
    np.random.seed(42)
    A_dense, A_sparse, _ = simple_system_gen(N, N, sparse.csr_array, tol=0.3)
    B_dense, B_sparse, _ = simple_system_gen(N, N, sparse.csr_array, tol=0.3)

    C_sparse = geam(A_sparse, B_sparse, 2.0, 3.0)
    assert np.allclose(C_sparse.todense(), 2.0 * A_dense + 3.0 * B_dense)

    C_sparse = geam(A_sparse, B_sparse, -1.0, 0.5, C=C_sparse)
    assert np.allclose(C_sparse.todense(), -1.0 * A_dense + 0.5 * B_dense)


@pytest.mark.parametrize(
    "alpha,beta", [(1.0, 1.0), (1.0, -1.0), (2.0, 0.0), (0.0, 3.0)]
)
def test_geam_various_scalars(alpha, beta):
    """Test geam with various scalar combinations."""
    N = 15
    np.random.seed(42)
    A_dense, A_sparse, _ = simple_system_gen(N, N, sparse.csr_array, tol=0.3)
    B_dense, B_sparse, _ = simple_system_gen(N, N, sparse.csr_array, tol=0.3)

    C_sparse = geam(A_sparse, B_sparse, alpha, beta)
    assert np.allclose(C_sparse.todense(), alpha * A_dense + beta * B_dense)


def test_geam_loop_with_C_reuse():
    """Test geam in a loop where C is reused across iterations."""
    N = 15
    np.random.seed(42)

    A_sparse = banded_matrix(N, 3)
    B_sparse = banded_matrix(N, 3)
    C_sparse = geam(A_sparse, B_sparse, 1.0, 1.0)

    for i in range(1, 5):
        A_new = banded_matrix(N, 3, init_with_ones=False)
        B_new = banded_matrix(N, 3, init_with_ones=False)
        scale_A, scale_B = float(i + 1), float(i + 2)

        C_sparse = geam(A_new, B_new, scale_A, scale_B, C=C_sparse)
        C_expected = scale_A * A_new.todense() + scale_B * B_new.todense()

        assert np.allclose(C_sparse.todense(), C_expected)


def test_geam_identical_matrices():
    """Test geam when A and B are identical."""
    N = 15
    np.random.seed(42)
    A_dense, A_sparse, _ = simple_system_gen(N, N, sparse.csr_array, tol=0.3)

    C_sparse = geam(A_sparse, A_sparse, 2.0, 3.0)
    assert np.allclose(C_sparse.todense(), 5.0 * A_dense)


def test_geam_disjoint_sparsity_patterns():
    """Test geam when A and B have disjoint sparsity patterns."""
    N = 15
    np.random.seed(42)

    A_dense = np.triu(np.random.rand(N, N))
    A_sparse = sparse.csr_array(A_dense)
    B_dense = np.tril(np.random.rand(N, N), k=-1)
    B_sparse = sparse.csr_array(B_dense)

    C_sparse = geam(A_sparse, B_sparse, 1.5, 2.5)
    assert np.allclose(C_sparse.todense(), 1.5 * A_dense + 2.5 * B_dense)


# =============================================================================
# Dunder Method Tests (__add__, __sub__, __radd__, __rsub__)
# =============================================================================


class TestCSRArithmetic:
    """Tests for CSR matrix arithmetic dunder methods."""

    @pytest.fixture
    def matrices(self):
        """Create test matrices."""
        N = 15
        np.random.seed(42)
        A_dense, A_sparse, _ = simple_system_gen(
            N, N, sparse.csr_array, tol=0.3
        )
        B_dense, B_sparse, _ = simple_system_gen(
            N, N, sparse.csr_array, tol=0.3
        )
        return A_dense, A_sparse, B_dense, B_sparse

    # -------------------------------------------------------------------------
    # Sparse + Sparse, Sparse - Sparse
    # -------------------------------------------------------------------------

    def test_add_sparse_sparse(self, matrices):
        """A + B where both are sparse."""
        A_dense, A_sparse, B_dense, B_sparse = matrices
        C = A_sparse + B_sparse
        assert np.allclose(C.todense(), A_dense + B_dense)

        C = A_sparse - B_sparse
        assert np.allclose(C.todense(), A_dense - B_dense)

    # -------------------------------------------------------------------------
    # Sparse + Dense, Dense + Sparse
    # -------------------------------------------------------------------------

    def test_add_sparse_dense(self, matrices):
        """sparse + dense returns dense."""
        A_dense, A_sparse, B_dense, _ = matrices
        C = A_sparse + B_dense
        assert np.allclose(C, A_dense + B_dense)

    @pytest.mark.skip(
        reason="cupynumeric intercepts dense+sparse before __radd__ is called"
    )
    def test_add_dense_sparse(self, matrices):
        """dense + sparse should return dense (currently broken in cupynumeric)."""
        A_dense, _, B_dense, B_sparse = matrices
        C = A_dense + B_sparse
        assert np.allclose(C, A_dense + B_dense)

    # -------------------------------------------------------------------------
    # Sparse - Dense, Dense - Sparse
    # -------------------------------------------------------------------------

    def test_sub_sparse_dense(self, matrices):
        """sparse - dense returns dense."""
        A_dense, A_sparse, B_dense, _ = matrices
        C = A_sparse - B_dense
        assert np.allclose(C, A_dense - B_dense)

    @pytest.mark.skip(
        reason="cupynumeric intercepts dense-sparse before __rsub__ is called"
    )
    def test_sub_dense_sparse(self, matrices):
        """dense - sparse should return dense (currently broken in cupynumeric)."""
        A_dense, _, B_dense, B_sparse = matrices
        C = A_dense - B_sparse
        assert np.allclose(C, A_dense - B_dense)

    # -------------------------------------------------------------------------
    # Sparse + Scalar, Scalar + Sparse
    # -------------------------------------------------------------------------

    def test_add_sparse_zero(self, matrices):
        """A + 0 should return a copy of A."""
        A_dense, A_sparse, _, _ = matrices
        C = A_sparse + 0
        assert np.allclose(C.todense(), A_dense)

        C = 0 + A_sparse
        assert np.allclose(C.todense(), A_dense)

    def test_add_sparse_nonzero_scalar_raises(self, matrices):
        """A + nonzero scalar should raise NotImplementedError."""
        _, A_sparse, _, _ = matrices
        with pytest.raises(NotImplementedError):
            _ = A_sparse + 5.0
        with pytest.raises(NotImplementedError):
            _ = 5.0 + A_sparse

    # -------------------------------------------------------------------------
    # Sparse - Scalar, Scalar - Sparse
    # -------------------------------------------------------------------------

    def test_sub_sparse_zero(self, matrices):
        """A - 0 should return a copy of A."""
        A_dense, A_sparse, _, _ = matrices
        C = A_sparse - 0
        assert np.allclose(C.todense(), A_dense)

        C = 0 - A_sparse
        assert np.allclose(C.todense(), -A_dense)

    def test_sub_sparse_nonzero_scalar_raises(self, matrices):
        """Subtracting a nonzero scalar should raise NotImplementedError."""
        _, A_sparse, _, _ = matrices
        with pytest.raises(NotImplementedError):
            _ = A_sparse - 5.0
        with pytest.raises(NotImplementedError):
            _ = 5.0 - A_sparse


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
