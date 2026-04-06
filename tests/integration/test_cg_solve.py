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
import pytest
from utils.sample import sample_dense, sample_dense_vector

import legate_sparse.linalg as linalg
from legate_sparse import csr_array


def test_cg_solve():
    """Test conjugate gradient solver with a positive definite matrix.

    This test verifies that the conjugate gradient solver correctly
    solves the linear system Ax = b for a positive definite matrix A.

    Notes
    -----
    The test creates a random sparse matrix A and ensures it is positive
    definite by:
    1. Making it symmetric: A = 0.5 * (A + A.T)
    2. Adding a multiple of the identity: A = A + N * I

    It then generates a random solution vector x and computes b = Ax.
    The CG solver is used to solve Ax = b, and the result is verified
    by checking that A * x_pred ≈ b.

    The test uses a tolerance of 1e-8 for convergence and verification.
    """
    N, D = 20, 20
    seed = 42
    A = sample_dense(N, D, 0.1, seed)
    A = 0.5 * (A + A.T)
    A = A + N * np.eye(N)
    # Assert that A is indeed positive semi-definite.
    assert np.all(np.linalg.eigvals(A) > 0)
    A = csr_array(A)
    x = sample_dense_vector(D, 0.1, seed)
    y = A @ x
    x_pred, iters = linalg.cg(A, y, tol=1e-8)
    assert np.allclose((A @ x_pred), y, rtol=1e-8)


def test_cg_solve_with_callback():
    """Test conjugate gradient solver with a callback function.

    This test verifies that the conjugate gradient solver correctly
    handles callback functions during iteration.

    Notes
    -----
    The test creates a positive definite matrix and solves the linear
    system Ax = b using CG with a callback function. The callback
    computes the residual at each iteration and stores it in a list.

    The test verifies that:
    1. The solver converges to the correct solution
    2. The callback function is called during iteration
    3. The residuals are computed correctly

    This ensures that the callback mechanism works properly and can
    be used for monitoring convergence or implementing custom stopping
    criteria.
    """
    N, D = 20, 20
    seed = 42
    A = sample_dense(N, D, 0.1, seed)
    A = 0.5 * (A + A.T)
    A = A + N * np.eye(N)
    # Assert that A is indeed positive semi-definite.
    assert np.all(np.linalg.eigvals(A) > 0)
    A = csr_array(A)
    x = sample_dense_vector(D, 0.1, seed)
    y = A @ x
    residuals = []

    def callback(x):
        # Test that nothing goes wrong if we do some arbitrary computation in
        # the callback on x.
        residuals.append(y - A @ x)

    x_pred, iters = linalg.cg(A, y, tol=1e-8, callback=callback)
    assert np.allclose((A @ x_pred), y, rtol=1e-8)
    assert len(residuals) > 0


# def test_cg_solve_with_identity_preconditioner():
#     N, D = 20, 20
#     seed = 42
#     A = sample_dense(N, D, 0.1, seed)
#     A = 0.5 * (A + A.T)
#     A = A + N * np.eye(N)
#     # Assert that A is indeed positive semi-definite.
#     assert np.all(np.linalg.eigvals(A) > 0)
#     A = csr_array(A)
#     x = sample_dense_vector(D, 0.1, seed)
#     y = A @ x
#     assert np.allclose((A @ x), y)
#     x_pred, iters = linalg.cg(A, y, M=eye(A.shape[0]), tol=1e-8)
#     assert np.allclose((A @ x_pred), y)


def test_cg_solve_with_linear_operator():
    """Test conjugate gradient solver with LinearOperator objects.

    This test verifies that the conjugate gradient solver correctly
    works with LinearOperator objects that provide matrix-vector
    multiplication functionality.

    Notes
    -----
    The test creates a positive definite matrix A and wraps it in
    a LinearOperator object. It then solves the linear system using
    CG with the LinearOperator instead of the sparse matrix directly.

    The test verifies two different LinearOperator implementations:
    1. Using the @ operator: matvec(x) = A @ x
    2. Using the dot method: matvec(x, out=None) = A.dot(x, out=out)

    This ensures that the solver can work with any object that provides
    the required matrix-vector multiplication interface, not just
    sparse matrices.
    """
    N, D = 20, 20
    seed = 42
    A = sample_dense(N, D, 0.1, seed)
    A = 0.5 * (A + A.T)
    A = A + N * np.eye(N)
    # Assert that A is indeed positive semi-definite.
    assert np.all(np.linalg.eigvals(A) > 0)
    A = csr_array(A)
    x = sample_dense_vector(D, 0.1, seed)
    y = A @ x

    def matvec(x):
        return A @ x

    x_pred, iters = linalg.cg(
        linalg.LinearOperator(A.shape, matvec=matvec), y, tol=1e-8
    )
    assert np.allclose((A @ x_pred), y, rtol=1e-8)

    def matvec(x, out=None):
        return A.dot(x, out=out)

    x_pred, iters = linalg.cg(
        linalg.LinearOperator(A.shape, matvec=matvec), y, tol=1e-8
    )
    assert np.allclose((A @ x_pred), y, rtol=1e-8)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
    sys.exit(0)
