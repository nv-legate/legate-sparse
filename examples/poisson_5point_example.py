# Copyright 2022-2025 NVIDIA Corporation
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

"""
Solve Poisson equation: -∇²u = f(x,y) on domain [0,1]×[0,1]
With Dirichlet boundary conditions: u = 0 on boundary. We use
a manufactured solution approach u(x,y) = sin(2πx) * sin(2πy)
and use that to compute the RHS.
"""

from __future__ import annotations

import argparse

from common import parse_common_args, get_arg_number


def create_poisson_mat(n, h):
    """
    Create the 2D Poisson equation discretization matrix using 5-point stencil.

    The 5-point stencil for -∇²u at point (i,j) is:
    -u_{i,j-1} - u_{i-1,j} + 4*u_{i,j} - u_{i+1,j} - u_{i,j+1} = h²*f_{i,j}

    Parameters
    ----------
    n : int
        Number of interior grid points in each direction (n×n grid)
    h : float
        Grid spacing (h = 1/(n+1))

    Returns
    -------
    A : sparse CSR matrix
        The discretization matrix of shape (n**2, n**2)
    """
    N = n * n  # Total number of unknowns

    # stencil:
    #    -1
    # -1  4  -1
    #    -1
    main_diag = 4.0 * np.ones(N) / (h * h)
    off_diag1 = -1.0 * np.ones(N - 1) / (h * h)
    off_diag2 = -1.0 * np.ones(N - n) / (h * h)

    # cupynumeric doesn't support non-unit strides in indexing,
    # so use a mask array to set every "n" elements to zero
    zero_out_indices = np.array(range(n - 1, N - 1, n), dtype=int)
    off_diag1[zero_out_indices] = 0.0

    # The offsets   : [-n,      -1,     0,     1,     n  ]
    # correspond to : [below, left, center, right, above ]
    diagonals = [off_diag2, off_diag1, main_diag, off_diag1, off_diag2]
    offsets = [-n, -1, 0, 1, n]

    # Create the sparse matrix and convert to CSR format
    return sparse.diags(
        diagonals, offsets, shape=(N, N), dtype=np.float64, format="csr"
    )


def manufactured_solution(x, y):
    "u(x,y) = sin(2πx) * sin(2πy) satisfies u=0 on the boundary of [0,1]×[0,1]"
    return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)


def compute_rhs(x, y):
    """
    Compute the right-hand side f(x,y) for the manufactured solution.

    For u(x,y) = sin(2πx) * sin(2πy), we have:
    -∇²u = 8π² * sin(2πx) * sin(2πy) = f(x,y)
    """
    return 8 * np.pi**2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)


def solve_poisson_2d(n, verbose=True) -> float:
    """
    Solve the 2D Poisson equation with Dirichlet boundary conditions.

    Parameters
    ----------
    n : int
        Number of interior grid points in each direction
    verbose : bool
        Whether to print detailed output

    Returns
    -------
    error : float
        The L2 error between numerical and analytical solutions
    """
    h = 1.0 / (n + 1)

    if verbose:
        print(f"Solving 2D Poisson equation on {n}×{n} grid")
        print(f"Grid spacing h = {h:.6f}")
        print(f"Total unknowns: {n * n}")

    # Create grid points (interior points only) and flatten it
    x = np.linspace(h, 1 - h, n)
    y = np.linspace(h, 1 - h, n)
    X, Y = np.meshgrid(x, y, indexing="ij")
    X_flat = X.flatten()
    Y_flat = Y.flatten()

    A = create_poisson_mat(n, h)
    b = compute_rhs(X_flat, Y_flat)

    if verbose:
        print(f"Matrix shape     : {A.shape}")
        print(f"Matrix non-zeros : {A.nnz}")
        print(f"Sparsity         : {A.nnz / (n * n) ** 2:.6f}")
        print("\nSolving linear system Ax = b using spsolve...")

    x_numerical = linalg.spsolve(A, b)
    x_analytical = manufactured_solution(X_flat, Y_flat)

    error_vec = x_numerical - x_analytical
    l2_error = np.linalg.norm(error_vec) * h  # Scale by h for L2 norm
    l_inf_error = np.max(np.abs(error_vec))
    relative_error = l2_error / (np.linalg.norm(x_analytical) * h)

    residual = A @ x_numerical - b
    residual_norm = np.linalg.norm(residual)

    if verbose:
        print("\nResults:")
        print(f"L2 error         : {l2_error:.6e}")
        print(f"L∞ error         : {l_inf_error:.6e}")
        print(f"Relative L2 error: {relative_error:.6e}")
        print(f"Residual norm ||Ax - b||: {residual_norm:.6e}")

    return l2_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve 2D Poisson equation with 5-point stencil"
    )
    parser.add_argument(
        "--size",
        "-n",
        type=str,
        default="32",
        help="Number of interior grid points in each direction (default: 32)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Use this argument for verbose output",
    )

    args, _ = parser.parse_known_args()
    package, timer, np, sparse, linalg, use_legate = parse_common_args()

    n_interior = get_arg_number(args.size)

    solve_poisson_2d(n_interior, verbose=args.verbose)

    print("\n" + "=" * 60)
    print("Verification: Testing with smaller grid for convergence check")
    print("=" * 60)

    # Perform convergence tests
    n1, n2 = n_interior, n_interior * 2
    l2_error1 = solve_poisson_2d(n1, verbose=False)
    l2_error2 = solve_poisson_2d(n2, verbose=False)

    convergence_rate = np.log2(l2_error1 / l2_error2)
    print(f"Grid refinement                  : {n1}×{n1} → {n2}×{n2}")
    print(f"Error reduction factor           : {l2_error1 / l2_error2:.3f}")
    print(f"Convergence rate                 : {convergence_rate:.3f}")
    print("Expected rate for 5-point stencil: ~2.0")

    if abs(convergence_rate - 2.0) < 0.5:
        print(
            "\n✓ Solution verified: convergence rate is close to expected value"
        )
    else:
        print("\n⚠ Warning: convergence rate differs from expected value")
