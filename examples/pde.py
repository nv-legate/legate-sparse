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

# This PDE solving application is derived from
# https://aquaulb.github.io/book_solving_pde_mooc/solving_pde_mooc/notebooks/05_IterativeMethods/05_01_Iteration_and_2D.html.

import argparse
import sys

from common import get_phase_procs, parse_common_args


def d2_mat_dirichlet_2d(nx, ny, dx, dy):
    """
    Constructs the matrix for the centered second-order accurate
    second-order derivative for Dirichlet boundary conditions in 2D

    Parameters
    ----------
    nx : integer
        number of grid points in the x direction
    ny : integer
        number of grid points in the y direction
    dx : float
        grid spacing in the x direction
    dy : float
        grid spacing in the y direction

    Returns
    -------
    d2mat : numpy.ndarray
        matrix to compute the centered second-order accurate first-order deri-
        vative with Dirichlet boundary conditions
    """
    a = 1.0 / dx**2
    g = 1.0 / dy**2
    c = -2.0 * a - 2.0 * g

    # The below is a slightly inefficient (but full cupynumeric) implementation
    # of the following python code to construct the input diagonal. We can't
    # use this code right now because cupynumeric doesn't support strided
    # slicing.
    #
    # diag_a = a * numpy.ones((nx-2)*(ny-2)-1)
    # diag_a[nx-3::nx-2] = 0.0
    diag_size = (nx - 2) * (ny - 2) - 1
    first = np.full((nx - 3), a)
    chunks = np.concatenate([np.zeros(1), first])
    diag_a = np.concatenate(
        [first, np.tile(chunks, (diag_size - (nx - 3)) // (nx - 2))]
    )
    diag_g = g * np.ones((nx - 2) * (ny - 3))
    diag_c = c * np.ones((nx - 2) * (ny - 2))

    # We construct a sequence of main diagonal elements,
    diagonals = [diag_g, diag_a, diag_c, diag_a, diag_g]
    # and a sequence of positions of the diagonal entries relative to the main
    # diagonal.
    offsets = [-(nx - 2), -1, 0, 1, nx - 2]

    # Call to the diags routine; note that diags return a representation of the
    # array; to explicitly obtain its ndarray realisation, the call to
    # .toarray() is needed. Note how the matrix has dimensions (nx-2)*(nx-2).
    # d2mat = diags(diagonals, offsets, dtype=np.float64).tocsr()
    # TODO (rohany): We want to have this conversion occur in parallel so that
    #  we can effectively weak scale. Unfortunately, I can't figure out how to
    #  adapt the scipy.sparse DIA->CSC method to work for DIA->CSR conversions.
    #  I made an attempt at using the transpose of the DIA matrix -> CSC -> CSR
    #  via a final transpose, but it turns out the direct implementation of
    #  transpose on DIA matrices uses alot of memory and is slow due to the use
    #  of indirection copies. Since we know that this matrix is symmetric, we
    #  directly use the DIA->CSC conversion, and then take the transpose to get
    #  a CSR matrix back.
    d2mat = sparse.diags(diagonals, offsets, dtype=np.float64).tocsr()

    # Return the final array
    return d2mat


def p_exact_2d(X, Y):
    """Computes the exact solution of the Poisson equation in the domain
    [0, 1]x[-0.5, 0.5] with rhs:
    b = (np.sin(np.pi * X) * np.cos(np.pi * Y) +
    np.sin(5.0 * np.pi * X) * np.cos(5.0 * np.pi * Y))

    Parameters
    ----------
    X : numpy.ndarray
        array of x coordinates for all grid points
    Y : numpy.ndarray
        array of y coordinates for all grid points

    Returns
    -------
    sol : numpy.ndarray
        exact solution of the Poisson equation
    """

    sol = -1.0 / (2.0 * np.pi**2) * np.sin(np.pi * X) * np.cos(np.pi * Y) - 1.0 / (
        50.0 * np.pi**2
    ) * np.sin(5.0 * np.pi * X) * np.cos(5.0 * np.pi * Y)

    return sol


def execute(nx, ny, plot, plot_fname, throughput, tol, max_iters, warmup_iters, timer):
    # Grid parameters.
    xmin, xmax = 0.0, 1.0  # limits in the x direction
    ymin, ymax = -0.5, 0.5  # limits in the y direction
    lx = xmax - xmin  # domain length in the x direction
    ly = ymax - ymin  # domain length in the y direction
    dx = lx / (nx - 1)  # grid spacing in the x direction
    dy = ly / (ny - 1)  # grid spacing in the y direction

    build, solve = get_phase_procs(use_legate)

    with build:
        # Create the gridline locations and the mesh grid;
        # see notebook 02_02_Runge_Kutta for more details
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
        if use_legate:
            # cuPyNumeric doesn't currently have meshgrid implemented,
            # but it is in progress. To enable scaling to large
            # datasets, explicitly perform the broadcasting
            # that meshgrid does internally.
            from legate_sparse.utils import (
                get_store_from_cupynumeric_array,
                store_to_cupynumeric_array,
            )

            x_store = get_store_from_cupynumeric_array(x)
            y_store = get_store_from_cupynumeric_array(y)
            x_t = x_store.transpose((0,)).promote(1, ny)
            y_t = y_store.promote(0, nx)
            X = store_to_cupynumeric_array(x_t)
            Y = store_to_cupynumeric_array(y_t)
        else:
            # We pass the argument `indexing='ij'` to np.meshgrid
            # as x and y should be associated respectively with the
            # rows and columns of X, Y.
            X, Y = np.meshgrid(x, y, indexing="ij")

        # Compute the rhs. Note that we non-dimensionalize the coordinates
        # x and y with the size of the domain in their respective dire-
        # ctions.
        b = np.sin(np.pi * X) * np.cos(np.pi * Y) + np.sin(5.0 * np.pi * X) * np.cos(
            5.0 * np.pi * Y
        )

        # b is currently a 2D array. We need to convert it to a column-major
        # ordered 1D array. This is done with the flatten numpy function.
        # For a physics-correct solution, b needs to be flattened in fortran
        # order. However, this is not implemented in cuPyNumeric right now.
        # Annoyingly, doing .T.flatten() raises an internal error in legate
        # when trying to invert the delinearize transform on certain processor
        # count combinations as well. Even more annoyingly, doing any sort
        # of flatten results in some bad assignment of equivalence sets within
        # Legion's dependence analysis. So if we're just testing solve
        # throughput, use an array of all ones.
        if throughput:
            n = b.shape[0] - 2
            bflat = np.ones((n * n,))
        else:
            bflat = b[1:-1, 1:-1].flatten("F")

        A = d2_mat_dirichlet_2d(nx, ny, dx, dy)

    with solve:
        # Warm up the runtime and legate by performing an SpMV on A
        # before timing. This makes sure that any deppart operations
        # using A are completed before timing.
        _ = A.dot(np.ones((A.shape[1],)))

        if throughput:
            assert max_iters > warmup_iters
            p_sol, iters = linalg.cg(A, bflat, rtol=tol, maxiter=warmup_iters)
            max_iters = max_iters - warmup_iters
            print(f"max_iters has been updated to: {max_iters}")

        timer.start()
        # If we're testing throughput, run only the prescribed number of iterations.
        if throughput:
            p_sol, iters = linalg.cg(A, bflat, rtol=tol, maxiter=max_iters)
        else:
            p_sol, iters = linalg.cg(A, bflat, rtol=tol)
        total = timer.stop()

        if throughput:
            print(
                f"CG Mesh: {nx}x{ny}, A numrows: {A.shape[0]} , ms / iter:"
                f" { total / max_iters }"  # noqa: E201, E202
            )
            sys.exit(0)
        else:
            norm_ini = np.linalg.norm(bflat)
            norm_res = np.linalg.norm(bflat - (A @ p_sol))
            # Check convergence with relative tolerance
            if norm_res <= norm_ini * tol:
                print(
                    f"CG converged after {iters} iterations, final residual relative norm:"
                    f" {norm_res / norm_ini}"  # noqa: E201, E202
                )
            else:
                print(
                    f"CG didn't converge after {iters} iterations, final residual relative"
                    f" norm: {norm_res / norm_ini}"
                )

            print(f"Total time: {total} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--nx",
        type=int,
        default=128,
        dest="nx",
        help="Number of points along X axis",
    )

    parser.add_argument(
        "-m",
        "--ny",
        type=int,
        default=128,
        dest="ny",
        help="Number of points along Y axis",
    )

    parser.add_argument(
        "-p",
        "--plot",
        dest="plot",
        action="store_true",
        help="Build plot for residuals",
    )

    parser.add_argument(
        "-f",
        "--plot_filename",
        dest="plot_fname",
        type=str,
        default="None",
        help="Intergrid transfer operator to use.",
    )

    parser.add_argument(
        "-t",
        "--throughput",
        dest="throughput",
        action="store_true",
        help="Measure only solve iterations",
    )

    parser.add_argument(
        "--tol",
        type=float,
        default=1e-10,
        dest="tol",
        help="convergence relative norm check threshold",
    )

    parser.add_argument(
        "-i",
        "--max-iters",
        type=int,
        default=None,
        dest="max_iters",
        help="Number of allowed linear iterations for the solver",
    )

    parser.add_argument(
        "-w",
        "--warmup-iters",
        type=int,
        default=None,
        dest="warmup_iters",
        help="Number of warmup iterations for the linear solver (for --throughput option)",
    )

    args, _ = parser.parse_known_args()
    _, timer, np, sparse, linalg, use_legate = parse_common_args()

    if args.throughput and args.max_iters is None:
        print("Must provide --max-iters when using -throughput.")
        sys.exit(1)

    execute(**vars(args), timer=timer)
