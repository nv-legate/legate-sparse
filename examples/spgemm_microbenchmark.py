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

"""Sparse Matrix-Matrix Multiplication Microbenchmark.

This script benchmarks sparse matrix-matrix multiplication performance
with configurable matrix sizes and generation methods. It supports:

- Banded matrix generation with specified non-zeros per row
- Loading matrices from Matrix Market files
- Stable mode for partition caching vs. fresh matrix creation
- Multiple backend support (Legate, CuPy, SciPy)

Command line arguments:
--nrows: Matrix size (supports k, m, g suffixes)
--nnz-per-row: Number of non-zeros per row for banded matrices
--stable: Enable partition caching by reusing matrices
--filename1: Load first matrix from Matrix Market file
--filename2: Load second matrix from Matrix Market file
--iters: Number of benchmark iterations
--package: Backend to use (legate, cupy, scipy)
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from common import (
    Timer,
    banded_matrix,
    get_arg_number,
    get_phase_procs,
    parse_common_args,
)

if TYPE_CHECKING:
    from legate_sparse import csr_array


def spgemm_dispatch(A: csr_array, B: csr_array) -> csr_array:
    """Dispatch sparse matrix-matrix multiplication operation.

    Parameters
    ----------
    A : sparse matrix
        First sparse matrix operand.
    B : sparse matrix
        Second sparse matrix operand.

    Returns
    -------
    sparse matrix
        The result of A @ B.

    Notes
    -----
    This function performs sparse matrix-matrix multiplication using
    the @ operator, which is supported by all backends (Legate, CuPy, SciPy).
    """
    C = A @ B
    return C


def get_matrices(
    N: int, nnz_per_row: int, fname1: str, fname2: str
) -> tuple[csr_array, csr_array]:
    """Get matrices for SpGEMM benchmark.

    Parameters
    ----------
    N : int
        Matrix size (N x N) for generated matrices.
    nnz_per_row : int
        Number of non-zeros per row for banded matrices.
    fname1 : str
        Filename for first matrix (empty string to generate).
    fname2 : str
        Filename for second matrix (empty string to use first matrix).

    Returns
    -------
    tuple
        (A, B) - two sparse matrices for multiplication.

    Notes
    -----
    If fname1 is provided, loads matrices from Matrix Market files.
    If fname2 is empty, uses the same matrix for both A and B.
    Otherwise, generates banded matrices with specified parameters.
    """
    if fname1 != "":
        # Read file from matrix
        A = sparse.mmread(fname1)
        if fname2 != "":
            B = sparse.mmread(fname2)
        else:
            B = A.copy()
        return A, B
    else:
        # Create a banded diagonal matrix with parameters from arguments.
        A = banded_matrix(N, nnz_per_row)
        return A, A.copy()


def run_spgemm(
    N: int,
    nnz_per_row: int,
    fname1: str,
    fname2: str,
    iters: int,
    stable: bool,
    timer: Timer,
) -> None:
    """Run sparse matrix-matrix multiplication benchmark.

    Parameters
    ----------
    N : int
        Matrix size for generated matrices.
    nnz_per_row : int
        Number of non-zeros per row for banded matrices.
    fname1 : str
        Filename for first matrix.
    fname2 : str
        Filename for second matrix.
    iters : int
        Number of benchmark iterations.
    stable : bool
        Whether to reuse matrices (allows partition caching).
    timer : Timer
        Timer object for measuring performance.

    Notes
    -----
    This function runs a benchmark of sparse matrix-matrix multiplication.
    It supports two modes:
    - stable=True: Reuses matrices, allowing partition caching
    - stable=False: Creates fresh matrices each iteration

    The function prints:
    - Matrix dimensions and non-zero counts
    - Number of iterations
    - Total time and time per iteration
    """
    warmup_iterations = 5

    if stable:
        # Do mapping once and let Legate to re-use cached partitions

        # Create a banded diagonal matrix with nnz_per_row diagonals.
        A, B = get_matrices(N, nnz_per_row, fname1, fname2)

        # Warmup
        for _ in range(warmup_iterations):
            spgemm_dispatch(A, B)

        timer.start()
        for i in range(iters):
            spgemm_dispatch(A, B)
        total = timer.stop()
    else:
        # Create matrix for each iteration thus invalidating existing paritions
        # So we measure _full_ spgemm time (partitioning and execution)

        total = 0.0
        for i in range(iters + warmup_iterations):
            # Create a banded diagonal matrix with nnz_per_row diagonals.
            A, B = get_matrices(N, nnz_per_row, fname1, fname2)

            timer.start()
            spgemm_dispatch(A, B)
            time = timer.stop()

            # Warmup
            if i >= warmup_iterations:
                total += time

    Cnnz = spgemm_dispatch(A, B).nnz

    print(f"Dimension of A                         : {A.shape}")
    print(f"Dimension of B                         : {B.shape}")
    print(f"NNZ of A                               : {A.nnz}")
    print(f"NNZ of B                               : {B.nnz}")
    print(f"NNZ of C                               : {Cnnz}")
    print(f"Number of iterations                   : {iters}")
    print(f"Total time (ms)                        : {total}")
    print(f"Time per iteration (ms)                : {total / iters}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--nrows",
        type=str,
        default="1k",
        dest="n",
        help="Number of rows in the generated matrix (accepts suffixes 'k', 'm', 'g')",
    )

    parser.add_argument(
        "--nnz-per-row",
        type=int,
        default=5,
        dest="nnz_per_row",
        help="Number of nnz per row for generated matrix",
    )

    parser.add_argument(
        "--stable",
        dest="stable",
        action="store_true",
        help="Reuse same matrices repeatedly, allowing partitions caching",
    )

    parser.add_argument(
        "--filename1",
        dest="fname_first",
        type=str,
        default="",
        help="Load A matrix from the file instead",
    )

    parser.add_argument(
        "--filename2",
        dest="fname_second",
        type=str,
        default="",
        help="If matrix A is loaded from file - this file will be used for matrix B",
    )

    parser.add_argument(
        "-i",
        "--iters",
        type=int,
        default=100,
        dest="iters",
        help="Number of repeats",
    )

    args, _ = parser.parse_known_args()
    package, timer, np, sparse, linalg, use_legate = parse_common_args()

    init_procs, bench_procs = get_phase_procs(use_legate)

    # we will get matrices inside, since we may want to measure SpGEMM on "fresh" matrices
    run_spgemm(
        get_arg_number(args.n),
        args.nnz_per_row,
        args.fname_first,
        args.fname_first,
        args.iters,
        args.stable,
        timer=timer,
    )
