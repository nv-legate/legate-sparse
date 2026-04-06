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

"""Sparse Matrix-Vector Multiplication Microbenchmark.

This script benchmarks sparse matrix-vector multiplication performance
across different matrix sizes and configurations. It supports:

- Matrix size sweeps with configurable min/max sizes
- Banded matrix generation with specified non-zeros per row
- Loading matrices from Matrix Market files
- Optional repartitioning to simulate data updates
- Multiple backend support (Legate, CuPy, SciPy)

Command line arguments:
--nmin: Minimum matrix size (supports k, m, g suffixes)
--nmax: Maximum matrix size (supports k, m, g suffixes)
--nnz-per-row: Number of non-zeros per row for banded matrices
--repartition: Enable alternating x/y vectors
--filename: Load matrix from Matrix Market file
--iters: Number of benchmark iterations
--from-diags: Use sparse.diags for matrix construction
--package: Backend to use (legate, cupy, scipy)
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

from common import (
    Timer,
    banded_matrix,
    get_arg_number,
    get_phase_procs,
    parse_common_args,
)

if TYPE_CHECKING:
    from legate_sparse import csr_array


# Writing to pre-allocated array is preferred
def spmv_dispatch(
    A: csr_array, x: Any, y: Any, i: int, repartition: bool
) -> None:
    """Dispatch sparse matrix-vector multiplication operation.

    Parameters
    ----------
    A : sparse matrix
        The sparse matrix to multiply with.
    x : array_like
        Input vector.
    y : array_like
        Output vector (pre-allocated).
    i : int
        Iteration index.
    repartition : bool
        Whether to alternate between y=A*x and x=A*y.

    Notes
    -----
    This function performs sparse matrix-vector multiplication with optional
    repartitioning. When repartition is True, it alternates between computing
    y = A*x and x = A*y to simulate data updates.

    For Legate, it uses the dot method with pre-allocated output arrays.
    For other backends, it uses the @ operator.
    """
    if use_legate:
        if repartition and i % 2:
            A.dot(y, out=x)
        else:
            A.dot(x, out=y)
    else:
        if repartition and i % 2:
            x = A @ y
        else:
            y = A @ x


def run_spmv(
    A: csr_array, iters: int, repartition: bool, timer: Timer
) -> None:
    """Run sparse matrix-vector multiplication benchmark.

    Parameters
    ----------
    A : sparse matrix
        The sparse matrix to benchmark.
    iters : int
        Number of iterations to run.
    repartition : bool
        Whether to use repartitioning (alternate x and y).
    timer : Timer
        Timer object for measuring performance.

    Notes
    -----
    This function runs a benchmark of sparse matrix-vector multiplication.
    It includes warm-up runs and measures the total time and time per iteration.

    The function prints:
    - Matrix dimensions and number of non-zeros
    - Number of iterations
    - Total elapsed time
    - Time per iteration
    """
    x = np.ones((A.shape[1],))
    y = np.zeros((A.shape[0],))

    assert not repartition or (A.shape[0] == A.shape[1]), (
        "Matrix should be square for switching x and y"
    )

    # Warm up runs
    warmup_iters = 5
    for i in range(warmup_iters):
        spmv_dispatch(A, x, y, i, repartition)

    timer.start()
    for i in range(iters):
        spmv_dispatch(A, x, y, i, repartition)
    total = timer.stop()

    print(f"Dimension of A                         : {A.shape}")
    print(f"NNZ of A                               : {A.nnz}")
    print(f"Number of iterations                   : {iters}")
    print(f"Total elapsed time (ms)                : {total}")
    print(f"Time per iteration (ms)                : {total / iters}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nmin",
        type=str,
        default="1k",
        dest="nmin",
        help="Min number of rows for sweep (accepts suffixes 'k', 'm', 'g')",
    )

    parser.add_argument(
        "--nmax",
        type=str,
        default="1k",
        dest="nmax",
        help="Max number of rows for sweep (accepts suffixes 'k', 'm', 'g')",
    )

    parser.add_argument(
        "--nnz-per-row",
        type=int,
        default=11,
        dest="nnz_per_row",
        help="Number of nnz per row for generated matrix",
    )

    parser.add_argument(
        "--repartition",
        dest="repartition",
        action="store_true",
        help="Alternate between y=A*x and x=A*y, simulating data updates",
    )

    parser.add_argument(
        "-f",
        "--filename",
        dest="fname",
        type=str,
        default="",
        help="Load matrix from the file instead",
    )

    parser.add_argument(
        "-i",
        "--iters",
        type=int,
        default=100,
        dest="iters",
        help="Number of repeats",
    )
    parser.add_argument(
        "-d",
        "--from-diags",
        action="store_true",
        default=False,
        dest="from_diags",
        help="Use scipy's sparse.diags API to generate the sparse matrix",
    )

    args, _ = parser.parse_known_args()
    package, timer, np, sparse, linalg, use_legate = parse_common_args()

    init_procs, bench_procs = get_phase_procs(use_legate)

    print(f"Processor kind for initialization      : {init_procs}")
    print(f"Processor kind for computation         : {bench_procs}")

    if args.fname != "":
        # Read file from matrix
        A = sparse.mmread(args.fname)
        with bench_procs:
            run_spmv(A, args.iters, args.repartition, timer=timer)
    else:
        # Create a banded diagonal matrix with parameters from arguments.
        N = get_arg_number(args.nmin)
        while N <= get_arg_number(args.nmax):
            with init_procs:
                A = banded_matrix(N, args.nnz_per_row, args.from_diags)
            with bench_procs:
                run_spmv(A, args.iters, args.repartition, timer=timer)
            N = N * 2
