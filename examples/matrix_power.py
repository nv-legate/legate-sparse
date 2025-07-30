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

"""Sparse Matrix Power Microbenchmark.

This script benchmarks sparse matrix power computation by performing repeated
matrix multiplication (A^n) and measuring performance at each step. It supports:

- Matrix generation with specified non-zeros per row or total non-zeros
- Configurable number of matrix multiplications (power exponent)
- Multiple backend support (Legate, CuPy, SciPy)

Command line arguments:
--nrows: Matrix size (supports k, m, g suffixes)
--nnz-per-row: Number of non-zeros per row for generated matrix
--nnz-total: Total number of non-zeros for generated matrix
--k: Number of matrix multiplications to perform
--nwarmups: Number of warmup iterations before timing
--same-sparsity-for-cpu-and-gpu: Use NumPy for consistent sparsity patterns
--random-seed: Random number seed for sparsity pattern generation
--package: Backend to use (legate, cupy, scipy)
"""

import argparse
from functools import reduce

import numpy.typing as npt
from common import get_arg_number, parse_common_args

# global states random_seed, rng
global random_seed, rng

# ----------------------------
# Matrix generation functions
# ----------------------------


def create_csr_with_nnz_per_row(nrows, nnz_per_row: int, dtype: npt.DTypeLike = None):
    """Create a CSR matrix with a prescribed number of nonzeros in each row.

    Parameters
    ----------
    nrows : int
        Number of rows in the matrix. Number of columns is same as number of rows.
    nnz_per_row : int
        Desired number of nonzero entries in each row.
    dtype : npt.DTypeLike, optional
        Datatype of the values. Should be one of floating point datatypes.
        Default is np.float32.

    Returns
    -------
    sparse matrix
        A CSR matrix with the specified sparsity pattern.

    Notes
    -----
    This function creates a square matrix where each row has exactly
    nnz_per_row non-zero entries. The column indices are randomly
    generated and sorted within each row.
    """
    dtype = np.float32 if dtype is None else dtype
    ncols = nrows
    nnz = nnz_per_row * nrows
    indptr = np.linspace(
        start=0, stop=nnz, num=nrows + 1, endpoint=True, dtype=np.int64
    )
    cols = rng.integers(0, ncols, nnz).reshape(ncols, nnz_per_row)
    cols = np.sort(cols, axis=1).flatten()
    vals = np.ones(nnz, dtype=dtype)
    matrix = sparse.csr_matrix((vals, cols, indptr), shape=(nrows, ncols))

    return matrix


def create_csr_with_nnz_total(nrows, nnz_total, dtype: npt.DTypeLike = None):
    """Create a CSR matrix with a prescribed total number of nonzeros.

    Parameters
    ----------
    nrows : int
        Number of rows in the matrix. Number of columns is same as number of rows.
    nnz_total : int
        Desired total number of nonzero entries in the matrix.
    dtype : npt.DTypeLike, optional
        Datatype of the values. Should be one of floating point datatypes.
        Default is np.float32.

    Returns
    -------
    sparse matrix
        A CSR matrix with the specified total number of non-zeros.

    Notes
    -----
    This function creates a square matrix with exactly nnz_total non-zero
    entries distributed randomly across the matrix. There is no guarantee
    about the number of non-zeros per row.
    """
    dtype = np.float32 if dtype is None else dtype
    ncols = nrows
    coo_rows = rng.integers(0, nrows, nnz_total)
    coo_cols = rng.integers(0, ncols, nnz_total)
    vals = np.ones(nnz_total, dtype=dtype)
    matrix = sparse.csr_matrix((vals, (coo_rows, coo_cols)), shape=(nrows, ncols))

    return matrix


# ------------------------
# Matrix Multiply routines
# ------------------------


def compute_A_power_k(A, timer, nwarmups: int = 2, k: int = 4):
    """Compute A^k and measure performance.

    Parameters
    ----------
    A : sparse matrix
        The input matrix to compute A^k.
    timer : Timer
        Timer instance to measure elapsed time.
    nwarmups : int, optional
        Number of warmup iterations before timing. Default is 2.
    k : int, optional
        Number of matrix multiplies or the exponent in A^k. Default is 4.

    Notes
    -----
    This function computes A^k by repeated matrix multiplication
    and measures the time for each step. It prints detailed timing
    information including:
    - Matrix dimensions and sparsity
    - Time for each multiplication step
    - Time for copying intermediate results
    - Overall sparsity of the final result
    """
    timer.start()
    B = A.copy()
    elapsed_time_init_copy = timer.stop()

    for _ in range(nwarmups):
        output = A @ B

    elapsed_time_spgemm = [-1.0] * k
    elapsed_time_copy = [-1.0] * k

    for hop in range(k):
        timer.start()
        output = A @ B
        elapsed_time_spgemm[hop] = timer.stop()
        timer.start()
        B = output.copy()
        elapsed_time_copy[hop] = timer.stop()

    # TODO: Wrap all the timing information in a dataclass
    nelems = reduce(lambda x, y: x * y, A.shape)
    sparsity_output = (nelems - output.nnz) * 100.0 / (A.shape[0] ** 2)

    print(f"Dimension of A                         : {A.shape}")
    print(f"Output matrix shape                    : {output.shape}")
    print(f"NNZ of A                               : {A.nnz}")
    print(f"NNZ of output                          : {output.nnz}")
    print(f"Sparsity of output (%)                 : {sparsity_output}")
    print(f"Total number of hops                   : {k}")
    print(f"Elapsed time for copy in init (ms)     : {elapsed_time_init_copy}")
    for hop in range(k):
        print(
            f"Elapsed time for spgemm for hop {hop} (ms) : {elapsed_time_spgemm[hop]}"
        )
        print(f"Elapsed time for copy   for hop {hop} (ms) : {elapsed_time_copy[hop]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-n",
        "--nrows",
        type=str,
        default="1k",
        dest="nrows",
        help="Number of rows in the generated matrix (accepts suffixes 'k', 'm', 'g')",
    )

    parser.add_argument(
        "--nnz-per-row",
        type=int,
        default=3,
        dest="nnz_per_row",
        help="Number of nnz per row for generated matrix",
    )

    parser.add_argument(
        "--nnz-total",
        type=str,
        default="-1",
        dest="nnz_total",
        help="Total number of nonzeros for the generated matrix. "
        "If both --nnz-per-row and --nnz-total are given, "
        "--nnz-total takes precedence",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=4,
        dest="k",
        help="Number of times A @ A is performed",
    )

    parser.add_argument(
        "--nwarmups",
        type=int,
        default=2,
        dest="nwarmups",
        help="Number of warmup iterations before A @ A is timed",
    )

    parser.add_argument(
        "--same-sparsity-for-cpu-and-gpu",
        action="store_true",
        help="Use NumPy to generate random nos regardless of --package",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random number seed that influences the sparsity pattern",
    )

    args, _ = parser.parse_known_args()
    _, timer, np, sparse, linalg, use_legate = parse_common_args()

    nrows = get_arg_number(args.nrows)
    nnz_total = get_arg_number(args.nnz_total)

    # this is a global variable
    global random_seed, rng
    random_seed = args.random_seed

    if args.same_sparsity_for_cpu_and_gpu:
        message = (
            "Using NumPy to generate random numbers and "
            "ensure sparsity pattern is the same across NumPy and "
            "cuPyNumeric"
        )
        print(message)

        import numpy as numpy

        rng = numpy.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng(random_seed)

    timer.start()
    if nnz_total > 0:
        A = create_csr_with_nnz_total(nrows, nnz_total, np.float32)
        print("Matrix created with total number of nonzeros")
    elif nnz_total < 0 and args.nnz_per_row > 0:
        A = create_csr_with_nnz_per_row(nrows, args.nnz_per_row, np.float32)
        print("Matrix created with number of nonzeros per row")
    elapsed_time_matrix_gen = timer.stop()

    compute_A_power_k(A, timer, int(args.nwarmups), int(args.k))

    print(f"Elapsed time in matrix creation (ms)   : {elapsed_time_matrix_gen}")
