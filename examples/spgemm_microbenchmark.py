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

import argparse

from common import banded_matrix, get_arg_number, get_phase_procs, parse_common_args


def spgemm_dispatch(A, B):
    C = A @ B
    return C


def get_matrices(N, nnz_per_row, fname1, fname2):
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


def run_spgemm(N, nnz_per_row, fname1, fname2, iters, stable, timer):
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

    print(
        f"SPGEMM {A.shape}x{B.shape} , nnz ({A.nnz})x({B.nnz})->({Cnnz}) : ms /"
        f" iteration: {total / iters}"
    )


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
    _, timer, np, sparse, linalg, use_legate = parse_common_args()

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
