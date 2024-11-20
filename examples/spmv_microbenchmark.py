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


# Writing to pre-allocated array is preferred
def spmv_dispatch(A, x, y, i, repartition):
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


def run_spmv(A, iters, repartition, timer):
    x = np.ones((A.shape[1],))
    y = np.zeros((A.shape[0],))

    assert not repartition or (
        A.shape[0] == A.shape[1]
    ), "Matrix should be square for switching x and y"

    # Warm up runs
    warmup_iters = 5
    for i in range(warmup_iters):
        spmv_dispatch(A, x, y, i, repartition)

    timer.start()
    for i in range(iters):
        spmv_dispatch(A, x, y, i, repartition)
    total = timer.stop()

    print(f"SPMV rows: {A.shape[0]}, nnz: {A.nnz} , ms / iter: {total / iters}")


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
    _, timer, np, sparse, linalg, use_legate = parse_common_args()

    init_procs, bench_procs = get_phase_procs(use_legate)

    print(f"Processor kind for initialization: {init_procs}")
    print(f"Processor kind for computation   : {bench_procs}")

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
