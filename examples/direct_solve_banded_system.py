import argparse
from common import get_arg_number, parse_common_args

"""Sparse Direct Solve Benchmark.

This script benchmarks sparse direct solve for a banded system of equations

"""


def create_system_of_eqns(nrows, dtype):
    """
    Creates a system of equations A*x = b where:
    - A has 4 on the main diagonal (k=0), 1 on the first and second upper diagonal (k=1, 2)
    - and 1 on the first lower diagonal (k=-1)
    - The solution x is [1, 2, 3, ..., nrows]
    - b is computed as A @ x
    """

    main_diag = np.full(nrows, 4.0)
    upper1_diag = np.ones(nrows - 1)
    upper2_diag = np.ones(nrows - 2)
    lower1_diag = np.ones(nrows - 1)

    A = sparse.diags(
        [lower1_diag, main_diag, upper1_diag, upper2_diag],
        offsets=[-1, 0, 1, 2],
        shape=(nrows, nrows),
        dtype=np.float64,
        format="csr",
    )
    x_expected = np.arange(1, nrows + 1, dtype=dtype)
    b = A @ x_expected

    return (A, b, x_expected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-n",
        "--nrows",
        type=str,
        default="12",
        dest="nrows",
        help="Number of rows in the generated matrix (accepts suffixes 'k', 'm', 'g')",
    )

    parser.add_argument(
        "--nwarmups",
        type=int,
        default=2,
        dest="nwarmups",
        help="Number of warmup iterations before spsolve is timed",
    )

    args, _ = parser.parse_known_args()
    package, timer, np, sparse, _, _ = parse_common_args()

    nrows = get_arg_number(args.nrows)
    nwarmups = args.nwarmups

    assert nrows > 0, "Matrix must contain atleast one row"
    assert nwarmups >= 0, "Warmup iterations must be >= 0"

    timer.start()
    A, b, x_expected = create_system_of_eqns(nrows, np.float64)
    elapsed_time_setup = timer.stop()

    for _ in range(nwarmups):
        x = sparse.linalg.spsolve(A, b)

    timer.start()
    x = sparse.linalg.spsolve(A, b)
    elapsed_time_solve = timer.stop()

    error_l2_norm = np.linalg.norm(x_expected - x) / np.linalg.norm(x_expected)

    print(f"Dimension of A              : {A.shape}")
    print(f"Dimension of b              : {b.shape}")
    print(f"Dimension of x              : {x.shape}")
    print(f"NNZ of A                    : {A.nnz}")
    print(f"Elapsed time for setup (ms) : {elapsed_time_setup}")
    print(f"Elapsed time for solve (ms) : {elapsed_time_solve}")
    print(f"Error in solution           : {error_l2_norm}")
