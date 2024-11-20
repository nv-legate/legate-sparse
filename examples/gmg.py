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


# Portions of this file are also subject to the following license:
#
# The MIT License (MIT)
#
# Copyright (c) 2008-2015 PyAMG Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse

# for some small data manipulations on host
import numpy
from common import diffusion2D, get_phase_procs, parse_common_args, poisson2D


def max_eigenvalue(A, iters=15):
    # Compute eigenvector associated with maximum eigenvalue via power
    # iteration.  This is the same as Steven's imp for estimating spectral
    # radius.
    x1 = np.random.rand(A.shape[1]).reshape(-1, 1)
    for _ in range(iters):
        x1 = A @ x1
        x1 /= np.linalg.norm(x1)
    # Compute and return max eigenvalue via Raleigh quotient.
    # This is np.dot(A @ x1, x1) / np.dot(x1, x1)
    # but since x1 is a unit vector, we can assume denominator is 1.
    return np.dot(x1.T, A @ x1).item()


class GMG(object):
    """
    Geometric Multigrid solver for the 2D Poisson problem.

    - Source on correctness of restriction / prolongation operators: [1]
    - Sources on V-cycle algorithm: [1, 2, 3, 4]
    - Source on preconditioned conjugate gradient and Gauss-Seidel smoothing: [4]

    [1] https://www.researchgate.net/publication/220690328_A_Multigrid_Tutorial_2nd_Edition
    [2] https://github.com/pyamg/pyamg
    [3] http://www.cs.columbia.edu/cg/pdfs/28_GPUSim.pdf
    [4] https://netlib.org/utk/people/JackDongarra/PAPERS/HPCG-benchmark.pdf
    """  # noqa: E501

    def __init__(self, A, shape, levels, smoother, gridop, machine):
        self.A = A
        self.shape = shape
        self.N = numpy.prod(self.shape)
        self.levels = levels
        self.restriction_op = {
            "injection": injection_operator,
            "linear": linear_operator,
        }[gridop]
        self.smoother = {"jacobi": WeightedJacobi}[smoother]()
        self.operators = self.compute_operators(A)
        self.temp = None
        self.machine = machine
        self.proc_kind = machine.preferred_target

    def compute_operators(self, A):
        operators = []
        dim = self.N
        self.smoother.init_level_params(A, 0)
        for level in range(self.levels):
            R, dim = self.compute_restriction_level(dim)
            P = R.T
            # assert sparse.issparse(P)
            A = R @ A @ P
            # assert sparse.issparse(A)
            self.smoother.init_level_params(A, level + 1)
            operators.append((R, A, P))
        return operators

    def cycle(self, r):
        # Kick off the cycle with the top-level machine.
        # TODO (marsaev): there are issues with scoping
        # disabling it for now
        return self._cycle(self.A, r, 0, self.machine)

    def _cycle(self, A, r, level, machine):
        if level == self.levels - 1:
            return self.smoother.coarse(A, r, None, level=level)
        x = None
        # Do one pre-smoothing iteration.
        R, coarse_A, P = self.operators[level]
        x = self.smoother.pre(A, r, x, level=level)
        # Compute the residual.
        fine_r = r - A.dot(x)

        # Restrict the residual.
        if use_legate:
            # TODO (marsaev): there col-split splmv optimization
            coarse_r = R.dot(fine_r)
        else:
            coarse_r = R.dot(fine_r)

        # Compute coarse solution using a subset of the machine.
        # TODO (marsaev): there are issues with scoping
        # disabling it for now
        coarse_x = self._cycle(coarse_A, coarse_r, level + 1, self.machine)

        fine_x = P @ coarse_x
        x_corrected = x + fine_x
        # Do one post-smoothing iteration.
        return self.smoother.post(A, r, x_corrected, level=level)

    def compute_restriction_level(self, fine_dim):
        return self.restriction_op(fine_dim)

    def linear_operator(self):
        return linalg.LinearOperator(
            self.A.shape, dtype=float, matvec=lambda r: self.cycle(r)
        )


class WeightedJacobi(object):
    def __init__(self, omega=4.0 / 3.0):
        # Basically, similar solution to PyAMG.
        self.level_params = []
        self._init_omega = omega

    def init_level_params(self, A, level):
        D_inv = 1.0 / A.diagonal()
        # We need to create a new sparse matrix with just this modified
        # diagonal of A. sparse.eye doesn't have this nob, but we can take
        # the output of sparse.eye and mess with it to get the matrix
        # that we want.
        D_inv_nnz = min(A.shape[0], A.shape[1])
        D_inv_mat = sparse.csr_array(
            (
                np.ones(D_inv_nnz).astype(A.dtype),
                (
                    np.arange(D_inv_nnz).astype(sparse.coord_ty),
                    np.arange(D_inv_nnz).astype(sparse.coord_ty),
                ),
            ),
            shape=A.shape,
            dtype=A.dtype,
            copy=False,
        )
        """
        sparse.eye(
            A.shape[0], n=A.shape[1], dtype=A.dtype, format="csr"
        )
        """
        D_inv_mat.data = 1.0 / D_inv
        spectral_radius = max_eigenvalue(A @ D_inv_mat, 1)
        omega = self._init_omega / spectral_radius
        self.level_params.append((omega, D_inv))
        assert len(self.level_params) - 1 == level

    def __call__(self, A, r, x, level):
        omega, D_inv = self.level_params[level]
        return (1 - omega) * x + omega * (r - A @ x + x / D_inv) * D_inv

    def pre(self, A, r, x, level):
        if x is not None:
            raise Exception("Expected x is None.")
        omega, D_inv = self.level_params[level]
        return omega * r * D_inv

    def post(self, A, r, x, level):
        omega, D_inv = self.level_params[level]
        return x + omega * (r - A @ x) * D_inv

    def coarse(self, A, r, x, level):
        return self.pre(A, r, x, level)
        # return sparse.linalg.spsolve(A, r)


def injection_operator(fine_dim):
    fine_shape = (int(np.sqrt(fine_dim)),) * 2
    coarse_shape = fine_shape[0] // 2, fine_shape[1] // 2
    coarse_dim = numpy.prod(coarse_shape)
    Rp = np.arange(coarse_dim + 1)
    Rx = np.ones((coarse_dim,), dtype=np.float64)
    ij = np.arange(coarse_dim, dtype=np.int64)
    i = ij % coarse_shape[1]
    j = ij // coarse_shape[1]
    Rj = 2 * i + 2 * j * coarse_shape[1]
    R = sparse.csr_matrix((Rx, Rj, Rp), shape=(coarse_dim, fine_dim), dtype=np.float64)
    return R, coarse_dim


def linear_operator(fine_dim):
    fine_shape = (int(np.sqrt(fine_dim)),) * 2
    coarse_shape = fine_shape[0] // 2, fine_shape[1] // 2
    coarse_dim = np.prod(coarse_shape)
    # Construct CSR directly.
    Rp = numpy.empty(coarse_dim + 1, dtype=np.int64)
    # Get an upper bound on the total number of non-zeroes, and construct Rj
    # and Rx based on this bound.  Computing this value exactly is tedious and
    # the extra allocation can be truncated at the end.  We won't need more
    # than 9*coarse_dim rows.
    nnz = 9 * coarse_dim
    Rj = numpy.empty((nnz,), dtype=np.int64)
    Rx = numpy.empty((nnz,), dtype=np.float64)
    p = 0

    def flatten(i, j):
        return i * fine_shape[1] + j

    for ij in range(coarse_dim):
        Rp[ij] = p
        # For linear interpolation,
        # we have 9 points over which to average in the 2d case.
        # The coefficient matrix will act as a stencil operator.
        i, j = (ij // coarse_shape[1]), (ij % coarse_shape[1])
        # Corners.
        # r[2*i-1, 2*j-1] = 1/16
        # r[2*i-1, 2*j+1] = 1/16
        # r[2*i+1, 2*j-1] = 1/16
        # r[2*i+1, 2*j+1] = 1/16
        # Edges.
        # r[2*i, 2*j+1] = 2/16
        # r[2*i, 2*j-1] = 2/16
        # r[2*i-1, 2*j] = 2/16
        # r[2*i+1, 2*j] = 2/16
        # Center.
        # r[2 * i, 2 * j] = 4/16
        # Ensure indices are constructed in order.
        # Assumes row-major ordering.
        if 0 <= 2 * i - 1:
            if 0 <= 2 * j - 1:
                # top-left
                Rj[p], Rx[p] = flatten(2 * i - 1, 2 * j - 1), 1 / 16
                p += 1
            # top-middle
            Rj[p], Rx[p] = flatten(2 * i - 1, 2 * j), 2 / 16
            p += 1
            if 2 * j + 1 < fine_dim:
                # top-right
                Rj[p], Rx[p] = flatten(2 * i - 1, 2 * j + 1), 1 / 16
                p += 1
        if 0 <= 2 * j - 1:
            # middle-left
            Rj[p], Rx[p] = flatten(2 * i, 2 * j - 1), 2 / 16
            p += 1
        # middle-middle
        Rj[p], Rx[p] = flatten(2 * i, 2 * j), 4 / 16
        p += 1
        if 2 * j + 1 < fine_dim:
            # middle-right
            Rj[p], Rx[p] = flatten(2 * i, 2 * j + 1), 2 / 16
            p += 1
        if 2 * i + 1 < fine_dim:
            if 0 <= 2 * j - 1:
                # bottom-left
                Rj[p], Rx[p] = flatten(2 * i + 1, 2 * j - 1), 1 / 16
                p += 1
            # bottom-middle
            Rj[p], Rx[p] = flatten(2 * i + 1, 2 * j), 2 / 16
            p += 1
            if 2 * j + 1 < fine_dim:
                # bottom-right
                Rj[p], Rx[p] = flatten(2 * i + 1, 2 * j + 1), 1 / 16
                p += 1

    Rp[coarse_dim] = p
    Rx, Rj, Rp = np.array(Rx[:p]), np.array(Rj[:p]), np.array(Rp)
    R = sparse.csr_matrix((Rx[:p], Rj[:p], Rp), shape=(coarse_dim, fine_dim))
    return R, coarse_dim


def required_driver_memory(N):
    NN = N * N
    fine_shape = (int(np.sqrt(NN)),) * 2
    coarse_shape = fine_shape[0] // 2, fine_shape[1] // 2
    coarse_dim = numpy.prod(coarse_shape)
    nnz = 9 * coarse_dim
    elements = nnz + coarse_dim + 1
    bytes = elements * 8
    mb = bytes / 10**6
    print("Max required driver memory for N=%d is %fMB" % (N, mb))


def print_diagnostics(operators):
    """Print basic statistics about the multigrid hierarchy."""
    output = "MultilevelSolver\n"
    output += f"Number of Levels:     {len(operators)}\n"
    # output += f"Operator Complexity: {operator_complexity(levels):6.3f}\n"
    # output += f"Grid Complexity:     {grid_complexity(levels):6.3f}\n"

    total_nnz = sum(level[1].nnz for level in operators)

    #          123456712345678901 123456789012 123456789
    #               0       10000        49600 [52.88%]
    output += "  level   unknowns     nonzeros\n"
    for n, level in enumerate(operators):
        A = level[1]
        ratio = 100 * A.nnz / total_nnz
        output += f"{n:>6} {A.shape[1]:>11} {A.nnz:>12} [{ratio:2.2f}%]\n"

    print(output)


def execute(N, data, smoother, gridop, levels, maxiter, tol, verbose, warmup, timer):
    build, solve = get_phase_procs(use_legate)

    if warmup:
        tA = diffusion2D(64, epsilon=0.1, theta=np.pi / 4)
        tB = tA.T
        tC = tB @ tA  # noqa: F841

    # Generate matrix
    timer.start()
    if data == "poisson":
        A = poisson2D(N)
        b = np.random.rand(N**2)
    elif data == "diffusion":
        A = diffusion2D(N)
        b = np.random.rand(N**2)
    else:
        raise NotImplementedError(data)
    print(f"GMG: {A.shape}")
    print(f"Data creation time: {timer.stop()} ms")

    assert smoother == "jacobi", "Only Jacobi smoother is currently supported."

    if verbose:

        def callback(x):
            print(f"Residual: {np.linalg.norm(b - (A @ x))}")

    else:
        callback = None

    required_driver_memory(N)
    # Setup
    timer.start()
    mg_solver = GMG(
        A=A,
        shape=(N, N),
        levels=levels,
        smoother=smoother,
        gridop=gridop,
        machine=solve,
    )
    M = mg_solver.linear_operator()
    print(f"GMG init time: {timer.stop()} ms")

    print_diagnostics(mg_solver.operators)

    # Warm up the runtime.
    float(
        np.linalg.norm(
            A.dot(
                np.zeros(
                    A.shape[1],
                )
            )
        )
    )
    float(
        np.linalg.norm(
            M.matvec(
                np.zeros(
                    M.shape[1],
                )
            )
        )
    )
    # Make another call to random here as well.
    float(np.linalg.norm(np.random.rand(b.shape[0])))

    # Solve
    timer.start()
    x, iters = linalg.cg(A, b, rtol=tol, maxiter=maxiter, M=M, callback=callback)
    total = timer.stop()

    norm_ini = np.linalg.norm(b)
    norm_res = np.linalg.norm(b - (A @ x))

    # Check convergence with relative tolerance
    if norm_res <= norm_ini * tol:
        print(
            f"Converged in {iters} iterations, final residual relative norm:"
            f" {norm_res/norm_ini}"  # noqa: E226
        )
    else:
        print(
            f"Failed to converge in {iters} iterations, final residual relative norm:"
            f" {norm_res/norm_ini}"  # noqa: E226
        )

    print(f"Solve Time: {total} ms")
    print(f"Iteration time: {total / iters} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=16,
        dest="N",
        help="number of elements in one dimension",
    )
    parser.add_argument(
        "-d",
        "--data",
        dest="data",
        choices=["poisson", "diffusion"],
        type=str,
        default="poisson",
        help="The problem instance to solve.",
    )
    parser.add_argument(
        "-s",
        "--smoother",
        dest="smoother",
        choices=["jacobi"],
        type=str,
        default="jacobi",
        help="Smoother to use.",
    )
    parser.add_argument(
        "-g",
        "--gridop",
        dest="gridop",
        choices=["linear", "injection"],
        type=str,
        default="injection",
        help="Intergrid transfer operator to use.",
    )
    parser.add_argument(
        "-l",
        "--levels",
        dest="levels",
        type=int,
        default=2,
        help="Number of multigrid levels.",
    )
    parser.add_argument(
        "-m",
        "--maxiter",
        type=int,
        default=200,
        dest="maxiter",
        help="bound the maximum number of iterations",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="print verbose output",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-10,
        dest="tol",
        help="Convergence relative norm check threshold",
    )

    parser.add_argument(
        "-w",
        "--warmup",
        dest="warmup",
        action="store_true",
        help="Perform some Warmup operations before running timings",
    )

    args, _ = parser.parse_known_args()
    _, timer, np, sparse, linalg, use_legate = parse_common_args()
    execute(**vars(args), timer=timer)
