# Copyright 2024 NVIDIA Corporation
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
import importlib

import numpy
from typing_extensions import Protocol


def get_arg_number(arg):
    """Parse a string argument that may contain size suffixes.

    Parameters
    ----------
    arg : str
        String argument that may end with 'k', 'm', or 'g' for
        kilobytes, megabytes, or gigabytes respectively.

    Returns
    -------
    int
        The parsed number with appropriate multiplier applied.

    Examples
    --------
    >>> get_arg_number("1024")
    1024
    >>> get_arg_number("1k")
    1024
    >>> get_arg_number("1m")
    1048576
    >>> get_arg_number("1g")
    1073741824
    """
    multiplier = 1
    arg = arg.lower()
    if len(arg) == 0:
        return 1
    elif arg[-1] == "k":
        multiplier = 1024
        arg = arg[:-1]
    elif arg[-1] == "m":
        multiplier = 1024 * 1024
        arg = arg[:-1]
    elif arg[-1] == "g":
        multiplier = 1024 * 1024 * 1024
        arg = arg[:-1]

    return int(arg) * multiplier


class Timer(Protocol):
    """Protocol for timer implementations.

    This protocol defines the interface that timer classes must implement
    for measuring execution time in the examples.
    """

    def start(self):
        """Start timing."""
        ...

    def stop(self):
        """Stop timing and return duration.

        Blocks execution until everything before it has completed.

        Returns
        -------
        float
            Duration since the last call to start(), in milliseconds.
        """
        ...


class LegateTimer(Timer):
    """Timer implementation using Legate's timing facilities.

    This timer uses Legate's internal timing mechanism for accurate
    measurement of GPU operations.
    """

    def __init__(self):
        self._start = None

    def start(self):
        """Start timing using Legate's time function."""
        from legate.timing import time

        self._start = time()

    def stop(self):
        """Stop timing and return duration in milliseconds."""
        from legate.timing import time

        _end = time()
        return (_end - self._start) / 1000.0


class CuPyTimer(Timer):
    """Timer implementation using CuPy's CUDA events.

    This timer uses CUDA events for accurate measurement of GPU operations
    in CuPy applications.
    """

    def __init__(self):
        self._start_event = None

    def start(self):
        """Start timing using CUDA events."""
        from cupy import cuda

        self._start_event = cuda.Event()
        self._start_event.record()

    def stop(self):
        """Stop timing and return duration in milliseconds."""
        from cupy import cuda

        end_event = cuda.Event()
        end_event.record()
        end_event.synchronize()
        return cuda.get_elapsed_time(self._start_event, end_event)


class NumPyTimer(Timer):
    """Timer implementation using Python's high-resolution timer.

    This timer uses Python's perf_counter_ns for accurate measurement
    of CPU operations in NumPy/SciPy applications.
    """

    def __init__(self):
        self._start_time = None

    def start(self):
        """Start timing using perf_counter_ns."""
        from time import perf_counter_ns

        self._start_time = perf_counter_ns() / 1000.0

    def stop(self):
        """Stop timing and return duration in milliseconds."""
        from time import perf_counter_ns

        end_time = perf_counter_ns() / 1000.0
        return (end_time - self._start_time) / 1000.0


# DummyScope is a class that is a no-op context
# manager so that we can run both CuPy and SciPy
# programs with resource scoping.
class DummyScope:
    """No-op context manager for resource scoping.

    This class provides a dummy context manager that does nothing,
    allowing the same code to run with both CuPy and SciPy programs
    that may or may not use resource scoping.
    """

    def __init__(self): ...

    def __enter__(self):
        """Enter the context (no-op)."""
        ...

    def __exit__(self, _, __, ___):
        """Exit the context (no-op)."""
        ...

    def __getitem__(self, item):
        """Return self for any indexing (no-op)."""
        return self

    def count(self, _):
        """Return 1 for any count operation."""
        return 1

    @property
    def preferred_kind(self):
        """Return None for preferred kind."""
        return None


def get_phase_procs(use_legate: bool):
    """Get processor configurations for different phases of computation.

    Parameters
    ----------
    use_legate : bool
        Whether to use Legate-specific processor configuration.

    Returns
    -------
    tuple
        (build_procs, solve_procs) - processor configurations for
        build and solve phases respectively.

    Notes
    -----
    When use_legate is True, this function queries the available
    processors and assigns them to different phases:
    - Build phase: Prefers CPUs, then OpenMP processors, then GPUs
    - Solve phase: Prefers GPUs, then OpenMP processors, then CPUs

    When use_legate is False, returns DummyScope objects.
    """
    if use_legate:
        from legate.core import TaskTarget, get_machine

        all_devices = get_machine()
        num_gpus = all_devices.count(TaskTarget.GPU)
        num_omps = all_devices.count(TaskTarget.OMP)
        num_cpus = all_devices.count(TaskTarget.CPU)

        # Prefer CPUs for the "build" phase of applications.
        # NOTE: the runtime increases by about 35% if both CPUs
        # and GPUs are used, so use just GPUs for both until that is
        # debugged
        if num_omps > 0:
            build_procs = all_devices.only(TaskTarget.OMP)
        elif num_cpus > 0:
            build_procs = all_devices.only(TaskTarget.CPU)
        elif num_gpus > 0:
            build_procs = all_devices.only(TaskTarget.GPU)

        # Prefer GPUs for the "solve" phase of applications.
        if num_gpus > 0:
            solve_procs = all_devices.only(TaskTarget.GPU)
        elif num_omps > 0:
            solve_procs = all_devices.only(TaskTarget.OMP)
        else:
            solve_procs = all_devices.only(TaskTarget.CPU)

        print(f"build_procs: {build_procs}, solve_procs: {solve_procs}")
        return build_procs, solve_procs
    else:
        return DummyScope(), DummyScope()


def parse_common_args():
    """Parse common command line arguments for example scripts.

    Returns
    -------
    tuple
        (package, timer, np, sparse, linalg, use_legate) where:
        - package: str - the selected package ("legate", "cupy", or "scipy")
        - timer: Timer - appropriate timer implementation
        - np: module - numpy/cupy/cupynumeric module
        - sparse: module - sparse matrix module
        - linalg: module - linear algebra module
        - use_legate: bool - whether Legate is being used

    Notes
    -----
    This function sets up the global environment with the appropriate
    modules based on the --package argument. It supports:
    - "legate": Uses cupynumeric, legate_sparse, and legate_sparse.linalg
    - "cupy": Uses cupy, cupyx.scipy.sparse, and cupyx.scipy.sparse.linalg
    - "scipy": Uses numpy, scipy.sparse, and scipy.sparse.linalg
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package",
        type=str,
        default="legate",
        choices=["legate", "cupy", "scipy"],
    )
    args, _ = parser.parse_known_args()

    if args.package == "legate":
        timer = LegateTimer()
        np_name = "cupynumeric"
        sp_name = "legate_sparse"
        lg_name = "legate_sparse.linalg"

        use_legate = True

    elif args.package == "cupy":
        timer = CuPyTimer()
        np_name = "cupy"
        sp_name = "cupyx.scipy.sparse"
        lg_name = "cupyx.scipy.sparse.linalg"

        use_legate = False
    else:
        timer = NumPyTimer()
        np_name = "numpy"
        sp_name = "scipy.sparse"
        lg_name = "scipy.sparse.linalg"

        use_legate = False

    globals()["np"] = importlib.import_module(np_name)
    globals()["sparse"] = importlib.import_module(sp_name)
    globals()["linalg"] = importlib.import_module(lg_name)

    return args.package, timer, np, sparse, linalg, use_legate


# Constructs banded matrix with 1.0 as values
#
# `diags` construct csr from dia array, while when from_diags=False
# we construct csr arrya directly - might be slightly faster
def banded_matrix(N, nnz_per_row, from_diags=False):
    """Construct a banded matrix with 1.0 as values.

    Parameters
    ----------
    N : int
        Size of the square matrix (N x N).
    nnz_per_row : int
        Number of non-zeros per row. Must be odd.
    from_diags : bool, optional
        If True, construct using sparse.diags then convert to CSR.
        If False, construct CSR array directly. Default is False.

    Returns
    -------
    sparse matrix
        A banded matrix in CSR format with 1.0 values.

    Raises
    ------
    AssertionError
        If N <= nnz_per_row or nnz_per_row is not odd.

    Notes
    -----
    The matrix has a banded structure with nnz_per_row non-zeros per row,
    centered around the main diagonal. The direct CSR construction method
    (from_diags=False) may be slightly faster than the diags method.

    Examples
    --------
    >>> A = banded_matrix(5, 3)
    >>> print(A.toarray())
    [[1. 1. 0. 0. 0.]
     [1. 1. 1. 0. 0.]
     [0. 1. 1. 1. 0.]
     [0. 0. 1. 1. 1.]
     [0. 0. 0. 1. 1.]]
    """
    if from_diags:
        return sparse.diags(
            [1] * nnz_per_row,
            [x - (nnz_per_row // 2) for x in range(nnz_per_row)],
            shape=(N, N),
            format="csr",
            dtype=np.float64,
        )
    else:
        assert N > nnz_per_row
        assert nnz_per_row % 2 == 1
        half_nnz = nnz_per_row // 2

        pred_nrows = nnz_per_row - half_nnz
        post_nrows = pred_nrows
        main_rows = N - pred_nrows - post_nrows

        pred = np.arange(nnz_per_row - half_nnz, nnz_per_row + 1)
        post = np.flip(pred)
        nnz_arr = np.concatenate((pred, np.ones(main_rows) * nnz_per_row, post))
        row_offsets = np.zeros(N + 1).astype(sparse.coord_ty)
        row_offsets[1 : N + 1] = np.cumsum(nnz_arr)
        nnz = row_offsets[-1]

        col_indices = np.tile(
            np.arange(-half_nnz, nnz_per_row - half_nnz), (N,)
        ) + np.repeat(np.arange(N), nnz_per_row)
        data = np.ones(N * nnz_per_row).astype(np.float64)
        mask = col_indices >= 0
        mask &= col_indices < N

        col_indices = col_indices[mask]
        data = data[mask]
        assert data.shape[0] == nnz
        assert col_indices.shape[0] == nnz

        return sparse.csr_array(
            (data, col_indices.astype(np.int64), row_offsets.astype(np.int64)),
            shape=(N, N),
            copy=False,
        )


def stencil_grid(S, grid, dtype=None, format=None):
    """Construct a sparse matrix resulting from a stencil
    discretization on rectilinear grids.

    Parameters
    ----------
    S : array_like
        The stencil array defining the pattern of connections.
    grid : tuple
        Grid dimensions (e.g., (N, N) for 2D grid).
    dtype : dtype, optional
        Data type of the matrix. If None, uses S.dtype.
    format : str, optional
        Output format. If None, returns CSR format.

    Returns
    -------
    sparse matrix
        A sparse matrix in CSR format representing the stencil on the grid.

    Notes
    -----
    This function constructs a sparse matrix that represents the application
    of a stencil operator on a regular grid. The stencil defines the pattern
    of connections between grid points.

    The function handles:
    - Boundary conditions by zeroing connections outside the grid
    - Duplicate diagonals by summing their contributions
    - Conversion to CSR format for efficient operations

    Examples
    --------
    >>> # 5-point stencil for 2D grid
    >>> S = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    >>> A = stencil_grid(S, (3, 3))
    >>> print(A.toarray())
    """
    N_v = int(numpy.prod(grid))  # number of vertices in the mesh
    N_s = int((S != 0).sum(dtype=int))  # number of nonzero stencil entries

    # diagonal offsets
    diags = np.zeros(N_s, dtype=int)

    # compute index offset of each dof within the stencil
    strides = numpy.cumprod([1] + list(reversed(grid)))[:-1]
    indices = tuple(i.copy() for i in S.nonzero())
    for i, s in zip(indices, S.shape):
        i -= s // 2

    for stride, coords in zip(strides, reversed(indices)):
        diags += stride * coords

    data = np.repeat(S[S != 0], N_v).reshape((N_s, N_v))

    indices = np.vstack(indices).T

    # zero boundary connections
    for idx in range(indices.shape[0]):
        # We do this instead of
        #  for index, diag in zip(indices, data):
        # to avoid unnecessary materialization into numpy arrays.
        index = indices[idx, :]
        diag = data[idx, :]
        diag = diag.reshape(grid)
        for n, i in enumerate(index):
            if i > 0:
                s = [slice(None)] * len(grid)
                s[n] = slice(0, i)
                s = tuple(s)
                diag[s] = 0
            elif i < 0:
                s = [slice(None)] * len(grid)
                s[n] = slice(i, None)
                s = tuple(s)
                diag[s] = 0

    # remove diagonals that lie outside matrix
    mask = abs(diags) < N_v
    if not mask.all():
        diags = diags[mask]
        data = data[mask]

    # sum duplicate diagonals
    if len(np.unique(diags)) != len(diags):
        new_diags = np.unique(diags)
        new_data = np.zeros((len(new_diags), data.shape[1]), dtype=data.dtype)

        for dia, dat in zip(diags, data):
            n = np.searchsorted(new_diags, dia)
            new_data[n, :] += dat

        diags = new_diags
        data = new_data

    return sparse.dia_array((data, diags), shape=(N_v, N_v)).tocsr()


def poisson2D(N):
    """Construct the 2D Poisson matrix.

    Parameters
    ----------
    N : int
        Grid size (N x N grid).

    Returns
    -------
    sparse matrix
        The 2D Poisson matrix in CSR format.

    Notes
    -----
    This constructs the standard 5-point stencil discretization of
    the 2D Poisson equation -u_xx - u_yy = f on an N x N grid.

    The matrix has the following structure:
    - Main diagonal: 4.0
    - Off-diagonals: -1.0 for horizontal and vertical connections

    Examples
    --------
    >>> A = poisson2D(3)
    >>> print(A.toarray())
    [[ 4. -1.  0. -1.  0.  0.  0.  0.  0.]
     [-1.  4. -1.  0. -1.  0.  0.  0.  0.]
     [ 0. -1.  4.  0.  0. -1.  0.  0.  0.]
     [-1.  0.  0.  4. -1.  0. -1.  0.  0.]
     [ 0. -1.  0. -1.  4. -1.  0. -1.  0.]
     [ 0.  0. -1.  0. -1.  4.  0.  0. -1.]
     [ 0.  0.  0. -1.  0.  0.  4. -1.  0.]
     [ 0.  0.  0.  0. -1.  0. -1.  4. -1.]
     [ 0.  0.  0.  0.  0. -1.  0. -1.  4.]]
    """
    diag_size = N * N - 1
    first = np.full((N - 1), -1.0)
    chunks = np.concatenate([np.zeros(1), first])
    diag_a = np.concatenate([first, np.tile(chunks, (diag_size - (N - 1)) // N)])
    diag_g = -1.0 * np.ones(N * (N - 1))
    diag_c = 4.0 * np.ones(N * N)

    # We construct a sequence of main diagonal elements,
    diagonals = [diag_g, diag_a, diag_c, diag_a, diag_g]
    # and a sequence of positions of the diagonal entries relative to the main
    # diagonal.
    offsets = [-N, -1, 0, 1, N]

    return sparse.diags(diagonals, offsets, dtype=np.float64).tocsr()


def diffusion2D(N, epsilon=1.0, theta=0.0):
    """Construct a 2D diffusion matrix with anisotropy.

    Parameters
    ----------
    N : int
        Grid size (N x N grid).
    epsilon : float, optional
        Anisotropy parameter. Default is 1.0 (isotropic).
    theta : float, optional
        Rotation angle in radians. Default is 0.0.

    Returns
    -------
    sparse matrix
        The 2D diffusion matrix in CSR format.

    Notes
    -----
    This constructs a 9-point stencil for the anisotropic diffusion equation:
    -div(K * grad(u)) = f

    where K is a diffusion tensor that depends on epsilon and theta.
    The stencil coefficients are computed based on the rotated diffusion tensor.

    Examples
    --------
    >>> # Isotropic diffusion
    >>> A = diffusion2D(3, epsilon=1.0, theta=0.0)
    >>> # Anisotropic diffusion
    >>> A = diffusion2D(3, epsilon=0.1, theta=np.pi/4)
    """
    eps = float(epsilon)  # for brevity
    theta = float(theta)

    C = np.cos(theta)
    S = np.sin(theta)
    CS = C * S
    CC = C**2
    SS = S**2

    a = (-1 * eps - 1) * CC + (-1 * eps - 1) * SS + (3 * eps - 3) * CS
    b = (2 * eps - 4) * CC + (-4 * eps + 2) * SS
    c = (-1 * eps - 1) * CC + (-1 * eps - 1) * SS + (-3 * eps + 3) * CS
    d = (-4 * eps + 2) * CC + (2 * eps - 4) * SS
    e = (8 * eps + 8) * CC + (8 * eps + 8) * SS

    stencil = np.array([[a, b, c], [d, e, d], [c, b, a]]) / 6.0
    return stencil_grid(stencil, (N, N))
