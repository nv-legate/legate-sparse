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
    def start(self): ...

    def stop(self):
        """
        Blocks execution until everything before it has completed. Returns the
        duration since the last call to start(), in milliseconds.
        """
        ...


class LegateTimer(Timer):
    def __init__(self):
        self._start = None

    def start(self):
        from legate.timing import time

        self._start = time()

    # returns time in milliseconds
    def stop(self):
        from legate.timing import time

        _end = time()
        return (_end - self._start) / 1000.0


class CuPyTimer(Timer):
    def __init__(self):
        self._start_event = None

    def start(self):
        from cupy import cuda

        self._start_event = cuda.Event()
        self._start_event.record()

    def stop(self):
        from cupy import cuda

        end_event = cuda.Event()
        end_event.record()
        end_event.synchronize()
        return cuda.get_elapsed_time(self._start_event, end_event)


class NumPyTimer(Timer):
    def __init__(self):
        self._start_time = None

    def start(self):
        from time import perf_counter_ns

        self._start_time = perf_counter_ns() / 1000.0

    def stop(self):
        from time import perf_counter_ns

        end_time = perf_counter_ns() / 1000.0
        return (end_time - self._start_time) / 1000.0


# DummyScope is a class that is a no-op context
# manager so that we can run both CuPy and SciPy
# programs with resource scoping.
class DummyScope:
    def __init__(self): ...

    def __enter__(self): ...

    def __exit__(self, _, __, ___): ...

    def __getitem__(self, item):
        return self

    def count(self, _):
        return 1

    @property
    def preferred_kind(self):
        return None


def get_phase_procs(use_legate: bool):
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
    if from_diags:
        print("Banded matrices will be generated using sparse.diags API")
        return sparse.diags(
            [1] * nnz_per_row,
            [x - (nnz_per_row // 2) for x in range(nnz_per_row)],
            shape=(N, N),
            format="csr",
            dtype=np.float64,
        )
    else:
        print("Banded matrices will be manually generated")
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
