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
from __future__ import annotations

import numpy as np
from legate.core import track_provenance, types

from .config import SparseOpCode
from .csr import csr_array
from .runtime import runtime
from .types import coord_ty, float64, nnz_ty
from .utils import store_to_cupynumeric_array


@track_provenance()
def mmread(source: str) -> csr_array:
    """Read a sparse matrix from a Matrix Market (.mtx) file.

    Parameters
    ----------
    source : str
        The filename or path to the Matrix Market file to read.

    Returns
    -------
    csr_array
        A sparse matrix in CSR format loaded from the file.

    Notes
    -----
    This function reads Matrix Market format files and converts them
    to CSR format. The Matrix Market format is a standard format for
    storing sparse matrices. For more information on the format, see
    https://math.nist.gov/MatrixMarket/formats.html.

    The function assumes that all nodes in the system can access the
    file, so no special file distribution is needed.

    The implementation reads the file in COO format and then converts
    to CSR format for efficient storage and operations.

    Examples
    --------
    >>> from legate_sparse import mmread
    >>> A = mmread("matrix.mtx")
    >>> print(A.shape)
    (1000, 1000)
    """
    # TODO (rohany): We'll assume for now that all of the nodes in the system
    # can access the file passed in, so we don't need to worry about where this
    # task gets mapped to.
    rows_store = runtime.create_store(coord_ty, ndim=1)
    cols_store = runtime.create_store(coord_ty, ndim=1)
    vals_store = runtime.create_store(float64, ndim=1)
    m_store = runtime.create_store(coord_ty, optimize_scalar=True, shape=(1,))
    n_store = runtime.create_store(coord_ty, optimize_scalar=True, shape=(1,))
    nnz_store = runtime.create_store(nnz_ty, optimize_scalar=True, shape=(1,))
    task = runtime.create_auto_task(SparseOpCode.READ_MTX_TO_COO)
    task.add_output(m_store)
    task.add_output(n_store)
    task.add_output(nnz_store)
    task.add_output(rows_store)
    task.add_output(cols_store)
    task.add_output(vals_store)
    task.add_scalar_arg(source, types.string_type)
    task.execute()

    m = int(
        np.asarray(m_store.get_physical_store().get_inline_allocation())[0]
    )
    n = int(
        np.asarray(n_store.get_physical_store().get_inline_allocation())[0]
    )
    nnz = int(
        np.asarray(nnz_store.get_physical_store().get_inline_allocation())[0]
    )
    # Slice down each store from the resulting size into the actual size.
    sl = slice(0, nnz)
    rows = store_to_cupynumeric_array(rows_store.slice(0, sl))
    cols = store_to_cupynumeric_array(cols_store.slice(0, sl))
    vals = store_to_cupynumeric_array(vals_store.slice(0, sl))
    return csr_array((vals, (rows, cols)), shape=(m, n))
