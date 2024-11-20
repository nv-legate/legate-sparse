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

import numpy as np
from legate.core import track_provenance  # type: ignore[attr-defined]
from legate.core import types  # type: ignore[attr-defined]

from .config import SparseOpCode
from .csr import csr_array
from .runtime import runtime
from .types import coord_ty, float64, nnz_ty
from .utils import store_to_cupynumeric_array


@track_provenance(runtime.sparse_library)
def mmread(source):
    # TODO (rohany): We'll assume for now that all of the nodes in the system
    # can access the file passed in, so we don't need to worry about where this
    # task gets mapped to.
    rows = runtime.create_store(coord_ty, ndim=1)
    cols = runtime.create_store(coord_ty, ndim=1)
    vals = runtime.create_store(float64, ndim=1)
    m = runtime.create_store(coord_ty, optimize_scalar=True, shape=(1,))
    n = runtime.create_store(coord_ty, optimize_scalar=True, shape=(1,))
    nnz = runtime.create_store(nnz_ty, optimize_scalar=True, shape=(1,))
    task = runtime.create_auto_task(SparseOpCode.READ_MTX_TO_COO)
    task.add_output(m)
    task.add_output(n)
    task.add_output(nnz)
    task.add_output(rows)
    task.add_output(cols)
    task.add_output(vals)
    task.add_scalar_arg(source, types.string_type)
    task.execute()

    m = int(np.asarray(m.get_physical_store().get_inline_allocation())[0])
    n = int(np.asarray(n.get_physical_store().get_inline_allocation())[0])
    nnz = int(np.asarray(nnz.get_physical_store().get_inline_allocation())[0])
    # Slice down each store from the resulting size into the actual size.
    sl = slice(0, nnz)
    rows = store_to_cupynumeric_array(rows.slice(0, sl))
    cols = store_to_cupynumeric_array(cols.slice(0, sl))
    vals = store_to_cupynumeric_array(vals.slice(0, sl))
    return csr_array((vals, (rows, cols)), shape=(m, n))
