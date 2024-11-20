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

import sys

import cupynumeric as np
import pytest
from legate.core import get_legate_runtime

import legate_sparse as sparse


def test_csr_from_csr_fixed():
    """
    2 0 0 0 1 0
    5 8 0 0 0 2
    0 0 3 4 0 0
    0 6 0 1 0 0
    9 0 0 0 4 0
    7 0 0 0 2 1
    """
    row_offsets = np.array([0, 2, 5, 7, 9, 11, 14], dtype=np.int64)
    csr_vals = np.array([2, 1, 5, 8, 2, 3, 4, 6, 1, 9, 4, 7, 2, 1], dtype=np.float64)
    col_indices = np.array([0, 4, 0, 1, 5, 2, 3, 1, 3, 0, 4, 0, 4, 5], dtype=np.int64)
    matrix_shape = (6, 6)

    A = sparse.csr_array(  # noqa: F841
        (csr_vals, col_indices, row_offsets), shape=matrix_shape
    )

    get_legate_runtime().issue_execution_fence(block=True)


@pytest.mark.parametrize("N", [7, 13])
@pytest.mark.parametrize("M", [5, 29])
def test_csr_from_csr_gen(N, M):
    nnz_per_row = np.random.randint(M, size=N)
    row_offsets = np.append([0], np.cumsum(nnz_per_row))
    nnz = row_offsets[-1]
    col_indices = np.random.randint(M, size=nnz)
    csr_vals = np.random.rand(nnz)
    matrix_shape = (N, M)

    A = sparse.csr_array(  # noqa: F841
        (csr_vals, col_indices, row_offsets), shape=matrix_shape
    )


@pytest.mark.parametrize("N", [7, 13])
@pytest.mark.parametrize("M", [5, 29])
def test_csr_from_empty(N, M):
    A = sparse.csr_array((N, M), dtype=np.float64)  # noqa: F841


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
