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

import legate_sparse as sparse


def test_csr_to_dense():
    row_offsets = np.array([0, 2, 5, 7, 9, 11, 14], dtype=np.int64)
    csr_vals = np.array([2, 1, 5, 8, 2, 3, 4, 6, 1, 9, 4, 7, 2, 1], dtype=np.float64)
    col_indices = np.array([0, 4, 0, 1, 5, 2, 3, 1, 3, 0, 4, 0, 4, 5], dtype=np.int64)
    matrix_shape = (6, 6)

    A = sparse.csr_array((csr_vals, col_indices, row_offsets), shape=matrix_shape)

    B = A.todense()
    expected_B = np.array(
        [
            [2, 0, 0, 0, 1, 0],
            [5, 8, 0, 0, 0, 2],
            [0, 0, 3, 4, 0, 0],
            [0, 6, 0, 1, 0, 0],
            [9, 0, 0, 0, 4, 0],
            [7, 0, 0, 0, 2, 1],
        ],
        dtype=np.float64,
    )

    assert (B == expected_B).all()


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
