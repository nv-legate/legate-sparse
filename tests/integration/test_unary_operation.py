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


def test_unary_operation():
    row_offsets = np.array([0, 2, 5, 7, 9, 11, 14], dtype=np.int64)
    csr_vals = np.array([2, 1, 5, 8, 2, 3, 4, 6, 1, 9, 4, 7, 2, 1], dtype=np.float64)
    col_indices = np.array([0, 4, 0, 1, 5, 2, 3, 1, 3, 0, 4, 0, 4, 5], dtype=np.int64)
    matrix_shape = (6, 6)

    A = sparse.csr_array((csr_vals, col_indices, row_offsets), shape=matrix_shape)

    B = A * 2
    Bvalues = np.asarray(B.vals)
    expected_Bvalues = np.array(
        [4, 2, 10, 16, 4, 6, 8, 12, 2, 18, 8, 14, 4, 2], dtype=np.float64
    )
    assert (Bvalues == expected_Bvalues).all()

    C = A.multiply(3)
    Cvalues = np.asarray(C.vals)
    expected_Cvalues = np.array(
        [6, 3, 15, 24, 6, 9, 12, 18, 3, 27, 12, 21, 6, 3], dtype=np.float64
    )
    assert (Cvalues == expected_Cvalues).all()

    D = A.conj().conj()
    assert np.all(np.isclose(A.todense(), D.todense()))


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
