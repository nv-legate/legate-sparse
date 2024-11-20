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
import scipy.sparse as sp

import legate_sparse as sparse


@pytest.mark.parametrize("N", [12, 34])
@pytest.mark.parametrize("diagonals", [3, 5])
@pytest.mark.parametrize("dtype", (np.float32, np.float64, np.complex64, np.complex128))
@pytest.mark.parametrize("fmt", ["csr", "dia"])
def test_diags(N, diagonals, dtype, fmt):
    A = sparse.diags(
        [1] * diagonals,
        [x - (diagonals // 2) for x in range(diagonals)],
        shape=(N, N),
        format=fmt,
        dtype=dtype,
    )

    if fmt == "dia":
        A = A.tocsr()

    B = sp.diags(
        [1] * diagonals,
        [x - (diagonals // 2) for x in range(diagonals)],
        shape=(N, N),
        format=fmt,
        dtype=dtype,
    )

    assert np.array_equal(A.todense(), B.todense())


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
