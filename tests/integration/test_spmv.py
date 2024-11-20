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
from utils.banded_matrix import banded_matrix
from utils.sample import simple_system_gen

import legate_sparse as sparse


@pytest.mark.parametrize("N", [5, 29])
@pytest.mark.parametrize("M", [7, 17])
@pytest.mark.parametrize("inline", [True, False])
def test_csr_spmv(N, M, inline):
    np.random.seed(0)
    A_dense, A, x = simple_system_gen(N, M, sparse.csr_array)

    if inline:
        y = np.ndarray((N,))
        A.dot(x, out=y)
    else:
        y = A @ x

    assert np.all(np.isclose(y, A_dense @ x))


@pytest.mark.parametrize("N", [5, 29])
@pytest.mark.parametrize("nnz_per_row", [3, 9])
@pytest.mark.parametrize("unsupported_dtype", ["int", "bool"])
def test_csr_spmv_unsupported_dtype(N, nnz_per_row, unsupported_dtype):
    np.random.seed(0)

    A = banded_matrix(N, nnz_per_row).astype(unsupported_dtype)
    x = np.ndarray((N,))

    expected_exp = NotImplementedError
    with pytest.raises(expected_exp):
        y = A.dot(x)  # noqa: F841


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
