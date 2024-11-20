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
def test_csr_spgemm(N):
    np.random.seed(0)
    A_dense, A, _ = simple_system_gen(N, N, sparse.csr_array)

    B = A.copy()

    C = A @ B

    assert np.all(np.isclose(C.todense(), A_dense @ A_dense))


@pytest.mark.parametrize("N", [5, 29])
@pytest.mark.parametrize("unsupported_dtype", ["int", "bool"])
def test_csr_spgemm_unsupported_dtype(N, unsupported_dtype):
    np.random.seed(0)

    nnz_per_row = 3
    A = banded_matrix(N, nnz_per_row).astype(unsupported_dtype)
    B = banded_matrix(N, nnz_per_row).astype(unsupported_dtype)

    expected_exp = NotImplementedError
    with pytest.raises(expected_exp):
        C = A @ B  # noqa: F841


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
