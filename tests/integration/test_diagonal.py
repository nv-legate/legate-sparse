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
from utils.sample import simple_system_gen

from legate_sparse import csr_array


@pytest.mark.parametrize("N", [7, 13])
@pytest.mark.parametrize("with_zeros", [True, False])
def test_csr_diagonal(N, with_zeros):
    M = N
    np.random.seed(0)
    A_dense, _, _ = simple_system_gen(N, M, None, tol=0.2)

    if not with_zeros:
        A_dense += np.eye(N, M)

    A = csr_array(A_dense)
    dense_diag = np.diagonal(A_dense)
    csr_diag = A.diagonal()

    assert np.all(np.isclose(dense_diag, csr_diag))


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
