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

import cupynumeric as np
import pytest
from utils.sample import sample_dense, sample_dense_vector

import legate_sparse.linalg as linalg
from legate_sparse import csr_array


def test_gmres_solve():
    N, D = 1000, 1000
    seed = 471014
    A = sample_dense(N, D, 0.1, seed)
    A = 0.5 * (A + A.T)
    A = A + N * np.eye(N)
    A = csr_array(A)
    x = sample_dense_vector(D, 0.1, seed)

    y = A @ x
    assert np.allclose((A @ x), y)

    x_pred, iters = linalg.gmres(A, y, atol=1e-5, tol=1e-5, maxiter=300)
    assert np.allclose((A @ x_pred), y, atol=1e-8)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
