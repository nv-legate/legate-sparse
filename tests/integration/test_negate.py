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

"""Tests for sparse matrix negation (__neg__)."""

import sys

import cupynumeric as np
import pytest
from utils.sample import simple_system_gen

import legate_sparse as sparse


def test_negate():
    """-A returns a sparse matrix with negated values."""
    N = 15
    np.random.seed(42)
    A_dense, A_sparse, _ = simple_system_gen(N, N, sparse.csr_array, tol=0.3)

    C = -A_sparse

    assert np.allclose(C.todense(), -A_dense)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
