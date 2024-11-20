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
from utils.sample import simple_system_gen

import legate_sparse as sparse


@pytest.mark.parametrize("N", [7, 13])
@pytest.mark.parametrize("M", [5, 29])
def test_csr_from_csr(N, M):
    np.random.seed(0)
    A_dense, A, _ = simple_system_gen(N, M, sparse.csr_array)

    get_legate_runtime().issue_execution_fence(block=True)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
