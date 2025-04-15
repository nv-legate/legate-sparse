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
import numpy
import pytest
from utils.sample import simple_system_gen

import legate_sparse as sparse


@pytest.mark.parametrize("N", [8, 13])
def test_nonzero(N):
    """
    This test checks that the nonzero method returns the correct indices for a sparse matrix.
    """
    np.random.seed(0)
    A_dense, _, _ = simple_system_gen(N, N, None, tol=0.2)

    r_numpy, c_numpy = numpy.nonzero(A_dense)

    A = sparse.csr_array(A_dense)
    r_scipy, c_scipy = A.nonzero()

    assert np.all(r_numpy == r_scipy)
    assert np.all(c_numpy == c_scipy)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
