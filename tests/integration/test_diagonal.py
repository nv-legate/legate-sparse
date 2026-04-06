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
    """Test diagonal extraction from CSR matrices.

    This test verifies that the diagonal() method correctly extracts
    the main diagonal from CSR matrices, comparing results with dense
    matrix diagonal extraction.

    Parameters
    ----------
    N : int
        Size of the square matrix (N x N).
    with_zeros : bool
        Whether to include zeros on the diagonal (True) or ensure
        non-zero diagonal elements (False).

    Notes
    -----
    The test creates a random sparse matrix and optionally adds the
    identity matrix to ensure non-zero diagonal elements. It then
    extracts the diagonal using both the sparse matrix's diagonal()
    method and numpy's diagonal() function on the dense version.

    The test verifies that:
    1. The diagonal elements are extracted correctly
    2. The results match between sparse and dense implementations
    3. The method works for both sparse and dense diagonals

    This is important because diagonal extraction is a common operation
    in linear algebra and should work consistently across different
    matrix formats.
    """
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
