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
from utils.sample import simple_system_gen

import legate_sparse as sparse

# TODO: Enable "eq" after #209 is fixed
COMPARISON_OPS = [
    ("gt", lambda x, y: x > y),
    ("lt", lambda x, y: x < y),
    ("ge", lambda x, y: x >= y),
    ("le", lambda x, y: x <= y),
    # ("eq", lambda x, y: x == y),
    ("ne", lambda x, y: x != y),
]


@pytest.mark.parametrize("N", [8, 13])
@pytest.mark.parametrize("threshold", [0.3, 0.5])
@pytest.mark.parametrize("op_name, op_func", COMPARISON_OPS)
def test_comparison_operation(N, threshold, op_name, op_func):
    """Test element-wise comparison operations on non-zero entries of the matrix

    Parameters
    ----------
    N : int
        Size of the test matrix
    threshold : float
        Value to compare against
    op_name : str
        Name of the comparison operation
    op_func : callable
        The comparison function to test
    """
    A_dense, A_sparse, _ = simple_system_gen(N, N, sparse.csr_array, tol=0.7)

    sparse_result = op_func(A_sparse, threshold)
    dense_result = op_func(A_dense[A_dense != 0], threshold)

    assert sparse_result.astype(int).sum() == dense_result.astype(int).sum()


@pytest.mark.parametrize("op_name, op_func", COMPARISON_OPS)
def test_comparison_error_cases(op_name, op_func):
    """Test error cases for comparison operations.

    Parameters
    ----------
    op_name : str
        Name of the comparison operation
    op_func : callable
        The comparison function to test
    """
    N = 8
    _, A_sparse, _ = simple_system_gen(N, N, sparse.csr_array, tol=0.7)

    # Test comparison with non-scalar values
    invalid_comparisons = [
        np.array([1, 2, 3]),  # 1D array
        np.array([[1, 2], [3, 4]]),  # 2D array
        "string",  # string
        [1, 2, 3],  # list
    ]

    for invalid_value in invalid_comparisons:
        with pytest.raises(AssertionError):
            op_func(A_sparse, invalid_value)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
