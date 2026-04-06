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
from legate_sparse.runtime import runtime


@pytest.mark.parametrize("N", [5, 29])
def test_csr_spgemm(N):
    """Test sparse matrix-matrix multiplication with CSR matrices.

    This test verifies that sparse matrix-matrix multiplication works
    correctly for different matrix sizes.

    Parameters
    ----------
    N : int
        Size of the square matrices (N x N).

    Notes
    -----
    The test creates a random sparse matrix A and computes A @ A using
    the sparse implementation. It then compares the result with the
    dense matrix multiplication A_dense @ A_dense to verify correctness.

    The test uses different matrix sizes to ensure the implementation
    works correctly for both small and larger matrices.
    """
    np.random.seed(0)
    A_dense, A, _ = simple_system_gen(N, N, sparse.csr_array)

    B = A.copy()

    C = A @ B

    assert np.all(np.isclose(C.todense(), A_dense @ A_dense))


@pytest.mark.parametrize("N", [5, 29])
@pytest.mark.parametrize("unsupported_dtype", ["int", "bool"])
def test_csr_spgemm_unsupported_dtype(N, unsupported_dtype):
    """Test that unsupported datatypes raise appropriate exceptions for SpGEMM.

    This test verifies that sparse matrix-matrix multiplication
    properly handles unsupported datatypes by raising NotImplementedError
    when running on GPU.

    Parameters
    ----------
    N : int
        Size of the square matrices.
    unsupported_dtype : str
        Datatype that is not supported for SpGEMM operations.

    Notes
    -----
    The test creates banded matrices with unsupported datatypes and
    attempts to perform matrix-matrix multiplication. On GPU systems,
    this should raise NotImplementedError since only floating-point
    and complex datatypes are supported for SpGEMM.

    Currently supported datatypes are float32, float64, complex64,
    and complex128.
    """
    np.random.seed(0)

    nnz_per_row = 3
    A = banded_matrix(N, nnz_per_row).astype(unsupported_dtype)
    B = banded_matrix(N, nnz_per_row).astype(unsupported_dtype)

    if runtime.num_gpus > 0:
        expected_exp = NotImplementedError
        with pytest.raises(expected_exp):
            C = A @ B  # noqa: F841


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
