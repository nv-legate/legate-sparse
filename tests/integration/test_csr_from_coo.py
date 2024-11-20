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
import numpy
import pytest
from utils.sample import simple_system_gen

import legate_sparse as sparse


@pytest.mark.parametrize("N", [7, 13])
@pytest.mark.parametrize("M", [5, 29])
def test_csr_from_coo(N, M):
    shape = (N, M)

    np.random.seed(0)

    # This can generate duplicates nnz
    # nnz = N*M // 2
    # row_ind = np.random.random_integers(0, high=(N-1), size=nnz)
    # col_ind = np.random.random_integers(0, high=(M-1), size=nnz)
    # vals = np.random.rand(nnz)

    # so we just extract sparsity from dense matrix
    A_dense_orig, _, _ = simple_system_gen(N, M, sparse.csr_array)
    nnzs = np.argwhere(A_dense_orig > 0.0)
    vals = A_dense_orig.ravel()
    vals = vals[vals > 0.0]

    row_ind, col_ind = nnzs[:, 0], nnzs[:, 1]

    # we want test on unsorted inputs
    perm = np.array(numpy.random.permutation(numpy.arange(row_ind.shape[0])))
    row_ind = row_ind[perm]
    col_ind = col_ind[perm]

    A = sparse.csr_array((vals, (row_ind, col_ind)), shape=shape)

    A_dense = np.zeros(shape=shape)
    for r, c, v in zip(row_ind, col_ind, vals):
        A_dense[r, c] = v

    assert np.all(np.isclose(A_dense, A.todense()))


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
