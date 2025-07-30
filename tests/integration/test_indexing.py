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

import cupynumeric
import numpy
import pytest

from legate_sparse import csr_matrix


class TestIndexingSetItem:
    """Test class for sparse matrix indexing and assignment operations.

    This class contains tests for various indexing scenarios including
    boolean masking, derived masks, and edge cases for sparse matrix
    assignment operations.
    """

    @pytest.mark.parametrize("N", [6, 9, 17])
    def test_incompatible_mask(self, N, create_matrix, create_mask):
        """Test indexing with incompatible mask sparsity patterns.

        This test checks that the mask is applied correctly to the matrix when
        the sparsity of mask is different from that of the matrix.

        Parameters
        ----------
        N : int
            Size of the square matrix.
        create_matrix : fixture
            Fixture to create test matrices.
        create_mask : fixture
            Fixture to create boolean masks.

        Notes
        -----
        While SciPy will apply the mask to all entries, Legate Sparse will only
        apply the mask to the non-zero entries of the matrix, so we can't compare
        to SciPy results for all entries. Instead, we check that:
        1. The number of non-zero entries are updated correctly
        2. The values are updated correctly for masked positions

        This test verifies that the sparse implementation correctly handles
        cases where the mask has a different sparsity pattern than the matrix.
        """
        _, A = create_matrix(N)
        _, mask = create_mask(N)

        mask_dense = numpy.asarray(mask.todense())
        A_dense = numpy.asarray(A.todense())

        value = 10.0
        A[mask] = value

        vals = A.get_data()
        num_nonzeros = numpy.count_nonzero(A_dense[mask_dense])

        # make sure the number of entries are updated correctly
        num_updated = (vals == value).astype(int).sum()
        assert num_updated == num_nonzeros

        # make sure the values are updated correctly
        A_dense = numpy.asarray(A.todense())
        assert numpy.allclose(A_dense[mask_dense].sum() / num_nonzeros, value)

        # TODO: Add a check/test for location of nonzeros as well

    @pytest.mark.parametrize("N", [8, 13, 24])
    def test_mask_derived_from_self(self, N, create_matrix):
        """Test indexing with mask derived from the matrix itself.

        This test checks that the mask is applied correctly to the matrix when
        the sparsity of mask is derived from the matrix.

        Parameters
        ----------
        N : int
            Size of the square matrix.
        create_matrix : fixture
            Fixture to create test matrices.

        Notes
        -----
        Our behavior matches that of SciPy when the mask is derived from
        the matrix itself, so we can compare against SciPy results for all entries.

        The test creates a mask based on a threshold comparison (A > threshold)
        and verifies that both SciPy and Legate Sparse produce identical results.
        """
        A_scipy, A_sparse = create_matrix(N)
        threshold = 0.2
        value = 10.0

        # Legate operations
        mask_sparse = A_sparse > threshold
        A_sparse[mask_sparse] = value

        # SciPy operations
        mask_scipy = A_scipy > threshold
        A_scipy[mask_scipy] = value

        # Make sure scipy and legate sparse matrices are the same
        A_scipy_dense = numpy.asarray(A_scipy.todense())
        A_sparse_dense = numpy.asarray(A_sparse.todense())
        assert numpy.all(
            numpy.allclose(A_scipy_dense, A_sparse_dense, rtol=1e-5, atol=1e-6)
        )

    @pytest.mark.parametrize("N", [8, 13, 24])
    def test_mask_all_true(self, N, create_matrix):
        """Test indexing behavior with a mask that is all True.

        This test checks indexing behavior when using a mask that is all True.
        Every non-zero element should be updated to the new value.

        Parameters
        ----------
        N : int
            Size of the square matrix.
        create_matrix : fixture
            Fixture to create test matrices.

        Notes
        -----
        The test creates a mask with the same sparsity pattern as the matrix
        but with all True values. This should result in all non-zero elements
        being updated to the specified value.
        """
        _, A = create_matrix(N)
        value = 10.0

        # Create mask with same sparsity pattern as A_sparse but all True values
        mask_all_true = A.copy()
        mask_all_true.data = numpy.ones(A.nnz, dtype=bool)

        A[mask_all_true] = value

        # All non-zero elements should be updated to value
        assert numpy.all(A.get_data() == value)

    @pytest.mark.parametrize("N", [8, 13, 24])
    def test_mask_all_false(self, N, create_matrix, create_mask):
        """Test indexing behavior with a mask that is all False.

        This test checks indexing behavior when using a mask that is all False.
        No elements should be modified.

        Parameters
        ----------
        N : int
            Size of the square matrix.
        create_matrix : fixture
            Fixture to create test matrices.
        create_mask : fixture
            Fixture to create boolean masks.

        Notes
        -----
        The test creates a mask with density=0 (all False values) and verifies
        that the matrix remains unchanged after the assignment operation.
        """
        _, A = create_matrix(N)
        _, mask_all_false = create_mask(N, density=0)
        A_copy = A.copy()

        value = 10.0
        A[mask_all_false] = value

        # # Matrix should remain unchanged
        assert numpy.all(A_copy.get_data() == A.get_data())

    def test_random_column_order(self):
        """Test indexing with randomly ordered column indices.

        This test verifies that indexing works correctly even when the
        column indices are not in sorted order within each row.

        Notes
        -----
        The test creates a matrix with randomly ordered column indices
        within rows. During instantiation, these indices get sorted to
        ensure proper indexing behavior.

        The test verifies that:
        1. The matrix is created correctly despite random column ordering
        2. Boolean indexing operations work correctly
        3. The number of elements replaced matches the expected count

        This is important because CSR format requires column indices to be
        sorted within each row for efficient operations.
        """
        row_indices = cupynumeric.array(
            [
                2,
                4,
                5,
                3,
                5,
                1,
                1,
                5,
                5,
            ]
        )
        col_indices = cupynumeric.array(
            [
                3,
                1,
                2,
                2,
                5,
                1,
                4,
                1,
                3,
            ]
        )
        data = cupynumeric.array([7.0, 9.0, 3.0, 4.0, 5.0, 19.0, 2.0, 99.0, 109.0])

        # note that the data in row 5 is ordered (2, 5, 1, 3),which will get
        # sorted to (1, 2, 5, 3) during instantiation, which is needed for indexing
        # to work correctly

        A_sparse = csr_matrix((data, (row_indices, col_indices)), shape=(6, 6))

        mask = A_sparse > 18.0
        A_sparse[mask] = 10.0

        data_sparse = A_sparse.get_data()
        num_replaced_sparse = (data_sparse == 10.0).sum()
        num_replaced_numpy = (data > 18.0).sum()

        # make sure the number of elements that needed to be replaced
        # in the data array gets replaced in the sparse matrix
        assert num_replaced_sparse == num_replaced_numpy


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
