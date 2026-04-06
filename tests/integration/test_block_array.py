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

"""Tests for block_array construction function."""

import sys

import cupynumeric as np
import pytest
import scipy.sparse as sp

import legate_sparse as sparse

# Temporary release unblock for a known cupynumeric runtime issue.
pytestmark = pytest.mark.skip(
    reason=(
        "Temporarily disabled for release unblock: "
        "https://github.com/nv-legate/cupynumeric/issues/1224"
    )
)


class TestBlockArray:
    """Tests for the block_array function."""

    def test_basic_2x2_blocks(self):
        """Test basic 2x2 block assembly."""
        A = sparse.csr_array(np.array([[1, 2], [3, 4]]))
        B = sparse.csr_array(np.array([[5, 6], [7, 8]]))
        C = sparse.csr_array(np.array([[9, 10], [11, 12]]))
        D = sparse.csr_array(np.array([[13, 14], [15, 16]]))

        result = sparse.block_array([[A, B], [C, D]])

        expected = np.array(
            [[1, 2, 5, 6], [3, 4, 7, 8], [9, 10, 13, 14], [11, 12, 15, 16]]
        )
        assert np.array_equal(result.todense(), expected)

    def test_with_none_blocks(self):
        """Test block assembly with None (zero) blocks."""
        A = sparse.csr_array(np.array([[1, 2], [3, 4]]))
        B = sparse.csr_array(np.array([[5, 6], [7, 8]]))

        result = sparse.block_array([[A, None], [None, B]])

        expected = np.array(
            [[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 5, 6], [0, 0, 7, 8]]
        )
        assert np.array_equal(result.todense(), expected)

    def test_rectangular_blocks(self):
        """Test with rectangular blocks."""
        A = sparse.csr_array(np.array([[1, 2, 3], [4, 5, 6]]))
        B = sparse.csr_array(np.array([[7], [8]]))

        result = sparse.block_array([[A, B]])

        expected = np.array([[1, 2, 3, 7], [4, 5, 6, 8]])
        assert np.array_equal(result.todense(), expected)

    def test_single_block(self):
        """Test with a single block."""
        A = sparse.csr_array(np.array([[1, 2], [3, 4]]))
        result = sparse.block_array([[A]])
        assert np.array_equal(result.todense(), A.todense())

    def test_dtype_inference(self):
        """Test that dtype is correctly inferred."""
        A = sparse.csr_array(np.array([[1.5, 2.5]]))
        B = sparse.csr_array(np.array([[3, 4]]))
        result = sparse.block_array([[A], [B]])
        assert result.dtype == np.float64

    def test_explicit_dtype(self):
        """Test explicit dtype specification."""
        A = sparse.csr_array(np.array([[1, 2]]))
        result = sparse.block_array([[A]], dtype=np.float32)
        assert result.dtype == np.float32

    def test_sparse_blocks(self):
        """Test with actual sparse blocks (blocks with zeros)."""
        # Create sparse matrices with actual zero patterns
        data_A = np.array([1, 0, 0, 2])
        A = sparse.csr_array(data_A.reshape(2, 2))

        data_B = np.array([0, 3, 4, 0])
        B = sparse.csr_array(data_B.reshape(2, 2))

        result = sparse.block_array([[A, B]])

        expected = np.array([[1, 0, 0, 3], [0, 2, 4, 0]])
        assert np.array_equal(result.todense(), expected)

    def test_matches_scipy(self):
        """Test that output matches SciPy's block_array."""
        np.random.seed(42)

        # Create random sparse blocks
        A_dense = np.random.rand(3, 4)
        B_dense = np.random.rand(3, 2)
        C_dense = np.random.rand(2, 4)
        D_dense = np.random.rand(2, 2)

        # SciPy version
        A_sp = sp.csr_array(A_dense)
        B_sp = sp.csr_array(B_dense)
        C_sp = sp.csr_array(C_dense)
        D_sp = sp.csr_array(D_dense)
        scipy_result = sp.block_array([[A_sp, B_sp], [C_sp, D_sp]]).todense()

        # Legate version
        A_lg = sparse.csr_array(A_dense)
        B_lg = sparse.csr_array(B_dense)
        C_lg = sparse.csr_array(C_dense)
        D_lg = sparse.csr_array(D_dense)
        legate_result = sparse.block_array(
            [[A_lg, B_lg], [C_lg, D_lg]]
        ).todense()

        assert np.allclose(legate_result, scipy_result)


class TestBlockArrayErrors:
    """Tests for block_array error handling."""

    def test_empty_blocks_raises(self):
        """Test that empty blocks raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            sparse.block_array([])

    def test_inconsistent_row_count_raises(self):
        """Test that inconsistent row counts raise ValueError."""
        A = sparse.csr_array(np.array([[1, 2], [3, 4]]))
        B = sparse.csr_array(np.array([[5, 6, 7]]))  # Only 1 row

        with pytest.raises(ValueError, match="rows"):
            sparse.block_array([[A, B]])

    def test_inconsistent_col_count_raises(self):
        """Test that inconsistent column counts raise ValueError."""
        A = sparse.csr_array(np.array([[1, 2], [3, 4]]))
        B = sparse.csr_array(np.array([[5], [6], [7]]))  # 3 rows, but 1 col
        C = sparse.csr_array(
            np.array([[8, 9]])
        )  # 1 row, but needs 3 cols below B

        with pytest.raises(ValueError):
            sparse.block_array([[A, B], [C, None]])

    def test_unsupported_format_raises(self):
        """Test that unsupported format raises ValueError."""
        A = sparse.csr_array(np.array([[1, 2]]))
        with pytest.raises(ValueError, match="csr"):
            sparse.block_array([[A]], format="coo")

    def test_non_csr_block_raises(self):
        """Test that non-CSR blocks raise TypeError."""
        A = sparse.csr_array(np.array([[1, 2]]))
        with pytest.raises(TypeError, match="csr_array"):
            sparse.block_array([[A, np.array([[3, 4]])]])


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
