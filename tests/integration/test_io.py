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

from pathlib import Path

import cupynumeric as np
import pytest
import scipy.io as sci_io

import legate_sparse.io as legate_io

TEST_DIR = Path(__file__).parent.parent


@pytest.fixture
def test_mtx_files():
    """Fixture providing paths to test Matrix Market files.

    This fixture returns a list of paths to various Matrix Market (.mtx)
    files that are used for testing the mmread functionality.

    Returns
    -------
    list
        List of file paths to test Matrix Market files.

    Notes
    -----
    The fixture includes various types of matrices:
    - test.mtx: Basic test matrix
    - GlossGT.mtx: Graph theory matrix
    - Ragusa18.mtx: Scientific computing matrix
    - cage4.mtx: Graph matrix
    - karate.mtx: Social network matrix

    These files are located in the testdata directory and provide
    different sparsity patterns and matrix properties for comprehensive
    testing of the Matrix Market reader.
    """
    mtx_files = [
        "test.mtx",
        "GlossGT.mtx",
        "Ragusa18.mtx",
        "cage4.mtx",
        "karate.mtx",
    ]
    return [str(TEST_DIR / "testdata" / mtx_file) for mtx_file in mtx_files]


def test_mmread(test_mtx_files):
    """Test Matrix Market file reading functionality.

    This test verifies that the legate_sparse Matrix Market reader
    produces the same results as SciPy's mmread function.

    Parameters
    ----------
    test_mtx_files : list
        List of Matrix Market file paths to test.

    Notes
    -----
    The test reads each Matrix Market file using both legate_sparse.io.mmread
    and scipy.io.mmread, then compares the results by converting both to
    dense format and checking for equality.

    This ensures that:
    1. The Matrix Market format is parsed correctly
    2. The sparse matrix structure is preserved
    3. The numerical values are read accurately
    4. The implementation is compatible with SciPy's reference implementation

    The test covers various matrix types and sizes to ensure robust
    parsing of the Matrix Market format.
    """
    for mtx_file in test_mtx_files:
        arr = legate_io.mmread(mtx_file)
        s = sci_io.mmread(mtx_file)
        assert np.array_equal(arr.todense(), s.todense())


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
