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
    mtx_files = [
        "test.mtx",
        "GlossGT.mtx",
        "Ragusa18.mtx",
        "cage4.mtx",
        "karate.mtx",
    ]
    return [str(TEST_DIR / "testdata" / mtx_file) for mtx_file in mtx_files]


def test_mmread(test_mtx_files):
    for mtx_file in test_mtx_files:
        arr = legate_io.mmread(mtx_file)
        s = sci_io.mmread(mtx_file)
        assert np.array_equal(arr.todense(), s.todense())


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
