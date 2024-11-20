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
import scipy.io as sci_io
from utils.common import test_mtx_files

# import legate_sparse as sparse
import legate_sparse.io as legate_io


@pytest.mark.parametrize("filename", test_mtx_files)
def test_mmread(filename):
    arr = legate_io.mmread(filename)
    s = sci_io.mmread(filename)
    assert np.array_equal(arr.todense(), s.todense())


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
