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

import legate_sparse.linalg as sparse_linalg


@pytest.mark.parametrize("y", [[2.0, 3.0]])
@pytest.mark.parametrize("x", [[0.0, 1.0]])
@pytest.mark.parametrize("a", [[2.0]])
@pytest.mark.parametrize("b", [[3.0]])
@pytest.mark.parametrize("isalpha", [True, False])
@pytest.mark.parametrize("negate", [True, False])
def test_cg_linalg(y, x, a, b, isalpha, negate):
    scalar = a[0] / b[0]
    if negate:
        scalar = -scalar
    alpha = scalar if isalpha else 1.0
    beta = 1.0 if isalpha else scalar
    expected_y = alpha * np.asarray(x) + beta * np.asarray(y)

    y = np.array(y)
    x = np.array(x)
    a = np.array(a)
    b = np.array(b)

    sparse_linalg.cg_axpby(y, x, a, b, isalpha=isalpha, negate=negate)

    assert np.allclose(expected_y, y)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
