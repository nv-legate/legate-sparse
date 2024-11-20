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

# Portions of this file are also subject to the following license:
#
# Copyright (c) 2001-2002 Enthought, Inc. 2003-2022, SciPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided
# with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived
# from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Portions of this file are also subject to the following license:
# Copyright (c) 2015 Preferred Infrastructure, Inc.
# Copyright (c) 2015 Preferred Networks, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import cupynumeric
import numpy

from .dia import dia_array


def diags(diagonals, offsets=0, shape=None, format=None, dtype=None):
    """
    Construct a sparse matrix from diagonals.
    Parameters
    ----------
    diagonals : sequence of array_like
        Sequence of arrays containing the matrix diagonals,
        corresponding to `offsets`.
    offsets : sequence of int or an int, optional
        Diagonals to set:
          - k = 0  the main diagonal (default)
          - k > 0  the kth upper diagonal
          - k < 0  the kth lower diagonal
    shape : tuple of int, optional
        Shape of the result. If omitted, a square matrix large enough
        to contain the diagonals is returned.
    format : {"dia", "csr", "csc", "lil", ...}, optional
        Matrix format of the result. By default (format=None) an
        appropriate sparse matrix format is returned. This choice is
        subject to change.
    dtype : dtype, optional
        Data type of the matrix.
    See Also
    --------
    spdiags : construct matrix from diagonals
    Notes
    -----
    This function differs from `spdiags` in the way it handles
    off-diagonals.
    The result from `diags` is the sparse equivalent of::
        np.diag(diagonals[0], offsets[0])
        + ...
        + np.diag(diagonals[k], offsets[k])
    Repeated diagonal offsets are disallowed.
    .. versionadded:: 0.11
    Examples
    --------
    >>> from scipy.sparse import diags
    >>> diagonals = [[1, 2, 3, 4], [1, 2, 3], [1, 2]]
    >>> diags(diagonals, [0, -1, 2]).toarray()
    array([[1, 0, 1, 0],
           [1, 2, 0, 2],
           [0, 2, 3, 0],
           [0, 0, 3, 4]])
    Broadcasting of scalars is supported (but shape needs to be
    specified):
    >>> diags([1, -2, 1], [-1, 0, 1], shape=(4, 4)).toarray()
    array([[-2.,  1.,  0.,  0.],
           [ 1., -2.,  1.,  0.],
           [ 0.,  1., -2.,  1.],
           [ 0.,  0.,  1., -2.]])
    If only one diagonal is wanted (as in `numpy.diag`), the following
    works as well:
    >>> diags([1, 2, 3], 1).toarray()
    array([[ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  2.,  0.],
           [ 0.,  0.,  0.,  3.],
           [ 0.,  0.,  0.,  0.]])
    """
    # if offsets is not a sequence, assume that there's only one diagonal
    if numpy.isscalar(offsets):
        # now check that there's actually only one diagonal
        if len(diagonals) == 0 or numpy.isscalar(diagonals[0]):
            diagonals = [cupynumeric.atleast_1d(diagonals)]
        else:
            raise ValueError("Different number of diagonals and offsets.")
    else:
        diagonals = list(map(cupynumeric.atleast_1d, diagonals))

    # Basic check
    if len(diagonals) != len(offsets):
        raise ValueError("Different number of diagonals and offsets.")

    # Determine shape, if omitted
    if shape is None:
        m = len(diagonals[0]) + abs(int(offsets[0]))
        shape = (m, m)

    # Determine data type, if omitted
    if dtype is None:
        raise NotImplementedError

    if format is not None and format not in ["csr", "dia"]:
        raise NotImplementedError

    # Construct data array
    m, n = shape

    M = max([min(m + offset, n - offset) + max(0, offset) for offset in offsets])
    M = max(0, M)
    data_arr = cupynumeric.zeros((len(offsets), M), dtype=dtype)

    K = min(m, n)

    for j, diagonal in enumerate(diagonals):
        offset = int(offsets[j])
        k = max(0, offset)
        length = min(m + offset, n - offset, K)
        if length < 0:
            raise ValueError("Offset %d (index %d) out of bounds" % (offset, j))
        try:
            data_arr[j, k : k + length] = diagonal[..., :length]
        except ValueError as e:
            if len(diagonal) != length and len(diagonal) != 1:
                raise ValueError(
                    "Diagonal length (index %d: %d at offset %d) does not "
                    "agree with matrix size (%d, %d)."
                    % (j, len(diagonal), offset, m, n)
                ) from e
            raise

    # We importantly don't perform this conversion to cupynumeric (involving
    # an attach operation) until we're done indexing into the list. This
    # avoid a cupynumeric crash involving restrictions in attach in pde.py.
    offsets = cupynumeric.atleast_1d(offsets)
    dia = dia_array((data_arr, offsets), shape=(m, n), dtype=dtype)
    if format == "csr":
        return dia.tocsr()
    return dia
