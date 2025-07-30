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

import numpy

# Define some common types. Hopefully as we make more
# progress in generalizing the compute kernels, we can
# remove this code.
coord_ty = numpy.dtype(numpy.int64)
"""Data type for coordinate indices in sparse matrices (int64)."""

nnz_ty = numpy.dtype(numpy.uint64)
"""Data type for non-zero counts in sparse matrices (uint64)."""

float64 = numpy.dtype(numpy.float64)
"""64-bit floating point data type."""

int32 = numpy.dtype(numpy.int32)
"""32-bit integer data type."""

int64 = numpy.dtype(numpy.int64)
"""64-bit integer data type."""

uint64 = numpy.dtype(numpy.uint64)
"""64-bit unsigned integer data type."""
