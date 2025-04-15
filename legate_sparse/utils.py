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

import math
import traceback
from typing import Any

import cupynumeric
import numpy
from legate.core import LogicalStore

import legate_sparse

from .runtime import runtime

# Datatypes that spmv and spgemm operations are supported for
SUPPORTED_DATATYPES = (
    numpy.float32,
    numpy.float64,
    numpy.complex64,
    numpy.complex128,
)


# find_last_user_stacklevel gets the last stack frame index
# within legate sparse.
def find_last_user_stacklevel() -> int:
    stacklevel = 1
    for frame, _ in traceback.walk_stack(None):
        if not frame.f_globals["__name__"].startswith("sparse"):
            break
        stacklevel += 1
    return stacklevel


# store_to_cupynumeric_array converts a store to a cuPyNumeric array.
def store_to_cupynumeric_array(store: LogicalStore):
    return cupynumeric.asarray(store)


# get_store_from_cupynumeric_array extracts a store from a cuPyNumeric array.
def get_store_from_cupynumeric_array(
    arr: cupynumeric.ndarray,
    copy=False,
) -> LogicalStore:
    if copy:
        # If requested to make a copy, do so.
        arr = cupynumeric.array(arr)

    data = arr.__legate_data_interface__["data"]
    array = data[next(iter(data))]
    store = array.data

    return store


# cast_to_store attempts to cast an arbitrary object into a store.
def cast_to_store(arr):
    if isinstance(arr, LogicalStore):
        return arr
    if isinstance(arr, numpy.ndarray):
        arr = cupynumeric.array(arr)
    if isinstance(arr, cupynumeric.ndarray):
        return get_store_from_cupynumeric_array(arr)
    raise NotImplementedError


# cast_arr attempts to cast an arbitrary object into a cupynumeric
# ndarray, with an optional desired type.
def cast_arr(arr, dtype=None):
    if isinstance(arr, LogicalStore):
        arr = store_to_cupynumeric_array(arr)
    elif not isinstance(arr, cupynumeric.ndarray):
        arr = cupynumeric.array(arr)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


# find_common_type performs a similar analysis to
# cupynumeric.ndarray.find_common_type to find a common type
# between all of the arguments.
def find_common_type(*args):
    array_types = list()
    scalar_types = list()
    for array in args:
        if legate_sparse.is_sparse_matrix(array):
            array_types.append(array.dtype)
        elif array.size == 1:
            scalar_types.append(array.dtype)
        else:
            array_types.append(array.dtype)
    return numpy.result_type(*array_types, *scalar_types)


# cast_to_common_type casts all arguments to the same common dtype.
def cast_to_common_type(*args):
    # Find a common type for all of the arguments.
    common_type = find_common_type(*args)
    # Cast each input to the common type. Ideally, if all of the
    # arguments are already the common type then this will
    # be a no-op.
    return tuple(arg.astype(common_type, copy=False) for arg in args)


# factor_int decomposes an integer into a close to square grid.
def factor_int(n):
    val = math.ceil(math.sqrt(n))
    val2 = int(n / val)
    while val2 * val != float(n):
        val -= 1
        val2 = int(n / val)
    return val, val2


# broadcast_store broadcasts a store to the desired input shape,
# or throws an error if the broadcast is not possible.
def broadcast_store(store: LogicalStore, shape: Any) -> LogicalStore:
    diff = len(shape) - store.ndim
    for dim in range(diff):
        store = store.promote(dim, shape[dim])
    for dim in range(len(shape)):
        if store.shape[dim] != shape[dim]:
            if store.shape[dim] != 1:
                raise ValueError(
                    f"Shape did not match along dimension {dim} "
                    "and the value is not equal to 1"
                )
            store = store.project(dim, 0).promote(dim, shape[dim])
    return store


def copy_store(store: LogicalStore) -> LogicalStore:
    res = runtime.create_store(store.type, store.shape)  # type: ignore
    runtime.legate_runtime.issue_copy(res, store)
    return res


def store_from_store_or_array(src, copy=False) -> LogicalStore:  # type: ignore
    "Get LogicalStore from a LogicalStore or array, potentially creating a copy"
    if isinstance(src, cupynumeric.ndarray):
        return get_store_from_cupynumeric_array(src, copy)
    elif isinstance(src, LogicalStore):
        return copy_store(src) if copy else src
    else:
        AssertionError("Wrong type for 'store_from_store_or_array()' utility")


def array_from_store_or_array(src, copy=False) -> cupynumeric.ndarray:  # type: ignore
    "Get array from a LogicalStore or array, potentially creating a copy"
    if isinstance(src, cupynumeric.ndarray):
        return src.copy() if copy else src
    elif isinstance(src, LogicalStore):
        return (
            store_to_cupynumeric_array(src).copy()
            if copy
            else store_to_cupynumeric_array(src)
        )
    else:
        AssertionError("Wrong type for 'array_from_store_or_array()' utility")
    # type: ignore


def get_storage_type(src):
    if isinstance(src, cupynumeric.ndarray):
        return src.dtype
    elif isinstance(src, LogicalStore):
        # there is legate.core to_core_dtype(), but here we need the opposite
        # doing via array now
        return cast_arr(src).dtype
    else:
        AssertionError("Wrong type for 'get_storage_type()' utility")
    # type: ignore


def is_dtype_supported(dtype: numpy.dtype) -> bool:
    """
    Does this datatype support spMV and spGEMM operations

    Parameters
    ----------
    dtype: np.dtype
        Input datatype to check if it supports spMV and spGEMM

    Returns
    -------
    valid: bool
        True if  dtype supports spMV and spGEMM
    """

    return dtype in SUPPORTED_DATATYPES


def is_dense(x) -> bool:
    """
    Is this object a dense cupynumeric array
    """
    return isinstance(x, cupynumeric.ndarray)


def is_scalar_like(x) -> bool:
    """
    Is this object a scalar like type
    """
    if isinstance(x, str):
        return False
    return cupynumeric.isscalar(x) or (is_dense(x) and x.ndim == 0)


def is_sparse(x) -> bool:
    """
    Is this object a legate sparse matrix
    """
    return legate_sparse.is_sparse_matrix(x)


def sort_by_rows_then_cols(rows: cupynumeric.ndarray, cols: cupynumeric.ndarray):
    """
    This function is a quick and dirty hack that does what np.lexsort does
    using argsort, but only for two keys.
    This is primarily used to to get the indices that we can use to sort data
    first by rows and then by columns

    Parameters
    ----------

    rows: cupynumeric.ndarray
        Indices of rows

    cols: cupynumeric.ndarray
        Indices of cols

    Returns
    -------
    sorted_indices:cupynumeric.ndarray
        Indices sorted by rows and then by columns, as given by numpy's lexsort
    """
    assert rows.size == cols.size

    # note that the lexsort reverses the order of key,
    # so this would be equivalent to np.lexsort((cols, rows))

    indices = cupynumeric.argsort(cols, kind="stable")
    order = cupynumeric.argsort(rows[indices], kind="stable")

    return indices[order]
