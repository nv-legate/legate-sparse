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
from __future__ import annotations

import math
import traceback
from typing import TYPE_CHECKING, cast

import cupynumeric as cn
import numpy as np
from legate.core import LogicalStore

import legate_sparse

from .runtime import runtime

if TYPE_CHECKING:
    from typing import Any

    import numpy.typing as npt

    from .csr import csr_array

# Datatypes that spmv and spgemm operations are supported for
SUPPORTED_DATATYPES = (np.float32, np.float64, np.complex64, np.complex128)
"""Supported datatypes for sparse matrix operations (SpMV and SpGEMM)."""


# find_last_user_stacklevel gets the last stack frame index
# within legate sparse.
def find_last_user_stacklevel() -> int:
    """Find the last stack frame index within legate sparse.

    Returns
    -------
    int
        The stack level of the last user code frame.

    Notes
    -----
    This function walks the stack to find the first frame that is not
    within the legate_sparse module, which is useful for determining
    the appropriate stack level for warnings.
    """
    stacklevel = 1
    for frame, _ in traceback.walk_stack(None):
        if not frame.f_globals["__name__"].startswith("sparse"):
            break
        stacklevel += 1
    return stacklevel


# store_to_cupynumeric_array converts a store to a cuPyNumeric array.
def store_to_cupynumeric_array(store: LogicalStore) -> cn.ndarray:
    """Convert a LogicalStore to a cupynumeric array.

    Parameters
    ----------
    store : LogicalStore
        The store to convert.

    Returns
    -------
    cupynumeric.ndarray
        The cupynumeric array representation of the store.
    """
    return cn.asarray(store)


# get_store_from_cupynumeric_array extracts a store from a cuPyNumeric array.
def get_store_from_cupynumeric_array(
    arr: cn.ndarray, copy: bool = False
) -> LogicalStore:
    """Extract a LogicalStore from a cupynumeric array.

    Parameters
    ----------
    arr : cupynumeric.ndarray
        The cupynumeric array to extract the store from.
    copy : bool, optional
        Whether to create a copy of the array first. Default is False.

    Returns
    -------
    LogicalStore
        The LogicalStore representation of the array.
    """
    if copy:
        # If requested to make a copy, do so.
        arr = cn.array(arr)

    data = arr.__legate_data_interface__["data"]
    array = data[next(iter(data))]
    store = array.data

    return cast(LogicalStore, store)


# cast_to_store attempts to cast an arbitrary object into a store.
def cast_to_store(arr: cn.ndarray | LogicalStore) -> LogicalStore:
    """Cast an arbitrary object to a LogicalStore.

    Parameters
    ----------
    arr : array_like or LogicalStore
        The object to cast.

    Returns
    -------
    LogicalStore
        The LogicalStore representation of the input.

    Raises
    ------
    NotImplementedError
        If the object cannot be cast to a LogicalStore.
    """
    if isinstance(arr, LogicalStore):
        return arr
    if isinstance(arr, np.ndarray):
        arr = cn.array(arr)
    if isinstance(arr, cn.ndarray):
        return get_store_from_cupynumeric_array(arr)
    raise NotImplementedError


# cast_arr attempts to cast an arbitrary object into a cupynumeric
# ndarray, with an optional desired type.
def cast_arr(
    arr: cn.ndarray | LogicalStore, dtype: npt.dtype[Any] | None = None
) -> cn.ndarray:
    """Cast an arbitrary object to a cupynumeric array.

    Parameters
    ----------
    arr : array_like or LogicalStore
        The object to cast.
    dtype : dtype, optional
        The desired data type. If None, preserves the original type.

    Returns
    -------
    cupynumeric.ndarray
        The cupynumeric array representation of the input.
    """
    if isinstance(arr, LogicalStore):
        arr = store_to_cupynumeric_array(arr)
    elif not isinstance(arr, cn.ndarray):
        arr = cn.array(arr)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def find_common_type(
    *args: cn.ndarray | csr_array | np.ndarray,
) -> npt.dtype[Any]:
    """Find the common data type for a set of arrays.

    This function performs a similar analysis to cupynumeric.ndarray.find_common_type
    to find a common type between all of the arguments.

    Parameters
    ----------
    *args : array_like
        Arrays to find the common type for.

    Returns
    -------
    numpy.dtype
        The common data type that can represent all input arrays.

    Notes
    -----
    The function handles sparse matrices, dense arrays, and scalars.
    For sparse matrices, it uses their dtype. For scalars (size == 1),
    they are treated separately from arrays.
    """
    array_types = list()
    scalar_types = list()
    for array in args:
        if legate_sparse.isspmatrix(array):
            array_types.append(array.dtype)
        elif array.size == 1:
            scalar_types.append(array.dtype)
        else:
            array_types.append(array.dtype)
    return np.result_type(*array_types, *scalar_types)


def factor_int(n: int) -> tuple[int, int]:
    """Split an integer into two close factors.

    Parameters
    ----------
    n : int
        The integer to factor.

    Returns
    -------
    tuple
        (val, val2) where val * val2 = n and val is close to sqrt(n).

    Notes
    -----
    This function finds two factors of n such that their product equals n
    and the first factor is close to the square root of n.
    """
    val = math.ceil(math.sqrt(n))
    val2 = int(n / val)
    while val2 * val != float(n):
        val -= 1
        val2 = int(n / val)
    return val, val2


def broadcast_store(
    store: LogicalStore, shape: tuple[int, ...]
) -> LogicalStore:
    """Broadcast a LogicalStore to the desired shape.

    Parameters
    ----------
    store : LogicalStore
        The store to broadcast.
    shape : tuple
        The target shape to broadcast to.

    Returns
    -------
    LogicalStore
        The broadcasted store.

    Raises
    ------
    ValueError
        If the broadcast is not possible.

    Notes
    -----
    This function handles both dimension promotion (adding new dimensions)
    and broadcasting (expanding dimensions of size 1).
    """
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
    """Create a copy of a LogicalStore.

    Parameters
    ----------
    store : LogicalStore
        The store to copy.

    Returns
    -------
    LogicalStore
        A new LogicalStore with the same data as the input.
    """
    res = runtime.create_store(store.type, store.shape)
    runtime.legate_runtime.issue_copy(res, store)
    return res


def store_from_store_or_array(
    src: LogicalStore | cn.ndarray, copy: bool = False
) -> LogicalStore:
    """Get LogicalStore from a LogicalStore or array, potentially creating a copy.

    Parameters
    ----------
    src : LogicalStore or cupynumeric.ndarray
        The source object to convert.
    copy : bool, optional
        Whether to create a copy. Default is False.

    Returns
    -------
    LogicalStore
        The LogicalStore representation of the input.

    Raises
    ------
    AssertionError
        If the input type is not supported.
    """
    if isinstance(src, cn.ndarray):
        return get_store_from_cupynumeric_array(src, copy)
    elif isinstance(src, LogicalStore):
        return copy_store(src) if copy else src
    else:
        raise AssertionError(
            "Wrong type for 'store_from_store_or_array()' utility"
        )


def array_from_store_or_array(
    src: LogicalStore | cn.ndarray, copy: bool = False
) -> cn.ndarray:
    """Get array from a LogicalStore or array, potentially creating a copy.

    Parameters
    ----------
    src : LogicalStore or cupynumeric.ndarray
        The source object to convert.
    copy : bool, optional
        Whether to create a copy. Default is False.

    Returns
    -------
    cupynumeric.ndarray
        The cupynumeric array representation of the input.

    Raises
    ------
    AssertionError
        If the input type is not supported.
    """
    if isinstance(src, cn.ndarray):
        return src.copy() if copy else src
    elif isinstance(src, LogicalStore):
        return (
            store_to_cupynumeric_array(src).copy()
            if copy
            else store_to_cupynumeric_array(src)
        )
    else:
        raise AssertionError(
            "Wrong type for 'array_from_store_or_array()' utility"
        )


def get_storage_type(src: LogicalStore | cn.ndarray) -> npt.dtype[Any]:
    """Get the storage type of an object.

    Parameters
    ----------
    src : LogicalStore or cupynumeric.ndarray
        The object to get the storage type for.

    Returns
    -------
    numpy.dtype
        The data type of the object.

    Raises
    ------
    AssertionError
        If the input type is not supported.
    """
    if isinstance(src, cn.ndarray):
        return src.dtype
    elif isinstance(src, LogicalStore):
        # there is legate.core to_core_dtype(), but here we need the opposite
        # doing via array now
        return cast_arr(src).dtype
    else:
        raise AssertionError("Wrong type for 'get_storage_type()' utility")


def is_dtype_supported(dtype: npt.dtype[Any]) -> bool:
    """Check if a datatype supports SpMV and SpGEMM operations.

    Parameters
    ----------
    dtype : numpy.dtype
        Input datatype to check if it supports SpMV and SpGEMM.

    Returns
    -------
    bool
        True if dtype supports SpMV and SpGEMM operations.

    Notes
    -----
    Currently supported datatypes are float32, float64, complex64, and complex128.
    """
    return dtype in SUPPORTED_DATATYPES


def is_dense(x: Any) -> bool:
    """Check if an object is a dense cupynumeric array.

    Parameters
    ----------
    x : object
        The object to check.

    Returns
    -------
    bool
        True if x is a cupynumeric.ndarray, False otherwise.
    """
    return isinstance(x, cn.ndarray)


def is_scalar_like(x: Any) -> bool:
    """Check if an object is a scalar-like type.

    Parameters
    ----------
    x : object
        The object to check.

    Returns
    -------
    bool
        True if x is a scalar or 0-dimensional array, False otherwise.

    Notes
    -----
    This function returns False for strings, even though they are scalar-like
    in some contexts, to avoid confusion with numeric scalars.
    """
    if isinstance(x, str):
        return False
    return cn.isscalar(x) or (is_dense(x) and x.ndim == 0)


def is_sparse(x: Any) -> bool:
    """Check if an object is a legate sparse matrix.

    Parameters
    ----------
    x : object
        The object to check.

    Returns
    -------
    bool
        True if x is a legate sparse matrix, False otherwise.
    """
    return legate_sparse.isspmatrix(x)


def sort_by_rows_then_cols(rows: cn.ndarray, cols: cn.ndarray) -> cn.ndarray:
    """Sort indices by rows first, then by columns.

    This function is a quick and dirty hack that does what np.lexsort does
    using argsort, but only for two keys. This is primarily used to get
    the indices that we can use to sort data first by rows and then by columns.

    Parameters
    ----------
    rows : cupynumeric.ndarray
        Indices of rows.
    cols : cupynumeric.ndarray
        Indices of columns.

    Returns
    -------
    cupynumeric.ndarray
        Indices sorted by rows and then by columns, as given by numpy's lexsort.

    Notes
    -----
    This function is equivalent to np.lexsort((cols, rows)) but implemented
    using stable sorting to ensure consistent results.

    Examples
    --------
    >>> import cupynumeric as np
    >>> rows = np.array([1, 0, 1, 0])
    >>> cols = np.array([2, 1, 1, 2])
    >>> indices = sort_by_rows_then_cols(rows, cols)
    >>> print(indices)  # [1, 3, 2, 0] - sorted by (row, col)
    """
    assert rows.size == cols.size

    # note that the lexsort reverses the order of key,
    # so this would be equivalent to np.lexsort((cols, rows))

    indices = cn.argsort(cols, kind="stable")
    order = cn.argsort(rows[indices], kind="stable")

    return indices[order]
