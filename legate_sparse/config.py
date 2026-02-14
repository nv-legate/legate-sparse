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

import os
import platform
from ctypes import CDLL, RTLD_GLOBAL
from enum import IntEnum, unique
from typing import Any, cast

import cffi  # type: ignore
from legate.core import Library, get_legate_runtime, types


class _LegateSparseSharedLib:
    """Internal class representing the shared library interface.

    This class defines the interface to the C++ shared library that
    implements the core sparse matrix operations.
    """

    LEGATE_SPARSE_LOAD_CUDALIBS: int
    LEGATE_SPARSE_UNLOAD_CUDALIBS: int

    LEGATE_SPARSE_CSR_TO_DENSE: int
    LEGATE_SPARSE_DENSE_TO_CSR: int
    LEGATE_SPARSE_DENSE_TO_CSR_NNZ: int
    LEGATE_SPARSE_ZIP_TO_RECT_1: int
    LEGATE_SPARSE_UNZIP_RECT_1: int
    LEGATE_SPARSE_SCALE_RECT_1: int
    LEGATE_SPARSE_EXPAND_POS_TO_COORDINATES: int

    LEGATE_SPARSE_FAST_IMAGE_RANGE: int

    LEGATE_SPARSE_READ_MTX_TO_COO: int

    LEGATE_SPARSE_CSR_DIAGONAL: int

    LEGATE_SPARSE_CSR_INDEXING_CSR: int

    LEGATE_SPARSE_CSR_SPMV_ROW_SPLIT: int
    LEGATE_SPARSE_SPGEMM_CSR_CSR_CSR_NNZ: int
    LEGATE_SPARSE_SPGEMM_CSR_CSR_CSR: int
    LEGATE_SPARSE_SPGEMM_CSR_CSR_CSR_GPU: int
    LEGATE_SPARSE_AXPBY: int
    LEGATE_SPARSE_SPSOLVE: int
    LEGATE_SPARSE_GEAM_CSR_CSR_SYMBOLIC: int
    LEGATE_SPARSE_GEAM_CSR_CSR_COMPUTE: int


def dlopen_no_autoclose(ffi: Any, lib_path: str) -> Any:
    """Load a shared library without automatic closing.

    Parameters
    ----------
    ffi : Any
        The CFFI interface object.
    lib_path : str
        Path to the shared library to load.

    Returns
    -------
    Any
        The loaded library object.

    Notes
    -----
    This function loads a shared library using CDLL and converts it to
    a CFFI object without automatic closing. This prevents issues with
    symbol cleanup during shutdown.
    """
    # Use an already-opened library handle, which cffi will convert to a
    # regular FFI object (using the definitions previously added using
    # ffi.cdef), but will not automatically dlclose() on collection.
    lib = CDLL(lib_path, mode=RTLD_GLOBAL)
    return ffi.dlopen(ffi.cast("void *", lib._handle))


# Load the LegateSparse library first so we have a shard object that
# we can use to initialize all these configuration enumerations
class LegateSparseLib:
    """Legate sparse matrix library loader.

    This class handles loading and registering the Legate sparse matrix
    library with the Legate runtime.
    """

    def __init__(self, name: str) -> None:
        """Initialize the Legate sparse library.

        Parameters
        ----------
        name : str
            The name of the library to load.
        """
        self.name = name
        self.runtime = None

        shared_lib_path = self.get_shared_library()
        assert shared_lib_path is not None
        header = self.get_c_header()
        ffi = cffi.FFI()
        if header is not None:
            ffi.cdef(header)
        # Don't use ffi.dlopen(), because that will call dlclose()
        # automatically when the object gets collected, thus removing
        # symbols that may be needed when destroying C++ objects later
        # (e.g. vtable entries, which will be queried for virtual
        # destructors), causing errors at shutdown.
        shared_lib = dlopen_no_autoclose(ffi, shared_lib_path)
        self.shared_object = cast(_LegateSparseSharedLib, shared_lib)

    def register(self) -> None:
        """Register the library with the Legate runtime."""
        callback = getattr(
            self.shared_object, "legate_sparse_perform_registration"
        )
        callback()

    def get_shared_library(self) -> str:
        """Get the path to the shared library.

        Returns
        -------
        str
            The full path to the shared library file.
        """
        from legate_sparse.install_info import libpath

        return os.path.join(
            libpath, "liblegate_sparse" + self.get_library_extension()
        )

    def get_legate_library(self) -> Library:
        """Get the Legate library object.

        Returns
        -------
        Library
            The Legate library object.
        """
        return get_legate_runtime().find_library(self.name)

    def get_c_header(self) -> str:
        """Get the C header for the library.

        Returns
        -------
        str
            The C header content.
        """
        from legate_sparse.install_info import header

        return header

    @staticmethod
    def get_library_extension() -> str:
        """Get the appropriate library extension for the current platform.

        Returns
        -------
        str
            The library extension ('.so' for Linux, '.dylib' for macOS).

        Raises
        ------
        RuntimeError
            If the platform is not supported.
        """
        os_name = platform.system()
        if os_name == "Linux":
            return ".so"
        elif os_name == "Darwin":
            return ".dylib"
        raise RuntimeError(f"unknown platform {os_name!r}")


SPARSE_LIB_NAME = "legate.sparse"
"""Name of the Legate sparse library."""

sparse_lib = LegateSparseLib(SPARSE_LIB_NAME)

# Guard against double registration (can happen during Sphinx documentation builds)
try:
    sparse_lib.register()
except Exception:
    # Library may already be registered from a previous import
    pass

_sparse = sparse_lib.shared_object
# has to be called after register()
_library = sparse_lib.get_legate_library()


# Match these to entries in sparse_c.h
@unique
class SparseOpCode(IntEnum):
    """Enumeration of sparse matrix operation codes.

    These codes correspond to the operations implemented in the C++
    shared library and are used to dispatch tasks to the appropriate
    kernels.
    """

    LOAD_CUDALIBS = _sparse.LEGATE_SPARSE_LOAD_CUDALIBS
    UNLOAD_CUDALIBS = _sparse.LEGATE_SPARSE_UNLOAD_CUDALIBS

    CSR_TO_DENSE = _sparse.LEGATE_SPARSE_CSR_TO_DENSE

    DENSE_TO_CSR = _sparse.LEGATE_SPARSE_DENSE_TO_CSR
    DENSE_TO_CSR_NNZ = _sparse.LEGATE_SPARSE_DENSE_TO_CSR_NNZ

    READ_MTX_TO_COO = _sparse.LEGATE_SPARSE_READ_MTX_TO_COO

    AXPBY = _sparse.LEGATE_SPARSE_AXPBY

    ZIP_TO_RECT1 = _sparse.LEGATE_SPARSE_ZIP_TO_RECT_1
    UNZIP_RECT1 = _sparse.LEGATE_SPARSE_UNZIP_RECT_1
    SCALE_RECT_1 = _sparse.LEGATE_SPARSE_SCALE_RECT_1
    FAST_IMAGE_RANGE = _sparse.LEGATE_SPARSE_FAST_IMAGE_RANGE
    EXPAND_POS_TO_COORDINATES = _sparse.LEGATE_SPARSE_EXPAND_POS_TO_COORDINATES

    CSR_DIAGONAL = _sparse.LEGATE_SPARSE_CSR_DIAGONAL

    CSR_INDEXING_CSR = _sparse.LEGATE_SPARSE_CSR_INDEXING_CSR

    CSR_SPMV_ROW_SPLIT = _sparse.LEGATE_SPARSE_CSR_SPMV_ROW_SPLIT

    SPGEMM_CSR_CSR_CSR_NNZ = _sparse.LEGATE_SPARSE_SPGEMM_CSR_CSR_CSR_NNZ
    SPGEMM_CSR_CSR_CSR = _sparse.LEGATE_SPARSE_SPGEMM_CSR_CSR_CSR
    SPGEMM_CSR_CSR_CSR_GPU = _sparse.LEGATE_SPARSE_SPGEMM_CSR_CSR_CSR_GPU

    SPSOLVE = _sparse.LEGATE_SPARSE_SPSOLVE
    GEAM_CSR_CSR_SYMBOLIC = _sparse.LEGATE_SPARSE_GEAM_CSR_CSR_SYMBOLIC
    GEAM_CSR_CSR_COMPUTE = _sparse.LEGATE_SPARSE_GEAM_CSR_CSR_COMPUTE


# Register some types for us to use.
rect1 = types.rect_type(1)
"""1-dimensional rectangle type used for compressed storage formats."""
