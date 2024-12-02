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

import os
import platform
from ctypes import CDLL, RTLD_GLOBAL
from enum import IntEnum, unique
from typing import Any, cast

import cffi  # type: ignore
from legate.core import Library, get_legate_runtime, types  # type: ignore[attr-defined]


class _LegateSparseSharedLib:
    LEGATE_SPARSE_DENSE_TO_CSR: int
    LEGATE_SPARSE_DENSE_TO_CSR_NNZ: int
    LEGATE_SPARSE_ZIP_TO_RECT_1: int
    LEGATE_SPARSE_UNZIP_RECT_1: int
    LEGATE_SPARSE_SCALE_RECT_1: int
    LEGATE_SPARSE_EXPAND_POS_TO_COORDINATES: int

    LEGATE_SPARSE_FAST_IMAGE_RANGE: int

    LEGATE_SPARSE_READ_MTX_TO_COO: int

    LEGATE_SPARSE_CSR_DIAGONAL: int

    LEGATE_SPARSE_CSR_SPMV_ROW_SPLIT: int
    LEGATE_SPARSE_SPGEMM_CSR_CSR_CSR_NNZ: int
    LEGATE_SPARSE_SPGEMM_CSR_CSR_CSR: int
    LEGATE_SPARSE_SPGEMM_CSR_CSR_CSR_GPU: int
    LEGATE_SPARSE_AXPBY: int

    LEGATE_SPARSE_PROJ_FN_1D_TO_2D: int
    LEGATE_SPARSE_LAST_PROJ_FN: int


def dlopen_no_autoclose(ffi: Any, lib_path: str) -> Any:
    # Use an already-opened library handle, which cffi will convert to a
    # regular FFI object (using the definitions previously added using
    # ffi.cdef), but will not automatically dlclose() on collection.
    lib = CDLL(lib_path, mode=RTLD_GLOBAL)
    return ffi.dlopen(ffi.cast("void *", lib._handle))


# Load the LegateSparse library first so we have a shard object that
# we can use to initialize all these configuration enumerations
class LegateSparseLib(Library):
    def __init__(self, name):
        self.name = name
        self.runtime = None
        self.shared_object = None

        self.name = name

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
        callback = getattr(self.shared_object, "legate_sparse_perform_registration")
        callback()

    def get_shared_library(self) -> str:
        from legate_sparse.install_info import libpath

        return os.path.join(libpath, "liblegate_sparse" + self.get_library_extension())

    def get_legate_library(self) -> Library:
        return get_legate_runtime().find_library(self.name)

    def get_c_header(self) -> str:
        from legate_sparse.install_info import header

        return header

    @staticmethod
    def get_library_extension() -> str:
        os_name = platform.system()
        if os_name == "Linux":
            return ".so"
        elif os_name == "Darwin":
            return ".dylib"
        raise RuntimeError(f"unknown platform {os_name!r}")


SPARSE_LIB_NAME = "legate.sparse"
sparse_lib = LegateSparseLib(SPARSE_LIB_NAME)
sparse_lib.register()
_sparse = sparse_lib.shared_object
# has to be called after register()
_library = sparse_lib.get_legate_library()


# Match these to entries in sparse_c.h
@unique
class SparseOpCode(IntEnum):
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

    CSR_SPMV_ROW_SPLIT = _sparse.LEGATE_SPARSE_CSR_SPMV_ROW_SPLIT

    SPGEMM_CSR_CSR_CSR_NNZ = _sparse.LEGATE_SPARSE_SPGEMM_CSR_CSR_CSR_NNZ
    SPGEMM_CSR_CSR_CSR = _sparse.LEGATE_SPARSE_SPGEMM_CSR_CSR_CSR
    SPGEMM_CSR_CSR_CSR_GPU = _sparse.LEGATE_SPARSE_SPGEMM_CSR_CSR_CSR_GPU


@unique
class SparseProjectionFunctor(IntEnum):
    PROMOTE_1D_TO_2D = _sparse.LEGATE_SPARSE_PROJ_FN_1D_TO_2D
    LAST_STATIC_PROJ_FN = _sparse.LEGATE_SPARSE_LAST_PROJ_FN


# Register some types for us to use.
rect1 = types.rect_type(1)
