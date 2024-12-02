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

from typing import TYPE_CHECKING

import numpy as np
from legate.core import AutoTask  # type: ignore[attr-defined]
from legate.core import LogicalStore  # type: ignore[attr-defined]
from legate.core import ManualTask  # type: ignore[attr-defined]
from legate.core import Shape  # type: ignore[attr-defined]
from legate.core import TaskTarget  # type: ignore[attr-defined]
from legate.core import get_legate_runtime  # type: ignore[attr-defined]
from legate.core import get_machine  # type: ignore[attr-defined]
from legate.core import types  # type: ignore[attr-defined]

from .config import SparseOpCode, _library

if TYPE_CHECKING:
    from typing import Optional, Union

    import numpy.typing as npt

TO_CORE_DTYPES = {
    np.dtype(np.bool_): types.bool_,
    np.dtype(np.int8): types.int8,
    np.dtype(np.int16): types.int16,
    np.dtype(np.int32): types.int32,
    np.dtype(np.int64): types.int64,
    np.dtype(np.uint8): types.uint8,
    np.dtype(np.uint16): types.uint16,
    np.dtype(np.uint32): types.uint32,
    np.dtype(np.uint64): types.uint64,
    np.dtype(np.float16): types.float16,
    np.dtype(np.float32): types.float32,
    np.dtype(np.float64): types.float64,
    np.dtype(np.complex64): types.complex64,
    np.dtype(np.complex128): types.complex128,
}


# TODO (marsaev): rename to SparseRuntime to avoid confusion?
class Runtime:
    def __init__(self, sparse_library):
        self.sparse_library = sparse_library
        self.legate_runtime = get_legate_runtime()
        self.legate_machine = get_machine()

        self.dynamic_projection_functor_id = 1
        self.proj_fn_1d_to_2d_cache = {}

        # Load all the necessary CUDA libraries if we have GPUs.
        if self.num_gpus > 0:
            # TODO (rohany): Also handle destroying the cuda libraries when the
            #  runtime is torn down.
            task = self.legate_runtime.create_manual_task(
                self.sparse_library,
                SparseOpCode.LOAD_CUDALIBS,
                launch_shape=Shape((self.num_gpus,)),
            )
            task.execute()
            self.legate_runtime.issue_execution_fence(block=True)

    @property
    def num_procs(self):
        return self.legate_machine.count(self.legate_machine.preferred_target)

    @property
    def num_gpus(self):
        return self.legate_machine.count(TaskTarget.GPU)

    def create_store(
        self,
        ty: Union[npt.DTypeLike],
        shape: Optional[Union[tuple[int, ...], Shape]] = None,
        optimize_scalar: bool = False,
        ndim: Optional[int] = None,
    ) -> LogicalStore:
        core_ty = TO_CORE_DTYPES[ty] if isinstance(ty, np.dtype) else ty
        return self.legate_runtime.create_store(
            core_ty, shape=shape, optimize_scalar=optimize_scalar, ndim=ndim
        )

    # only OpCode
    def create_auto_task(self, OpCode) -> AutoTask:
        return self.legate_runtime.create_auto_task(self.sparse_library, OpCode)

    # OpCode and launch domains
    def create_manual_task(self, OpCode, *args) -> ManualTask:
        return self.legate_runtime.create_manual_task(
            self.sparse_library, OpCode, *args
        )


# TODO (marsaev): rename to sparse_runtime to avoid confusion?
runtime = Runtime(_library)
