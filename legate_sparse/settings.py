# Copyright 2023-2024 NVIDIA Corporation
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
#
from __future__ import annotations

from legate.util.settings import (
    PrioritizedSetting,
    Settings,
    convert_bool,
    convert_str,
)

__all__ = ("settings",)


class SparseRuntimeSettings(Settings):
    fast_spgemm: PrioritizedSetting[bool] = PrioritizedSetting(
        "fast-spgemm",
        "LEGATE_SPARSE_FAST_SPGEMM",
        default=False,
        convert=convert_bool,
        help="""
        Switch to faster CUSPARSE_SPGEMM_ALG1, which, however, use
        significantly more FB memory. It will be used by default when cusparse<12.1,
        where memory-restricted SpGEMM was introduced.
        """,
    )

    cudss_commnccl_loc: PrioritizedSetting[bool] = PrioritizedSetting(
        "cudss-comm-lib",
        "CUDSS_COMM_LIB",
        default="",
        convert=convert_str,
        help="""
        For multi-gpu runs, set CUDSS_COMM_LIB env to /path/to/libcudss_commlayer_nccl.so
        """,
    )


settings = SparseRuntimeSettings()
