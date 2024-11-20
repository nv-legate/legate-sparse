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
"""

Not sure what is supposed to go in here...

"""

import scipy.sparse as _sp  # type: ignore

from .coverage import clone_module  # noqa: F401
from .csr import csr_array, csr_matrix  # noqa: F401
from .module import *  # noqa: F401

clone_module(_sp, globals())

del clone_module
del _sp
