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

import cupynumeric
import numpy
import scipy.sparse as scpy
import scipy.stats as stats


class Normal(stats.rv_continuous):
    def _rvs(self, *args, size=None, random_state=None):
        return random_state.standard_normal(size)


def sample(N: int, D: int, density: float, seed: int):
    NormalType = Normal(seed=seed)
    SeededNormal = NormalType()
    return scpy.random(
        N,
        D,
        density=density,
        format="csr",
        dtype=numpy.float64,
        random_state=seed,
        data_rvs=SeededNormal.rvs,
    )


def sample_dense(N: int, D: int, density: float, seed: int):
    return numpy.asarray(sample(N, D, density, seed).todense())


def sample_dense_vector(N: int, density: float, seed: int):
    return sample_dense(N, 1, density, seed).squeeze()


def simple_system_gen(N, M, cls, tol=0.5):
    a_dense = cupynumeric.random.rand(N, M)
    x = cupynumeric.random.rand(M)
    a_dense = cupynumeric.where(a_dense < tol, a_dense, 0)

    a_sparse = None if cls is None else cls(a_dense)

    return a_dense, a_sparse, x
