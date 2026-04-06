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

import cupynumeric as np
import numpy
import scipy.sparse as scpy
import scipy.stats as stats


class Normal(stats.rv_continuous):
    """Custom normal distribution class for reproducible random sampling.

    This class extends scipy.stats.rv_continuous to provide a custom
    normal distribution that can be used with scipy.sparse.random for
    generating sparse matrices with reproducible random values.

    Notes
    -----
    The _rvs method generates standard normal random variates using
    the provided random_state for reproducibility.
    """

    def _rvs(self, *args, size=None, random_state=None):
        """Generate standard normal random variates.

        Parameters
        ----------
        size : int or tuple, optional
            Number of random variates to generate.
        random_state : numpy.random.RandomState, optional
            Random state for reproducibility.

        Returns
        -------
        numpy.ndarray
            Array of standard normal random variates.
        """
        return random_state.standard_normal(size)


def sample(N: int, D: int, density: float, seed: int):
    """Generate a sparse matrix with random values from a normal distribution.

    Parameters
    ----------
    N : int
        Number of rows in the matrix.
    D : int
        Number of columns in the matrix.
    density : float
        Density of non-zero elements (between 0 and 1).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    scipy.sparse.csr_matrix
        A sparse matrix in CSR format with random normal values.

    Notes
    -----
    This function uses scipy.sparse.random with a custom normal distribution
    to generate sparse matrices with reproducible random values. The matrix
    is returned in CSR format.

    """
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
    """Generate a dense matrix with random values from a normal distribution.

    Parameters
    ----------
    N : int
        Number of rows in the matrix.
    D : int
        Number of columns in the matrix.
    density : float
        Density of non-zero elements (between 0 and 1).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        A dense matrix with random normal values.

    Notes
    -----
    This function generates a sparse matrix using sample() and then
    converts it to dense format. This is useful for creating test
    matrices that can be compared with sparse implementations.

    """
    return numpy.asarray(sample(N, D, density, seed).todense())


def sample_dense_vector(N: int, density: float, seed: int):
    """Generate a dense vector with random values from a normal distribution.

    Parameters
    ----------
    N : int
        Length of the vector.
    density : float
        Density of non-zero elements (between 0 and 1).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        A dense vector with random normal values.

    Notes
    -----
    This function generates a dense matrix with one column using
    sample_dense() and then squeezes it to a 1D vector.

    """
    return sample_dense(N, 1, density, seed).squeeze()


def simple_system_gen(N, M, cls, tol=0.5):
    """Generate a simple linear system for testing.

    Parameters
    ----------
    N : int
        Number of rows in the matrix.
    M : int
        Number of columns in the matrix.
    cls : type or None
        Class to use for creating the sparse matrix. If None, no sparse
        matrix is created.
    tol : float, optional
        Threshold for sparsity. Values below this threshold are set to zero.
        Default is 0.5.

    Returns
    -------
    tuple
        (a_dense, a_sparse, x) where:
        - a_dense: Dense matrix
        - a_sparse: Sparse matrix (or None if cls is None)
        - x: Dense vector

    Notes
    -----
    This function generates a random dense matrix and vector, then
    applies a threshold to create sparsity. The sparse matrix is
    created using the provided class if specified.

    """
    a_dense = np.random.rand(N, M)
    x = np.random.rand(M)
    a_dense = np.where(a_dense < tol, a_dense, 0)

    a_sparse = None if cls is None else cls(a_dense)

    return a_dense, a_sparse, x
