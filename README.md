<!--
Copyright 2023-2024 NVIDIA Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

-->


# Legate Sparse

Legate Sparse is a [Legate](https://github.com/nv-legate/legate.core) library
that aims to provide a distributed and accelerated drop-in replacement for the
[scipy.sparse](https://docs.scipy.org/doc/scipy/reference/sparse.html) library
on top of the [Legate](https://github.com/nv-legate/legate.core) runtime. 
Legate Sparse interoperates with 
[cuPyNumeric](https://github.com/nv-legate/cupynumeric/tree/main),
a distributed and accelerated drop-in replacement 
for [NumPy](https://numpy.org/doc/stable/reference/index.html#reference), to
enable writing programs that operate on distributed dense and sparse arrays.
Take a look at the `examples` directory for some applications that can 
use Legate Sparse. We have implemented
an explicit partial-differential equation (PDE) [solver](examples/pde.py) 
and [Geometric multi-grid](examples/gmg.py) solver. 
More complex and interesting applications are on the way -- stay tuned!

Legate Sparse is currently in alpha and supports a subset of APIs 
and options from scipy.sparse, so if you need an API, please open 
an issue and give us a summary of its usage. 

# Installation

To use Legate Sparse, `legate` and `cupynumeric` libraries have to be installed. 
They can be installed either by pulling the respective conda packages 
or by manually building from source. For more information, 
see build instructions for [Legate](https://github.com/nv-legate/legate.core) 
and [cuPyNumeric](https://github.com/nv-legate/cupynumeric/tree/main).

Follow the steps in this section.

## Use conda packages

To create a new environment and install: 
```
conda create -n myenv -c conda-forge -c legate cupynumeric legate-sparse
```

or to install in an existing environment:
```
conda install -c conda-forge -c legate cupynumeric legate-sparse
```

# Usage

To write programs using Legate Sparse, import the `legate_sparse` module, which
contains methods and types found in `scipy.sparse`. Note that the module is imported as `legate_sparse`
and not `legate.sparse`. Here is an example program saved as `main.py`. 

For more details on how to run legate programs, check 
our [documentation](https://docs.nvidia.com/cupynumeric/24.06/).
To run the application on a single GPU, use this command:

`legate --gpus 1 ./main.py`


```[python]
import legate_sparse as sparse
import cupynumeric as np

# number of diagonals in the matrix (including main diagonal)
n_diagonals = 3 

# number of rows in the matrix
nrows = 5 

# generate two tridiaonal matrices (n_diagonals=3) and multiply them
A = sparse.diags(
        [1] * n_diagonals,
        [x - (n_diagonals // 2) for x in range(n_diagonals)],
        shape = (nrows, nrows),
        format="csr",
        dtype=np.float64,
)

B = sparse.diags(
        [3] * n_diagonals,
        [x - (n_diagonals // 2) for x in range(n_diagonals)],
        shape = (nrows, nrows),
        format="csr",
        dtype=np.float64,
)

# spGEMM operation: multiplication of two sparse matrices
C = A @ B 
print(C.todense())
print()

# spMV operation: multiplication of a sparse matrix and a dense vector
x = np.ones(nrows)
C = A @ x 
print(C)

assert np.array_equal(A.todense().sum(axis=1), C)
"""
[[6. 6. 3. 0. 0.]
 [6. 9. 6. 3. 0.]
 [3. 6. 9. 6. 3.]
 [0. 3. 6. 9. 6.]
 [0. 0. 3. 6. 6.]]

[2. 3. 3. 3. 2.]
"""
```

For more examples, check the `examples` directory.
