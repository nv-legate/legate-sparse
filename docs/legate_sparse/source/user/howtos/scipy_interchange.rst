.. _scipy_interchange:

Interchanging Legate Sparse and SciPy Sparse
============================================

While developing applications, it might be beneficial to use SciPy Sparse
APIs in the beginning while still prototyping the application and then
replace them with Legate Sparse APIs later. This way the user can focus on
quickly prototyping the application and then later optimize the application
for better performance.

All examples under the ``examples`` directory can be run using SciPy
Sparse or Legate Sparse using the ``--package`` option. For instance,
to run the ``pde.py`` example using SciPy Sparse, you can run the
following command:

.. code-block:: sh

    legate pde.py --package scipy --throughput --max-iters 25 --warmup-iters 3

Similarly, to run the ``pde.py`` example using Legate Sparse, you can
run the following command:

.. code-block:: sh

    legate pde.py --package legate --throughput --max-iters 25 --warmup-iters 3

This will internally replace the import statements from

.. code-block:: python

    import scipy.sparse as sparse
    import numpy as np

to

.. code-block:: python

    import legate_sparse as sparse
    import cupynumeric as np


The example scripts under the ``examples`` directory all support a
special ``--package`` command line option to make it simple to switch
between Legate Sparse and SciPy Sparse. Additionally, you can change
the default for the ``--package`` option to enable using
the package of your choice by default.
