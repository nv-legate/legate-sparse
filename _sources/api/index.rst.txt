API Reference
=============

.. currentmodule:: legate_sparse

.. toctree::
   :hidden:

   linalg
   settings

Submodules
==========

.. autosummary::

    linalg - Sparse linear algebra routines


Sparse array classes
====================

.. autosummary::
   :toctree: generated/

    csr_array - Compressed Sparse Row array
    dia_array - Sparse array with DIAgonal storage

Sparse array functions
======================

.. autosummary::
   :toctree: generated/

   geam - Generalized sparse matrix addition (CSR)


Identifying sparse arrays
=========================

.. autosummary::
   :toctree: generated/

   issparse - Check if the argument is a sparse object (array or matrix).


Sparse matrix classes
=====================

.. autosummary::
   :toctree: generated/

   csr_matrix - Compressed Sparse Row matrix
   dia_matrix - Sparse matrix with DIAgonal storage


Identifying sparse matrices
===========================

.. autosummary::
   :toctree: generated/

   issparse
   isspmatrix
   isspmatrix_csr

Input/Output
============

.. autosummary::
   :toctree: generated/

   mmread
