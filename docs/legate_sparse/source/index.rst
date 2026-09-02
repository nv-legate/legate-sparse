:html_theme.sidebar_secondary.remove:

NVIDIA Legate Sparse
====================

Legate Sparse is a `Legate`_ library that aims to provide a distributed
and accelerated drop-in replacement for the `scipy.sparse`_ library
on top of the `Legate`_ runtime. Legate Sparse interoperates with
`cuPyNumeric`_, a distributed and accelerated drop-in replacement
for `NumPy`_, to enable writing programs that operate on
distributed dense and sparse arrays.

Users can write Legate Sparse programs in Python and run
them on a CPU or a GPU and as soon as their problem size increases,
they can run them on a cluster of GPUs or CPUs without
any changes to their code.

.. note::

  The final Legate Sparse release supports a subset of APIs and options from
  scipy.sparse, including CSR and DIA formats for sparse matrices.

.. toctree::
  :maxdepth: 1
  :caption: Contents:

  installation
  user/index
  examples/index
  api/index
  faqs
  developer/index


Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
