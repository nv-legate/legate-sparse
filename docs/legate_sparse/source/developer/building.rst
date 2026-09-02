.. _building legate sparse from source:

Building from source
====================

Basic build
-----------

Users must have a working installation of the `Legate`_ and `CuPyNumeric`_
libraries prior to installing Legate Sparse.

.. note::
    Installing Legate Sparse by itself will not automatically install Legate or CuPyNumeric.

For more information on installing Legate and CuPyNumeric from source, see
`Building Legate From Source`_ and `Building CuPyNumeric From Source`_.

Once all dependencies are installed, you can simply invoke ``./install.py`` from
the Legate Sparse top-level directory. The build will automatically pick up the
configuration used when building Legate (e.g. the CUDA Toolkit directory).
