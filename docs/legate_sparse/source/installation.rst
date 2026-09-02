Installation
============

Installaing Legate Sparse also installs the `Legate`_ and `cuPyNumeric`_ libraries.
The recommended way to install Legate Sparse is to use the conda packages.
For detailed information on building Legate and cuPyNumeric from source,
see build instructions for `Legate`_ and `cuPyNumeric`_.

Installing Conda Packages
-------------------------

The ``legate-sparse`` conda package already depends on ``legate`` and ``cupynumeric``,
and it will install these dependencies automatically.

To create a new environment and install:

.. code-block:: sh

    conda create -n myenv -c legate -c conda-forge legate-sparse


or to install in an existing environment:

.. code-block:: sh

    conda install -c legate -c conda-forge legate-sparse

Conda packages are available for Linux for CPU (Intel and ARM) and GPU (NVIDIA).


Verify your Installation
------------------------

A simple smoke test to verify your installation:

.. code-block:: python

    import legate_sparse as sparse

If this works, you are ready to use Legate Sparse! Try running our  `examples`_.


Building from source
---------------------

Use ``./install.py`` to build Legate Sparse from source.

.. code-block:: sh

    cd legate-sparse/
    ./install.py --verbose --clean

Licenses
--------

Legate Sparse is licensed under the `Apache License, Version 2.0`_.
