.. _advanced:

Advanced topics
===============

Multi-node execution
--------------------

Using Legate
~~~~~~~~~~~~

Legate Sparse programs can be run in parallel by using the ``--nodes`` option to
the ``legate`` driver, followed by the number of nodes to be used.
When running on 2+ nodes, a task launcher must be specified.

Legate Sparse currently supports using ``mpirun``, ``srun``, and ``jsrun`` as task
launchers for multi-node execution via the ``--launcher`` command like
arguments:

.. code-block:: sh

  legate --launcher srun --nodes 2 script.py <script options>

For more information on using Legate for multi-node executions, see
`Distributed Launch <https://docs.nvidia.com/legate/latest/usage.html#distributed-launch>`_.

Using a manual task manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: sh

  mpirun -np N legate script.py <script options>

It is also possible to use "standard python" in place of the ``legate`` driver, like this:

.. code-block:: sh

  python script.py <script options>

This invocation will use most of the hardware resources (CPU cores, GPUs, RAM, etc.)
available on the machine, but can be controlled. See `Resource Allocation`_ for more details.

The custom ``legate`` driver can be used to control the hardware resources used by the program
instead of the standard python interpreter, and is useful for multi-node execution as well.
It makes configuration of the hardware resources used by the program easier, and more execution
options. See `Usage`_ for more details.

Passing Legion and Realm arguments
----------------------------------

It is also possible to pass options to the Legion and Realm runtime directly,
by way of the ``LEGION_DEFAULT_ARGS`` and ``REALM_DEFAULT_ARGS`` environment
variables, for example:

.. code-block:: sh

    LEGION_DEFAULT_ARGS="-ll:cputsc" legate script.py

Resource Scoping
----------------

It is possible to scope the execution of a Legate Sparse program to a specific
processor type such as OMP, CPU, or GPU. This is useful in cases where, for instance,
the creation of sparse matrices is memory intensive and can be done on the CPU and
parallelized using OpenMP, while the iterative
sparse matrix operations can be done on the GPU.

The implementation of resource scoping in Legate Sparse is based on the
`Resource Scoping <https://docs.nvidia.com/legate/latest/usage.html#resource-scoping>`_
feature of Legate.

The following pseudocode shows how to use resource scoping to create a sparse matrix
on the CPU using OpenMP and solve it on the GPU.

.. code-block:: python

    from legate.core import TaskTarget, get_machine

    # get all available processors
    all_devices = get_machine()
    num_gpus = all_devices.count(TaskTarget.GPU)
    num_omps = all_devices.count(TaskTarget.OMP)
    num_cpus = all_devices.count(TaskTarget.CPU)

    # processors for the build and solve phases
    solve_procs = None
    build_procs = None

    # set processors for the build phase
    if num_omps > 0:
        build_procs = all_devices.only(TaskTarget.OMP)
    elif num_gpus > 0:
        build_procs = all_devices.only(TaskTarget.GPU)
    else:
        build_procs = all_devices.only(TaskTarget.CPU)

    # set processors for the solve phase
    if num_gpus > 0:
        solve_procs = all_devices.only(TaskTarget.GPU)
    elif num_omps > 0:
        solve_procs = all_devices.only(TaskTarget.OMP)
    else:
        solve_procs = all_devices.only(TaskTarget.CPU)

    # perform the compute on respective processors
    with build_procs:
        mat, rhs = build_sparse_matrix()

    with solve_procs:
        solution = solve_sparse_matrix(mat, rhs)


For more information on how to use this, see `common.py`_.

This can be executed with the following command, where we use 4 OMP threads and 2 GPUs,
but can be configured to use more OMP threads and GPUs.

.. code-block:: sh

    legate --omps 1 --ompthreads 4 --gpus 2 --fbmem 45000 script.py <script-options>
