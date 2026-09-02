.. _benchmarking:

Performance Benchmarking
========================

Using Legate timing tools
-------------------------

Use legate's timing API to measure elapsed time, rather than standard Python
timers. Legate Sparse executes work asynchronously when possible, and a standard
Python timer will only measure the time taken to launch the work, not the time
spent in actual computation.

Make sure warm-up iterations, initialization, I/O, and other one-time
computations are excluded while timing iterative computations.

Here is an example of how to measure elapsed time in milliseconds:

.. code-block:: python

    import legate_sparse as sparse
    from legate.timing import time

    init() # Initialization step

    # Do few warm-up iterations
    for i in range(n_warmup_iters):
        compute()

    start = time()
    for i in range(niters):
        compute()
    end = time()

    elapsed_millisecs = (end - start)/1000.0

    dump_data() # I/O


Guidelines for performance benchmarks
-------------------------------------

Manual partitioning of data for use with message-passing from Python (say,
using mpi4py package) is not supported. If your code is manually
partitioned using MPI4Py or any other library, please rewrite the code
for sequential execution without any partitioning or communication primitives.

Ensure that the problem size is large enough to offset runtime overheads
associated with tasks. A rule of thumb is that the problem size should be large
enough for a task granularity of about one millisecond on the GPU. There can
be parts of the algorithm that don't lend themselves to concurrent execution
due to dependencies between tasks. Having a large problem size is
crucial to offset the runtime overheads associated with tasks in such cases.
