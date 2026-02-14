# Copyright 2022-2024 NVIDIA Corporation
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

# Portions of this file are also subject to the following license:
#
# Copyright (c) 2001-2002 Enthought, Inc. 2003-2022, SciPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided
# with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived
# from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Portions of this file are also subject to the following license:
# Copyright (c) 2015 Preferred Infrastructure, Inc.
# Copyright (c) 2015 Preferred Networks, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""
Sparse linear algebra (:mod:`legate_sparse.linalg`)
===================================================

.. currentmodule:: legate_sparse.linalg

Abstract linear operators
-------------------------

.. autosummary::
   :toctree: generated/

   LinearOperator -- abstract representation of a linear operator

Solving linear problems
-----------------------

Iterative methods for linear equation systems:

.. autosummary::
   :toctree: generated/

   cg -- Use Conjugate Gradient iteration to solve Ax = b
   gmres -- Use Generalized Minimal RESidual iteration to solve Ax = b

"""

from __future__ import annotations

import inspect
import warnings
from typing import TYPE_CHECKING, Protocol

import cupynumeric as cn
import numpy as np
from legate.core import align, image, track_provenance, types

from .config import SparseOpCode
from .runtime import runtime
from .utils import get_store_from_cupynumeric_array, store_to_cupynumeric_array

if TYPE_CHECKING:
    from typing import Any

    import numpy.typing as npt


class LOCallable(Protocol):
    def __call__(
        self, x: cn.ndarray, out: cn.ndarray | None = None
    ) -> cn.ndarray: ...


# We have to implement our own / copy the LinearOperator class from
# scipy as it invokes numpy directly causing all sorts of inline
# allocations and ping-ponging of instances between memories.
class LinearOperator:
    """Common interface for performing matrix vector products

    Many iterative methods (e.g. cg, gmres) do not need to know the
    individual entries of a matrix to solve a linear system A*x=b.
    Such solvers only require the computation of matrix vector
    products, A*v where v is a dense vector.  This class serves as
    an abstract interface between iterative solvers and matrix-like
    objects.

    To construct a concrete LinearOperator, either pass appropriate
    callables to the constructor of this class, or subclass it.

    A subclass must implement either one of the methods ``_matvec``
    and ``_matmat``, and the attributes/properties ``shape`` (pair of
    integers) and ``dtype`` (may be None). It may call the ``__init__``
    on this class to have these attributes validated. Implementing
    ``_matvec`` automatically implements ``_matmat`` (using a naive
    algorithm) and vice-versa.

    Optionally, a subclass may implement ``_rmatvec`` or ``_adjoint``
    to implement the Hermitian adjoint (conjugate transpose). As with
    ``_matvec`` and ``_matmat``, implementing either ``_rmatvec`` or
    ``_adjoint`` implements the other automatically. Implementing
    ``_adjoint`` is preferable; ``_rmatvec`` is mostly there for
    backwards compatibility.

    Parameters
    ----------
    shape : tuple
        Matrix dimensions (M, N).
    matvec : callable f(v)
        Returns returns A * v.
    rmatvec : callable f(v)
        Returns A^H * v, where A^H is the conjugate transpose of A.
    matmat : callable f(V)
        Returns A * V, where V is a dense matrix with dimensions (N, K).
    dtype : dtype
        Data type of the matrix.
    rmatmat : callable f(V)
        Returns A^H * V, where V is a dense matrix with dimensions (M, K).

    Attributes
    ----------
    args : tuple
        For linear operators describing products etc. of other linear
        operators, the operands of the binary operation.
    ndim : int
        Number of dimensions (this is always 2)

    See Also
    --------
    aslinearoperator : Construct LinearOperators

    Notes
    -----
    The user-defined matvec() function must properly handle the case
    where v has shape (N,) as well as the (N,1) case.  The shape of
    the return type is handled internally by LinearOperator.

    LinearOperator instances can also be multiplied, added with each
    other and exponentiated, all lazily: the result of these operations
    is always a new, composite LinearOperator, that defers linear
    operations to the original operators and combines the results.

    More details regarding how to subclass a LinearOperator and several
    examples of concrete LinearOperator instances can be found in the
    external project `PyLops <https://pylops.readthedocs.io>`_.


    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse.linalg import LinearOperator
    >>> def mv(v):
    ...     return np.array([2*v[0], 3*v[1]])
    ...
    >>> A = LinearOperator((2,2), matvec=mv)
    >>> A
    <2x2 _CustomLinearOperator with dtype=float64>
    >>> A.matvec(np.ones(2))
    array([ 2.,  3.])
    >>> A * np.ones(2)
    array([ 2.,  3.])

    """

    ndim = 2

    def __new__(cls, *args: Any, **kwargs: Any) -> LinearOperator:
        if cls is LinearOperator:
            # Operate as _CustomLinearOperator factory.
            return super(LinearOperator, cls).__new__(_CustomLinearOperator)
        else:
            obj = super(LinearOperator, cls).__new__(cls)

            if (
                type(obj)._matvec == LinearOperator._matvec
                and type(obj)._matmat == LinearOperator._matmat
            ):
                warnings.warn(
                    "LinearOperator subclass should implement"
                    " at least one of _matvec and _matmat.",
                    category=RuntimeWarning,
                    stacklevel=2,
                )

            return obj

    def __init__(
        self, dtype: npt.dtype[Any] | None, shape: tuple[int, ...]
    ) -> None:
        """Initialize this LinearOperator.

        To be called by subclasses. ``dtype`` may be None; ``shape`` should
        be convertible to a length-2 tuple.
        """
        if dtype is not None:
            dtype = np.dtype(dtype)

        shape = tuple(shape)
        self.dtype = dtype
        self.shape = shape

    def _init_dtype(self) -> None:
        """Called from subclasses at the end of the __init__ routine."""
        if self.dtype is None:
            v = cn.zeros(self.shape[-1])
            self.dtype = cn.asarray(self.matvec(v)).dtype

    def _matmat(
        self, x: cn.ndarray, out: cn.ndarray | None = None
    ) -> cn.ndarray:
        """Default matrix-matrix multiplication handler."""
        raise NotImplementedError

    def _matvec(
        self, x: cn.ndarray, out: cn.ndarray | None = None
    ) -> cn.ndarray:
        """Default matrix-vector multiplication handler.

        If self is a linear operator of shape (M, N), then this method will
        be called on a shape (N,) or (N, 1) ndarray, and should return a
        shape (M,) or (M, 1) ndarray.

        This default implementation falls back on _matmat, so defining that
        will define matrix-vector multiplication as well.
        """
        raise NotImplementedError

    def matvec(
        self, x: cn.ndarray, out: cn.ndarray | None = None
    ) -> cn.ndarray:
        """Matrix-vector multiplication.

        Performs the operation y=A*x where A is an MxN linear
        operator and x is a column vector or 1-d array.

        Parameters
        ----------
        x : {matrix, ndarray}
            An array with shape (N,) or (N,1).

        Returns
        -------
        y : {matrix, ndarray}
            A matrix or ndarray with shape (M,) or (M,1) depending
            on the type and shape of the x argument.

        Notes
        -----
        This matvec wraps the user-specified matvec routine or overridden
        _matvec method to ensure that y has the correct shape and type.

        """
        M, N = self.shape

        if x.shape != (N,) and x.shape != (N, 1):
            raise ValueError("dimension mismatch")

        y = cn.asarray(self._matvec(x, out=out))

        if x.ndim == 1:
            # TODO (hme): This is a cuPyNumeric bug, reshape should accept an
            # integer.
            y = y.reshape((M,))
        elif x.ndim == 2:
            y = y.reshape(M, 1)
        else:
            raise ValueError("invalid shape returned by user-defined matvec()")

        return y

    def _rmatvec(
        self, x: cn.ndarray, out: cn.ndarray | None = None
    ) -> cn.ndarray:
        """Default implementation of _rmatvec; defers to adjoint."""
        raise NotImplementedError

    def rmatvec(
        self, x: cn.ndarray, out: cn.ndarray | None = None
    ) -> cn.ndarray:
        """Adjoint matrix-vector multiplication.

        Performs the operation y = A^H * x where A is an MxN linear
        operator and x is a column vector or 1-d array.

        Parameters
        ----------
        x : {matrix, ndarray}
            An array with shape (M,) or (M,1).

        Returns
        -------
        y : {matrix, ndarray}
            A matrix or ndarray with shape (N,) or (N,1) depending
            on the type and shape of the x argument.

        Notes
        -----
        This rmatvec wraps the user-specified rmatvec routine or overridden
        _rmatvec method to ensure that y has the correct shape and type.

        """
        M, N = self.shape

        if x.shape != (M,) and x.shape != (M, 1):
            raise ValueError("dimension mismatch")

        y = cn.asarray(self._rmatvec(x, out=out))

        if x.ndim == 1:
            y = y.reshape(N)
        elif x.ndim == 2:
            y = y.reshape(N, 1)
        else:
            raise ValueError(
                "invalid shape returned by user-defined rmatvec()"
            )

        return y


# _CustomLinearOperator is a LinearOperator defined by user-specified
# operations. It is lifted from scipy.sparse.
class _CustomLinearOperator(LinearOperator):
    """Linear operator defined in terms of user-specified operations."""

    _matvec_impl: LOCallable
    _rmatvec_impl: LOCallable | None

    def __init__(
        self,
        shape: tuple[int, ...],
        matvec: LOCallable,
        rmatvec: LOCallable | None = None,
        matmat: LOCallable | None = None,
        dtype: npt.dtype[Any] | None = None,
        rmatmat: LOCallable | None = None,
    ) -> None:
        super().__init__(dtype, shape)

        self.args = ()

        self._matvec_impl = matvec
        self._rmatvec_impl = rmatvec

        # Check if the implementations of matvec and rmatvec have the out=
        # parameter.
        self._matvec_has_out = self._has_out(self._matvec_impl)
        self._rmatvec_has_out = self._has_out(self._rmatvec_impl)

        self._init_dtype()

    def _matvec(
        self, x: cn.ndarray, out: cn.ndarray | None = None
    ) -> cn.ndarray:
        if self._matvec_has_out:
            return self._matvec_impl(x, out=out)
        else:
            if out is None:
                return self._matvec_impl(x)
            else:
                out[:] = self._matvec_impl(x)
                return out

    def _rmatvec(
        self, x: cn.ndarray, out: cn.ndarray | None = None
    ) -> cn.ndarray:
        assert self._rmatvec_impl is not None
        func = self._rmatvec_impl
        if func is None:
            raise NotImplementedError("rmatvec is not defined")
        if self._rmatvec_has_out:
            return self._rmatvec_impl(x, out=out)
        else:
            if out is None:
                return self._rmatvec_impl(x)
            else:
                result = self._rmatvec_impl(x)
                out[:] = result
                return out

    def _has_out(self, o: LOCallable | None) -> bool:
        if o is None:
            return False
        sig = inspect.signature(o)
        return "out" in sig.parameters


# _SparseMatrixLinearOperator is an overload of LinearOperator to wrap
# sparse matrices as a linear operator. It caches the conjugate transpose
# of the sparse matrices to avoid repeat conversions.
class _SparseMatrixLinearOperator(LinearOperator):
    AH: cn.ndarray | None

    def __init__(self, A: cn.ndarray) -> None:
        self.A = A
        self.AH = None
        super().__init__(A.dtype, A.shape)

    def _matvec(
        self, x: cn.ndarray, out: cn.ndarray | None = None
    ) -> cn.ndarray:
        return self.A.dot(x, out=out)

    def _rmatvec(
        self, x: cn.ndarray, out: cn.ndarray | None = None
    ) -> cn.ndarray:
        if self.AH is None:
            self.AH = self.A.T.conj()
        assert self.AH is not None
        return self.AH.dot(x, out=out)


# IdentityOperator is a no-op linear operator, and is lifted from
# scipy.sparse.
class IdentityOperator(LinearOperator):
    def __init__(
        self, shape: tuple[int, ...], dtype: npt.dtype[Any] | None = None
    ) -> None:
        super().__init__(dtype, shape)

    def _matvec(
        self, x: cn.ndarray, out: cn.ndarray | None = None
    ) -> cn.ndarray:
        # If out is specified, copy the input into the output.
        if out is not None:
            out[:] = x
            return out
        else:
            # To make things easier for external users of this class, copy
            # the input to avoid silently aliasing the input array.
            return x.copy()

    def _rmatvec(
        self, x: cn.ndarray, out: cn.ndarray | None = None
    ) -> cn.ndarray:
        # If out is specified, copy the input into the output.
        if out is not None:
            out[:] = x
            return out
        else:
            # To make things easier for external users of this class, copy
            # the input to avoid silently aliasing the input array.
            return x.copy()


def make_linear_operator(A: Any | LinearOperator) -> LinearOperator:
    """Convert a matrix to a LinearOperator.

    Parameters
    ----------
    A : array_like, sparse matrix, or LinearOperator
        The matrix to convert.

    Returns
    -------
    LinearOperator
        A LinearOperator representation of A.

    Notes
    -----
    If A is already a LinearOperator, it is returned unchanged.
    Otherwise, A is wrapped in a _SparseMatrixLinearOperator.
    """
    if isinstance(A, LinearOperator):
        return A
    else:
        return _SparseMatrixLinearOperator(A)


# cg_axpby is a specialized implementation of the operation
# y = alpha * x + beta * y for CG solvers in a Legion context.
# Instead of explicitly providing alpha and beta, we accept
# a and b futures, which will be fused into a computation of
# a / b within the task, control over whether a/b should be
# interpreted as alpha or beta, and finally whether a/b
# should be negated. This allows for avoiding unnecessary
# future operations to compute new futures, and avoids
# allocating unnecessary futures.
@track_provenance(nested=True)
def cg_axpby(
    y: cn.ndarray,
    x: cn.ndarray,
    a: cn.ndarray,
    b: cn.ndarray,
    isalpha: bool = True,
    negate: bool = False,
) -> cn.ndarray:
    """Perform fused vector operation for CG solvers.

    This function performs the operation y = alpha * x + beta * y where
    alpha and beta are computed as a/b within the task. This avoids
    unnecessary future operations and memory allocations.

    Parameters
    ----------
    y : cupynumeric.ndarray
        Output vector that will be modified in-place.
    x : cupynumeric.ndarray
        Input vector for the operation.
    a : cupynumeric.ndarray
        Numerator for computing alpha or beta.
    b : cupynumeric.ndarray
        Denominator for computing alpha or beta.
    isalpha : bool, optional
        If True, a/b is interpreted as alpha. If False, as beta.
        Default is True.
    negate : bool, optional
        If True, negate the computed coefficient. Default is False.

    Returns
    -------
    cupynumeric.ndarray
        The modified y vector (same as input y).

    Notes
    -----
    This is a specialized implementation for CG solvers that fuses
    coefficient computation with vector operations to avoid unnecessary
    memory allocations and future operations in the Legion runtime.
    """
    y_store = get_store_from_cupynumeric_array(y)
    x_store = get_store_from_cupynumeric_array(x)
    task = runtime.create_auto_task(SparseOpCode.AXPBY)
    task.add_output(y_store)
    task.add_input(x_store)
    a_store = get_store_from_cupynumeric_array(a)
    b_store = get_store_from_cupynumeric_array(b)
    task.add_input(a_store)
    task.add_input(b_store)
    task.add_broadcast(a_store)
    task.add_broadcast(b_store)
    task.add_scalar_arg(isalpha, types.bool_)
    task.add_scalar_arg(negate, types.bool_)
    task.add_input(y_store)
    task.add_alignment(y_store, x_store)
    task.execute()
    return y


def _get_atol_rtol(
    b_norm: float | cn.ndarray,
    tol: float | None = None,
    atol: float = 0.0,
    rtol: float = 1e-5,
) -> tuple[float, float]:
    """Compute absolute and relative tolerances for convergence.

    Parameters
    ----------
    b_norm : float
        Norm of the right-hand side vector.
    tol : float, optional
        Legacy tolerance parameter. If provided, overrides rtol.
    atol : float, optional
        Absolute tolerance. Default is 0.0.
    rtol : float, optional
        Relative tolerance. Default is 1e-5.

    Returns
    -------
    tuple
        (atol, rtol) - computed absolute and relative tolerances.

    Notes
    -----
    If atol is None, it is set to rtol. The final atol is the maximum
    of the provided atol and rtol * b_norm.
    """
    rtol = float(tol) if tol is not None else rtol

    if atol is None:
        atol = rtol

    atol = max(float(atol), float(rtol) * float(b_norm))

    return atol, rtol


def cg(
    A: Any | LinearOperator,
    b: cn.ndarray,
    x0: cn.ndarray | None = None,
    tol: float | None = None,
    maxiter: int | None = None,
    M: Any | LinearOperator | None = None,
    callback: Any | None = None,
    atol: float = 0.0,
    rtol: float = 1e-5,
    conv_test_iters: int = 25,
) -> tuple[cn.ndarray, int]:
    """Solve a linear system using the Conjugate Gradient method.

    Parameters
    ----------
    A : sparse matrix or LinearOperator
        The coefficient matrix of the linear system.
    b : cupynumeric.ndarray
        Right-hand side of the linear system.
    x0 : cupynumeric.ndarray, optional
        Initial guess for the solution. If None, uses zero vector.
    tol : float, optional
        Legacy tolerance parameter. If provided, overrides rtol.
    maxiter : int, optional
        Maximum number of iterations. If None, uses 10 * n.
    M : sparse matrix or LinearOperator, optional
        Preconditioner for A. If None, uses identity.
    callback : callable, optional
        User-specified function called after each iteration.
    atol : float, optional
        Absolute tolerance for convergence. Default is 0.0.
    rtol : float, optional
        Relative tolerance for convergence. Default is 1e-5.
    conv_test_iters : int, optional
        Number of iterations between convergence tests. Default is 25.

    Returns
    -------
    tuple
        (x, info) where x is the solution and info is zero if solution is
        converged else number of iterations

    Raises
    ------
    AssertionError
        If b is not 1D or A is not square.

    Notes
    -----
    This implementation follows SciPy's CG solver semantics closely.
    The method uses fused vector operations to avoid unnecessary
    memory allocations and improve performance.

    Convergence is tested every conv_test_iters iterations to avoid
    the overhead of computing the residual norm in every iteration.

    Examples
    --------
    >>> import cupynumeric as np
    >>> from legate_sparse import csr_array, linalg
    >>> A = csr_array([[4, 1], [1, 3]])
    >>> b = np.array([1, 2])
    >>> x, iters = linalg.cg(A, b)
    >>> print(f"Solution: {x}, Iterations: {iters}")
    """
    # We keep semantics as close as possible to scipy.cg.
    # https://github.com/scipy/scipy/blob/v1.9.0/scipy/sparse/linalg/_isolve/iterative.py#L298-L385
    assert len(b.shape) == 1 or (len(b.shape) == 2 and b.shape[1] == 1)
    assert len(A.shape) == 2 and A.shape[0] == A.shape[1]

    b_norm = cn.linalg.norm(b)
    atol, _ = _get_atol_rtol(b_norm, tol, atol, rtol)

    n = b.shape[0]
    if maxiter is None:
        maxiter = n * 10

    A = make_linear_operator(A)
    M = (
        IdentityOperator(A.shape, dtype=A.dtype)
        if M is None
        else make_linear_operator(M)
    )
    x = cn.zeros(n) if x0 is None else x0.copy()
    p = cn.zeros(n)

    # This implementation is adapted from CuPy's CG solve:
    # https://github.com/cupy/cupy/blob/master/cupyx/scipy/sparse/linalg/_iterative.py.
    # # Hold onto several temps to store allocations used in each iteration.
    r = b - A.matvec(x)
    iters = 0
    rho: int | cn.ndarray = 0
    z = None
    q = None

    converged = False
    while iters < maxiter:
        z = M.matvec(r, out=z)
        rho1 = rho
        rho = r.dot(z)
        if iters == 0:
            # Make sure not to take an alias to z here, since we
            # modify p in place.
            p[:] = z
        else:
            # Utilize a fused vector addition with scalar multiplication
            # kernel. Computes p = p * beta + z, where beta = rho / rho1.
            cg_axpby(p, z, rho, rho1, isalpha=False, negate=False)
        q = A.matvec(p, out=q)
        pq = p.dot(q)
        # Utilize fused vector adds here as well.
        # Computes x += alpha * p, where alpha = rho / pq.
        cg_axpby(x, p, rho, pq, isalpha=True, negate=False)
        # Computes r -= alpha * Ap.
        cg_axpby(r, q, rho, pq, isalpha=True, negate=True)
        iters += 1
        if callback is not None:
            callback(x)
        if (
            iters % conv_test_iters == 0 or iters == (maxiter - 1)
        ) and cn.linalg.norm(r) < atol:
            converged = True
            # Test convergence every conv_test_iters iterations.
            break

    info = 0
    if iters == maxiter and not converged:
        info = iters

    return x, info


# This implementation of GMRES is lifted from the cupy implementation:
# https://github.com/cupy/cupy/blob/9d2e2381ae7f33a42291d1bf8271484c9d2a55ac/cupyx/scipy/sparse/linalg/_iterative.py#L94.
def gmres(
    A: Any | LinearOperator,
    b: cn.ndarray,
    x0: cn.ndarray | None = None,
    tol: float | None = None,
    restart: int | None = None,
    maxiter: int | None = None,
    M: Any | LinearOperator | None = None,
    callback: Any = None,
    restrt: int | None = None,
    atol: float = 0.0,
    callback_type: str | None = None,
    rtol: float = 1e-5,
) -> tuple[cn.ndarray, int]:
    """Solve a linear system using the Generalized Minimal Residual method.

    Parameters
    ----------
    A : sparse matrix or LinearOperator
        The coefficient matrix of the linear system.
    b : cupynumeric.ndarray
        Right-hand side of the linear system with shape (n,) or (n, 1).
    x0 : cupynumeric.ndarray, optional
        Starting guess for the solution. If None, uses zero vector.
    tol : float, optional
        Legacy tolerance parameter. If provided, overrides rtol.
    restart : int, optional
        Number of iterations between restarts. Larger values increase
        iteration cost but may be necessary for convergence. Default is 20.
    maxiter : int, optional
        Maximum number of iterations. If None, uses 10 * n.
    M : sparse matrix or LinearOperator, optional
        Preconditioner for A. The preconditioner should approximate
        the inverse of A. If None, uses identity.
    callback : callable, optional
        User-specified function called on every restart.
    restrt : int, optional
        Deprecated alias for restart parameter.
    atol : float, optional
        Absolute tolerance for convergence. Default is 0.0.
    callback_type : str, optional
        Type of callback argument: 'x' for current solution vector,
        'pr_norm' for relative preconditioned residual norm. Default is 'pr_norm'.
    rtol : float, optional
        Relative tolerance for convergence. Default is 1e-5.

    Returns
    -------
    tuple
        (x, info) where x is the converged solution and info provides
        convergence information.

    Raises
    ------
    AssertionError
        If b is not 1D or A is not square.
    ValueError
        If callback_type is not 'x' or 'pr_norm'.

    Notes
    -----
    This implementation is adapted from CuPy's GMRES solver.
    The method uses Arnoldi iteration to build a Krylov subspace
    and solves the least squares problem in that subspace.

    For convergence, the residual must satisfy:
    norm(b - A @ x) <= max(rtol * norm(b), atol)

    The restart parameter controls the trade-off between memory usage
    and convergence rate. Larger restart values may improve convergence
    but require more memory.

    References
    ----------
    M. Wang, H. Klie, M. Parashar and H. Sudan, "Solving Sparse Linear
    Systems on NVIDIA Tesla GPUs", ICCS 2009 (2009).

    Examples
    --------
    >>> import cupynumeric as np
    >>> from legate_sparse import csr_array, linalg
    >>> A = csr_array([[4, 1, 0], [1, 3, 1], [0, 1, 2]])
    >>> b = np.array([1, 2, 3])
    >>> x, info = linalg.gmres(A, b, restart=10)
    >>> print(f"Solution: {x}, Info: {info}")
    """
    assert len(b.shape) == 1 or (len(b.shape) == 2 and b.shape[1] == 1)
    assert len(A.shape) == 2 and A.shape[0] == A.shape[1]
    # cannot specify both restart and restrt
    assert restrt is None or not restart

    if restrt is not None:
        restart = restrt

    A = make_linear_operator(A)
    n = A.shape[0]
    M = (
        IdentityOperator(A.shape, dtype=A.dtype)
        if M is None
        else make_linear_operator(M)
    )
    x = cn.zeros(n) if x0 is None else x0.copy()

    b_norm = cn.linalg.norm(b)
    atol, _ = _get_atol_rtol(b_norm, tol, atol, rtol)

    if maxiter is None:
        maxiter = n * 10
    if restart is None:
        restart = 20
    restart = min(restart, n)
    if callback_type is None:
        callback_type = "pr_norm"
    if callback_type not in ("x", "pr_norm"):
        raise ValueError("Unknown callback_type: {}".format(callback_type))
    if callback is None:
        callback_type = None

    V = cn.empty((n, restart), dtype=A.dtype)
    H: Any = cn.zeros((restart + 1, restart), dtype=A.dtype)
    e: Any = cn.zeros((restart + 1,), dtype=A.dtype)

    def compute_hu(u: cn.ndarray, j: int) -> tuple[cn.ndarray, cn.ndarray]:
        """Compute Householder transformation for Arnoldi iteration.

        Parameters
        ----------
        u : cupynumeric.ndarray
            Vector to be transformed.
        j : int
            Current iteration index.

        Returns
        -------
        tuple
            (h, u) where h contains the Householder coefficients and
            u is the transformed vector.

        Notes
        -----
        This function computes the Householder transformation that
        orthogonalizes the current vector against the previous basis
        vectors in the Arnoldi iteration.
        """
        h = V[:, : j + 1].conj().T @ u
        u -= V[:, : j + 1] @ h
        return h, u

    iters = 0
    while True:
        mx = M.matvec(x)
        r = b - A.matvec(mx)
        r_norm = cn.linalg.norm(r)
        if callback_type == "x":
            callback(mx)
        elif callback_type == "pr_norm" and iters > 0:
            callback(r_norm / b_norm)
        if r_norm <= atol or iters >= maxiter:
            break
        v = r / r_norm
        V[:, 0] = v
        e[0] = r_norm

        # Arnoldi iteration.
        for j in range(restart):
            z = M.matvec(v)
            u = A.matvec(z)
            H[: j + 1, j], u = compute_hu(u, j)
            H[j + 1, j] = cn.linalg.norm(u)
            if j + 1 < restart:
                v = u / H[j + 1, j]
                V[:, j + 1] = v

        # Note: The least-square solution to equation Hy = e is computed on CPU
        # because it is faster if tha matrix size is small.
        ret = cn.linalg.lstsq(H, e)  # type: ignore [attr-defined]
        y = ret[0]
        x += V @ y
        iters += restart

    info = 0
    if iters == maxiter and not (r_norm <= atol):
        info = iters
    return mx, info


def spsolve(A: Any, b: np.ndarray) -> np.ndarray:
    """
    Solve a linear system of equation Ax=b by factorizing A

    Parameters
    ----------
    A : csr_array
        Input sparse matrix of shape (N, N).
    b : cupynumeric.ndarray
        Dense vector of shape (N,).

    Returns
    -------
    x : cupynumeric.ndarray
        Dense vector of shape (N,), that solves A x = b.

    Raises
    ------
    RuntimeError
        If attempted to solve on any configuration other than one GPU
    ValueError
        If the RHS is not one-dimensional

    Notes
    -----
    This function uses cuDSS to perform the sparse direct solve, which
    computes the reordering on Host.

    """

    # TODO:
    # Support multi-dimensional RHS. Note that cuDSS only supports
    # column-major order for x and b, so we need to update the
    # mapper for those stores. Partitioning constraints will also need to
    # be changed since alignment constraints will need both stores
    # to be of the same dimension (e.g., we cannot align pos (1D)
    # and b (say, 2D) without manipulating the stores

    # NOTE: multi-gpu runs might hang with cuda < 13.0.0.
    # For multi-gpu runs, the user is expected to set the path to
    # libcudss_comm_nccl.so in the env CUDSS_COMM_LIB
    if runtime.num_gpus == 0:
        raise RuntimeError("spsolve is currently supported only for GPU(s)")

    if b.ndim != 1:
        raise ValueError(f"RHS must be 1D. Dimension of b is: {b.ndim}")

    b_store = get_store_from_cupynumeric_array(b)
    x_store = runtime.create_store(b.dtype, shape=(A.shape[1],))

    task = runtime.create_auto_task(SparseOpCode.SPSOLVE)

    pos_part = task.add_input(A.pos)
    crd_part = task.add_input(A.crd)
    vals_part = task.add_input(A.vals)
    b_part = task.add_input(b_store)
    x_part = task.add_output(x_store)
    task.add_scalar_arg(A.shape[0], types.uint64)  # global nrows
    task.add_scalar_arg(A.vals.size, types.uint64)  # global nnz

    # Add communicator
    task.add_communicator("nccl")

    # Since we don't support multi-gpu or multi-cpu runs, these constraints
    # are not particularly relevant right now, but they enable
    # debugging the multi-gpu hang. The matrix and the vectors are
    # partitioned row-wise without any sparsity-dependent constraints
    # that is typical in other API implementations in legate-sparse
    # that use mathlibs (e.g., cuSparse). This passes on the responsibility
    # of inserting appropriate communication primitives to the
    # underlying math library, cuDSS. This is why we don't constraint the
    # partition of x to the image of crd (e.g., like in SpMv in csr.py)
    task.add_constraint(image(pos_part, crd_part))
    task.add_constraint(image(pos_part, vals_part))
    task.add_constraint(align(x_part, pos_part))
    task.add_constraint(align(b_part, pos_part))

    task.execute()

    return store_to_cupynumeric_array(x_store)


# this function has been adapted from cupy's implementation of `eigsh`:
# https://github.com/cupy/cupy/blob/v13.6.0/cupyx/scipy/sparse/linalg/_eigen.py
def eigsh(
    a,
    k=6,
    *,
    which="LM",
    v0=None,
    ncv=None,
    maxiter=None,
    tol=0,
    return_eigenvectors=True,
):
    def _lanczos(a, V, u, alpha, beta, i_start, i_end):
        for i in range(i_start, i_end):
            u[...] = a.matvec(V[i])
            alpha[i] = cn.dot(V[i].conj(), u)

            # Full reorthogonalization with "twice is enough" strategy
            # for improved numerical stability. This matches the approach
            # used in robust Lanczos implementations.
            # First pass
            coeffs = V[: i + 1].conj() @ u
            u -= coeffs @ V[: i + 1]
            # Second pass for numerical stability
            coeffs2 = V[: i + 1].conj() @ u
            u -= coeffs2 @ V[: i + 1]

            beta[i] = cn.linalg.norm(u)
            if i >= i_end - 1:
                break
            V[i + 1] = u / beta[i]

    def _eigsh_solve_ritz(alpha, beta, beta_k, k, which):
        # Note: This is done on the CPU using numpy, following CuPy's approach.
        # This avoids numerical issues that can occur with GPU-based eigh
        # on small tridiagonal matrices from the thick-restart Lanczos.
        alpha_np = np.array(alpha)
        beta_np = np.array(beta)
        t = np.diag(alpha_np)
        t = t + np.diag(beta_np[:-1], k=1)
        t = t + np.diag(beta_np[:-1], k=-1)
        if beta_k is not None:
            beta_k_np = np.array(beta_k)
            t[k, :k] = beta_k_np
            t[:k, k] = beta_k_np
        w, s = np.linalg.eigh(t)

        # Pick-up k ritz-values and ritz-vectors
        if which == "LA":
            idx = np.argsort(w)
            wk = w[idx[-k:]]
            sk = s[:, idx[-k:]]
        elif which == "LM":
            idx = np.argsort(np.absolute(w))
            wk = w[idx[-k:]]
            sk = s[:, idx[-k:]]
        elif which == "SA":
            idx = np.argsort(w)
            wk = w[idx[:k]]
            sk = s[:, idx[:k]]
        # Convert back to cupynumeric arrays
        return cn.array(wk), cn.array(sk)

    # Convert to LinearOperator for uniform matvec interface
    a = make_linear_operator(a)
    n = a.shape[0]
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("expected square matrix (shape: {})".format(a.shape))
    if a.dtype.char not in "fdFD":
        raise TypeError("unsupprted dtype (actual: {})".format(a.dtype))
    if k <= 0:
        raise ValueError("k must be greater than 0 (actual: {})".format(k))
    if k >= n:
        raise ValueError("k must be smaller than n (actual: {})".format(k))
    if which not in ("LM", "LA", "SA"):
        raise ValueError(
            "which must be 'LM','LA'or'SA' (actual: {})".format(which)
        )
    if ncv is None:
        ncv = min(max(2 * k, k + 32), n - 1)
    else:
        ncv = min(max(ncv, k + 2), n - 1)
    if maxiter is None:
        maxiter = 10 * n
    if tol == 0:
        tol = cn.finfo(a.dtype).eps

    if k + 1 == ncv:
        raise ValueError(
            f"k must be smaller than ncv - 1 (k + 1 < ncv < n)."
            f" ncv: {ncv}, k: {k}, n: {n}"
        )

    alpha = cn.zeros((ncv,), dtype=a.dtype)
    beta = cn.zeros((ncv,), dtype=a.dtype.char.lower())
    V = cn.empty((ncv, n), dtype=a.dtype)

    if v0 is None:
        u = cn.random.random((n,)).astype(a.dtype)
        V[0] = u / cn.linalg.norm(u)
    else:
        u = v0
        V[0] = v0 / cn.linalg.norm(v0)

    _lanczos(a, V, u, alpha, beta, 0, ncv)

    iter_current = ncv
    w, s = _eigsh_solve_ritz(alpha, beta, None, k, which)
    x = V.T @ s

    beta_k = beta[-1] * s[-1, :]
    res = cn.linalg.norm(beta_k)

    iter_increment = ncv - k
    # Track initial beta scale for detecting relative breakdown
    # When beta[k] is too small relative to the typical beta values,
    # the thick restart becomes numerically unstable
    initial_beta_scale = cn.max(cn.abs(beta[:-1]))

    while res > tol and iter_current < maxiter:
        beta[:k] = 0
        alpha[:k] = w
        V[:k] = x.T

        # Full reorthogonalization with "twice is enough" (same as in _lanczos)
        coeffs = V[:k].conj() @ u
        u = u - coeffs @ V[:k]
        coeffs2 = V[:k].conj() @ u
        u = u - coeffs2 @ V[:k]

        u_norm = cn.linalg.norm(u)
        # Check for numerical breakdown: if u_norm is too small relative
        # to initial scale, the thick restart becomes numerically unstable.
        # A ratio < 0.1 indicates potential numerical issues.
        if u_norm < 0.1 * initial_beta_scale:
            # Accept current eigenvalues as converged
            break

        V[k] = u / u_norm
        u[...] = a.matvec(V[k])
        alpha[k] = cn.dot(V[k].conj(), u)
        u -= alpha[k] * V[k]
        u -= V[:k].T @ beta_k
        beta[k] = cn.linalg.norm(u)

        # Check for numerical breakdown after computing beta[k]
        # If beta[k] is very small relative to initial scale,
        # continuing will cause numerical instability
        if beta[k] < 0.1 * initial_beta_scale:
            # Accept current eigenvalues as converged
            break

        # note that this can run into Out of bounds error
        # in legate if `k` is not properly constrained
        # in the initial part of the algorithm
        V[k + 1] = u / beta[k]

        _lanczos(a, V, u, alpha, beta, k + 1, ncv)
        w, s = _eigsh_solve_ritz(alpha, beta, beta_k, k, which)
        x = V.T @ s
        beta_k = beta[-1] * s[-1, :]
        res = cn.linalg.norm(beta_k)

        iter_current += iter_increment

    if return_eigenvectors:
        idx = cn.argsort(w)
        return w[idx], x[:, idx]
    else:
        return cn.sort(w)
