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

import inspect
import warnings

import cupynumeric as np
from legate.core import track_provenance  # type: ignore[attr-defined]
from legate.core import types  # type: ignore[attr-defined]

from .config import SparseOpCode
from .runtime import runtime
from .utils import get_store_from_cupynumeric_array


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

    def __new__(cls, *args, **kwargs):
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

    def __init__(self, dtype, shape):
        """Initialize this LinearOperator.

        To be called by subclasses. ``dtype`` may be None; ``shape`` should
        be convertible to a length-2 tuple.
        """
        if dtype is not None:
            dtype = np.dtype(dtype)

        shape = tuple(shape)
        self.dtype = dtype
        self.shape = shape

    def _init_dtype(self):
        """Called from subclasses at the end of the __init__ routine."""
        if self.dtype is None:
            v = np.zeros(self.shape[-1])
            self.dtype = np.asarray(self.matvec(v)).dtype

    def _matvec(self, x, out=None):
        """Default matrix-vector multiplication handler.

        If self is a linear operator of shape (M, N), then this method will
        be called on a shape (N,) or (N, 1) ndarray, and should return a
        shape (M,) or (M, 1) ndarray.

        This default implementation falls back on _matmat, so defining that
        will define matrix-vector multiplication as well.
        """
        raise NotImplementedError

    def matvec(self, x, out=None):
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

        y = np.asarray(self._matvec(x, out=out))

        if x.ndim == 1:
            # TODO (hme): This is a cuPyNumeric bug, reshape should accept an
            # integer.
            y = y.reshape((M,))
        elif x.ndim == 2:
            y = y.reshape(M, 1)
        else:
            raise ValueError("invalid shape returned by user-defined matvec()")

        return y

    def _rmatvec(self, x, out=None):
        """Default implementation of _rmatvec; defers to adjoint."""
        raise NotImplementedError

    def rmatvec(self, x, out=None):
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

        y = np.asarray(self._rmatvec(x, out=out))

        if x.ndim == 1:
            y = y.reshape(N)
        elif x.ndim == 2:
            y = y.reshape(N, 1)
        else:
            raise ValueError("invalid shape returned by user-defined rmatvec()")

        return y


# _CustomLinearOperator is a LinearOperator defined by user-specified
# operations. It is lifted from scipy.sparse.
class _CustomLinearOperator(LinearOperator):
    """Linear operator defined in terms of user-specified operations."""

    def __init__(
        self,
        shape,
        matvec,
        rmatvec=None,
        matmat=None,
        dtype=None,
        rmatmat=None,
    ):
        super().__init__(dtype, shape)

        self.args = ()

        self.__matvec_impl = matvec
        self.__rmatvec_impl = rmatvec

        # Check if the implementations of matvec and rmatvec have the out=
        # parameter.
        self._matvec_has_out = self._has_out(self.__matvec_impl)
        self._rmatvec_has_out = self._has_out(self.__rmatvec_impl)

        self._init_dtype()

    def _matvec(self, x, out=None):
        if self._matvec_has_out:
            return self.__matvec_impl(x, out=out)
        else:
            if out is None:
                return self.__matvec_impl(x)
            else:
                out[:] = self.__matvec_impl(x)
                return out

    def _rmatvec(self, x, out=None):
        func = self.__rmatvec_impl
        if func is None:
            raise NotImplementedError("rmatvec is not defined")
        if self._rmatvec_has_out:
            return self.__rmatvec_impl(x, out=out)
        else:
            if out is None:
                return self.__rmatvec_impl(x)
            else:
                result = self.__rmatvec_impl(x)
                out[:] = result
                return out

    def _has_out(self, o):
        if o is None:
            return False
        sig = inspect.signature(o)
        for key, param in sig.parameters.items():
            if key == "out":
                return True
        return False


# _SparseMatrixLinearOperator is an overload of LinearOperator to wrap
# sparse matrices as a linear operator. It caches the conjugate transpose
# of the sparse matrices to avoid repeat conversions.
class _SparseMatrixLinearOperator(LinearOperator):
    def __init__(self, A):
        self.A = A
        self.AH = None
        super().__init__(A.dtype, A.shape)

    def _matvec(self, x, out=None):
        return self.A.dot(x, out=out)

    def _rmatvec(self, x, out=None):
        if self.AH is None:
            self.AH = self.A.T.conj(copy=False)
        return self.AH.dot(x, out=out)


# IdentityOperator is a no-op linear operator, and is lifted from
# scipy.sparse.
class IdentityOperator(LinearOperator):
    def __init__(self, shape, dtype=None):
        super().__init__(dtype, shape)

    def _matvec(self, x, out=None):
        # If out is specified, copy the input into the output.
        if out is not None:
            out[:] = x
            return out
        else:
            # To make things easier for external users of this class, copy
            # the input to avoid silently aliasing the input array.
            return x.copy()

    def _rmatvec(self, x, out=None):
        # If out is specified, copy the input into the output.
        if out is not None:
            out[:] = x
            return out
        else:
            # To make things easier for external users of this class, copy
            # the input to avoid silently aliasing the input array.
            return x.copy()


def make_linear_operator(A):
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
def cg_axpby(y, x, a, b, isalpha=True, negate=False):
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


def _get_atol_rtol(b_norm, tol=None, atol=0.0, rtol=1e-5):
    rtol = float(tol) if tol is not None else rtol

    if atol is None:
        atol = rtol

    atol = max(float(atol), float(rtol) * float(b_norm))

    return atol, rtol


def cg(
    A,
    b,
    x0=None,
    tol=None,
    maxiter=None,
    M=None,
    callback=None,
    atol=0.0,
    rtol=1e-5,
    conv_test_iters=25,
):
    # We keep semantics as close as possible to scipy.cg.
    # https://github.com/scipy/scipy/blob/v1.9.0/scipy/sparse/linalg/_isolve/iterative.py#L298-L385
    assert len(b.shape) == 1 or (len(b.shape) == 2 and b.shape[1] == 1)
    assert len(A.shape) == 2 and A.shape[0] == A.shape[1]

    bnrm2 = np.linalg.norm(b)
    atol, _ = _get_atol_rtol(bnrm2, tol, atol, rtol)

    n = b.shape[0]
    if maxiter is None:
        maxiter = n * 10

    A = make_linear_operator(A)
    M = (
        IdentityOperator(A.shape, dtype=A.dtype)
        if M is None
        else make_linear_operator(M)
    )
    x = np.zeros(n) if x0 is None else x0.copy()
    p = np.zeros(n)

    # This implementation is adapted from CuPy's CG solve:
    # https://github.com/cupy/cupy/blob/master/cupyx/scipy/sparse/linalg/_iterative.py.
    # # Hold onto several temps to store allocations used in each iteration.
    r = b - A.matvec(x)
    iters = 0
    rho = 0
    z = None
    q = None

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
        if (iters % conv_test_iters == 0 or iters == (maxiter - 1)) and np.linalg.norm(
            r
        ) < atol:
            # Test convergence every conv_test_iters iterations.
            break

    return x, iters


# This implementation of GMRES is lifted from the cupy implementation:
# https://github.com/cupy/cupy/blob/9d2e2381ae7f33a42291d1bf8271484c9d2a55ac/cupyx/scipy/sparse/linalg/_iterative.py#L94.
def gmres(
    A,
    b,
    x0=None,
    tol=None,
    restart=None,
    maxiter=None,
    M=None,
    callback=None,
    restrt=None,
    atol=0.0,
    callback_type=None,
    rtol=1e-5,
):
    """Uses Generalized Minimal RESidual iteration to solve ``Ax = b``.
    Args:
        A (ndarray, spmatrix or LinearOperator): The real or complex
            matrix of the linear system with shape ``(n, n)``. ``A`` must be
            :class:`cupy.ndarray`, :class:`cupyx.scipy.sparse.spmatrix` or
            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.
        b (cupy.ndarray): Right hand side of the linear system with shape
            ``(n,)`` or ``(n, 1)``.
        x0 (cupy.ndarray): Starting guess for the solution.
        tol (float): Tolerance for convergence. This argument is optional,
            deprecated in favour of ``rtol``.
        restart (int): Number of iterations between restarts. Larger values
            increase iteration cost, but may be necessary for convergence.
        maxiter (int): Maximum number of iterations.
        M (ndarray, spmatrix or LinearOperator): Preconditioner for ``A``.
            The preconditioner should approximate the inverse of ``A``.
            ``M`` must be :class:`cupy.ndarray`,
            :class:`cupyx.scipy.sparse.spmatrix` or
            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.
        callback (function): User-specified function to call on every restart.
            It is called as ``callback(arg)``, where ``arg`` is selected by
            ``callback_type``.
        callback_type (str): 'x' or 'pr_norm'. If 'x', the current solution
            vector is used as an argument of callback function. if 'pr_norm',
            relative (preconditioned) residual norm is used as an arugment.
        atol, rtol (float): Tolerance for convergence. For convergence,
            ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` should be satisfied.
            The default is ``atol=0.`` and ``rtol=1e-5``.
    Returns:
        tuple:
            It returns ``x`` (cupy.ndarray) and ``info`` (int) where ``x`` is
            the converged solution and ``info`` provides convergence
            information.
    Reference:
        M. Wang, H. Klie, M. Parashar and H. Sudan, "Solving Sparse Linear
        Systems on NVIDIA Tesla GPUs", ICCS 2009 (2009).
    .. seealso:: :func:`scipy.sparse.linalg.gmres`
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
    x = np.zeros(n) if x0 is None else x0.copy()

    bnrm2 = np.linalg.norm(b)
    atol, _ = _get_atol_rtol(bnrm2, tol, atol, rtol)

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

    V = np.empty((n, restart), dtype=A.dtype)
    H = np.zeros((restart + 1, restart), dtype=A.dtype)
    e = np.zeros((restart + 1,), dtype=A.dtype)

    def compute_hu(u, j):
        h = V[:, : j + 1].conj().T @ u
        u -= V[:, : j + 1] @ h
        return h, u

    iters = 0
    while True:
        mx = M.matvec(x)
        r = b - A.matvec(mx)
        r_norm = np.linalg.norm(r)
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
            H[j + 1, j] = np.linalg.norm(u)
            if j + 1 < restart:
                v = u / H[j + 1, j]
                V[:, j + 1] = v

        # Note: The least-square solution to equation Hy = e is computed on CPU
        # because it is faster if tha matrix size is small.
        ret = np.linalg.lstsq(H, e)
        y = ret[0]
        x += V @ y
        iters += restart

    info = 0
    if iters == maxiter and not (r_norm <= atol):
        info = iters
    return mx, info
