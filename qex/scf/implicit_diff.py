"""Implicit differentiation of the SCF fixed point.

The SCF cycle defines a fixed-point map on the density matrix,

    dm* = T(dm*, theta),

where ``theta`` are the differentiable inputs (XC network params, h1e, eri, ...)
and one application of ``T`` is a single Kohn-Sham step: build the Fock matrix
from ``dm``, solve the generalized eigenproblem, occupy, and rebuild ``dm``.

The two ways to get gradients of ``dm*`` w.r.t. ``theta``:

- **Unrolling** (what ``rks_loss_scan`` does by default): differentiate straight
  through every cycle of the loop. Correct, but the backward pass retraces the
  whole eigensolve + XC network ``max_cycle`` times, so memory and the reverse
  graph grow linearly in ``max_cycle`` — the dominant cost for an expensive XC
  (QCNN).

- **Implicit differentiation** (this module): differentiate the fixed-point
  *condition* instead of the iteration. By the implicit function theorem, with
  ``F(dm, theta) = dm - T(dm, theta) = 0`` at the solution,

      d dm*/d theta = -(dF/ddm)^{-1} (dF/dtheta).

  The forward solve runs under ``stop_gradient`` (no graph kept), and the VJP is
  recovered from a single linear solve against ``(I - dT/ddm)^T`` plus one VJP
  through ``T`` w.r.t. ``theta``. Cost is independent of how many cycles the
  forward solve took, so it is cheaper to trace/compile and scales to many
  cycles and large networks.

``custom_fixed_point`` here is a small, self-contained ``jax.custom_vjp`` wrapper
specialised to a density-matrix fixed point. It deliberately does not depend on
``pyscfad`` machinery (which is tied to pyscfad's pytree SCF objects); the qex
SCF loop is plain JAX, so a plain-JAX wrapper is the right fit and is trivial to
test in isolation.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from chex import Array

# A density-matrix fixed point that has not converged to ~machine precision will
# give a biased implicit gradient (the IFT assumes F(dm*, theta) = 0). This is a
# soft default tolerance used by the forward solve's early-exit; it does not gate
# correctness, only how many cycles we spend.
_DEFAULT_FP_TOL = 1e-9


def _default_linear_solve(matvec: Callable[[Array], Array], b: Array) -> Array:
    """Solve ``A u = b`` for the IFT cotangent via GMRES.

    ``matvec`` applies ``A = (I - dT/ddm)^T`` to a density-matrix-shaped vector;
    ``b`` is the incoming cotangent ``dm_bar``. GMRES only needs the matvec, so we
    never materialise the (n^2 x n^2) Jacobian. Restart/iteration caps are small
    because the SCF Jacobian is well-conditioned near convergence.
    """
    u, _ = jax.scipy.sparse.linalg.gmres(
        matvec, b, tol=1e-8, atol=1e-8, restart=20, maxiter=40,
    )
    return u


def custom_fixed_point(
    step_fn: Callable[[Array, object], Array],
    *,
    max_cycle: int = 50,
    tol: float = _DEFAULT_FP_TOL,
    linear_solve: Callable | None = None,
) -> Callable[[Array, object], Array]:
    """Wrap an SCF step into an implicitly-differentiable fixed-point solver.

    Parameters
    ----------
    step_fn : callable ``(dm, theta) -> dm``
        One SCF iteration on the density matrix. Must be a pure function of
        ``(dm, theta)``; ``theta`` is any pytree of differentiable inputs (e.g.
        ``(params, eri, h1e, s1e, ao_grid, grid_weights, ...)``). DIIS must NOT
        live inside ``step_fn`` — DIIS only accelerates the forward solve and
        does not change the fixed point, so it would only add non-differentiable
        history state here.
    max_cycle : int
        Forward fixed-point iterations (upper bound; the solve early-exits once
        ``|dm_{k+1} - dm_k|`` drops below ``tol``).
    tol : float
        Forward convergence tolerance for the early exit.
    linear_solve : callable ``(matvec, b) -> u``, optional
        Linear solver for the backward IFT system. Defaults to GMRES.

    Returns
    -------
    callable ``(dm0, theta) -> dm*``
        Differentiable in ``theta`` (and ``dm0``, though ``dm0``'s gradient is
        zero at a converged fixed point — the solution does not depend on the
        starting guess). The forward pass carries no autodiff graph through the
        cycles; the backward pass is a single linear solve, independent of how
        many cycles ran.
    """
    if linear_solve is None:
        linear_solve = _default_linear_solve

    @jax.custom_vjp
    def solver(dm0: Array, theta: object) -> Array:
        return _solve_forward(step_fn, dm0, theta, max_cycle, tol)

    def solver_fwd(dm0: Array, theta: object):
        dm_star = _solve_forward(step_fn, dm0, theta, max_cycle, tol)
        # Residuals are arrays only (the converged dm* and theta). dm0 is not
        # needed: the gradient w.r.t. the initial guess is zero at the fixed point.
        return dm_star, (dm_star, theta)

    # Implicit function theorem. With F(dm, theta) = dm - T(dm, theta) = 0 at the
    # solution, d dm*/d theta = -(dF/ddm)^{-1} (dF/dtheta); for a cotangent dm_bar
    # the input cotangents come from one linear solve against (I - dT/ddm)^T plus
    # one VJP through T w.r.t. theta. This differentiates T (hence the eigensolve)
    # a second time, so the eigensolver must be twice-differentiable
    # (qex uses `eigh.eigh_gen`).
    #
    # `_backward` is its own `@jax.jit`. The trace boundary is load-bearing: when
    # the whole loss runs under the caller's `jax.jit` + `jax.grad`, computing
    # these VJPs inline would fold the second-order eigensolve trace into the
    # outer linearization and leak a `LinearizeTracer`. Isolating it makes the
    # outer pass treat the backward rule as a single primitive.
    @jax.jit
    def _backward(dm_star, theta, dm_bar):
        _, vjp_dm = jax.vjp(lambda dm: step_fn(dm, theta), dm_star)
        _, vjp_theta = jax.vjp(lambda th: step_fn(dm_star, th), theta)

        # Solve (I - dT/ddm)^T u = dm_bar.  matvec(u) = u - (dT/ddm)^T u.
        def matvec(u):
            (jt_u,) = vjp_dm(u)
            return u - jt_u

        u = linear_solve(matvec, dm_bar)
        (theta_bar,) = vjp_theta(u)
        return theta_bar

    def solver_bwd(res, dm_bar: Array):
        dm_star, theta = res
        theta_bar = _backward(dm_star, theta, dm_bar)
        # dm0 is a nondiff-in-practice input: report zero cotangent.
        dm0_bar = jnp.zeros_like(dm_star)
        return dm0_bar, theta_bar

    solver.defvjp(solver_fwd, solver_bwd)
    return solver


def _solve_forward(
    step_fn: Callable[[Array, object], Array],
    dm0: Array,
    theta: object,
    max_cycle: int,
    tol: float,
) -> Array:
    """Iterate ``step_fn`` to a density-matrix fixed point under ``stop_gradient``.

    Runs inside ``lax.scan`` with a converged-flag carry so the trace is a single
    body (compile cost independent of ``max_cycle``) and extra cycles after
    convergence are cheap no-ops. The whole solve is wrapped in
    ``stop_gradient`` so no reverse graph is built — gradients come from the
    custom VJP, not from unrolling this loop.
    """
    theta = jax.lax.stop_gradient(theta)
    dm0 = jax.lax.stop_gradient(dm0)

    def body(carry, _):
        dm, converged = carry
        dm_next = step_fn(dm, theta)
        # Freeze the iterate once converged so later cycles don't drift.
        dm_next = jnp.where(converged, dm, dm_next)
        delta = jnp.linalg.norm(dm_next - dm)
        converged = converged | (delta < tol)
        return (dm_next, converged), None

    (dm_star, _), _ = jax.lax.scan(
        body, (dm0, jnp.array(False)), None, length=max_cycle,
    )
    return jax.lax.stop_gradient(dm_star)
