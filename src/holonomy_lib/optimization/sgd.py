"""Riemannian steepest descent (RSGD).

For a Riemannian manifold M with `projection` (ambient → tangent) and
`retraction` (move along tangent → new point), Riemannian SGD takes
the ambient gradient `g`, projects it to the tangent space at the
current point, scales by `-lr`, and retracts back onto M.

  ξ = projection(point, -lr · g)
  point' = retraction(point, ξ)

References:
  Absil, P.-A., Mahony, R., Sepulchre, R. (2008). Optimization
    Algorithms on Matrix Manifolds. Princeton University Press,
    §4.1-§4.2.
  Bonnabel, S. (2013). Stochastic gradient descent on Riemannian
    manifolds. IEEE TAC 58(9):2217-2229.
"""

from __future__ import annotations

from typing import Any

from holonomy_lib.optimization.base import RiemannianOptimizer
from holonomy_lib.provenance import with_provenance


def riemannian_sgd_step(
    manifold: Any,
    point: Any,
    ambient_grad: Any,
    lr: float,
) -> Any:
    """Functional one-step Riemannian SGD.

    Args:
      manifold: object exposing `projection(point, ambient)` and
        `retraction(point, tangent)` methods (e.g.
        `FixedRankManifold`, `SPDManifold`).
      point: current point on the manifold.
      ambient_grad: ambient-space gradient of the objective.
      lr: learning rate (positive scalar).

    Returns:
      new_point: the manifold-constrained next iterate.

    References:
      Absil-Mahony-Sepulchre (2008), §4.1.
    """
    # Project the negative ambient gradient onto the tangent space at
    # `point`, then retract. Equivalent to projecting `g` then negating
    # and scaling, but doing the sign before the projection is one
    # fewer tensor allocation. The projection is linear so the two
    # orderings give the same result up to floating-point.
    tangent = manifold.projection(point, _scale(ambient_grad, -lr))
    return manifold.retraction(point, tangent)


def _scale(x: Any, alpha: float) -> Any:
    """Multiply by a scalar, preserving the structure (tensor or tuple)."""
    if isinstance(x, tuple):
        return tuple(alpha * xi for xi in x)
    return alpha * x


class RiemannianSGD(RiemannianOptimizer):
    """Stateful wrapper around `riemannian_sgd_step`.

    Args:
      manifold: the underlying manifold object (see `riemannian_sgd_step`).
      lr: learning rate. Default `1e-2` (the torch.optim.SGD default).

    Example:
      >>> from holonomy_lib.manifolds import SPDManifold
      >>> mfd = SPDManifold(n=4)
      >>> opt = RiemannianSGD(mfd, lr=0.05)
      >>> point = mfd.random_point(batch_size=1)
      >>> # in a training loop:
      >>> ambient_grad = compute_ambient_grad(point)
      >>> point = opt.step(point, ambient_grad)
    """

    def __init__(self, manifold: Any, lr: float = 1e-2) -> None:
        super().__init__(manifold)
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        self.lr = lr

    def step(self, point: Any, ambient_grad: Any) -> Any:
        return riemannian_sgd_step(self.manifold, point, ambient_grad, self.lr)
