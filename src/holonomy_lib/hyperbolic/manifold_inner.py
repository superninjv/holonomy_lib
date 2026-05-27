"""Manifold-aware inner product — cosine-analog on a Riemannian manifold.

For points x, y on a Riemannian manifold M, the natural "inner
product" is not the ambient dot product (which ignores the geometry)
but the Riemannian inner product of their tangent representations at
a common base point. The standard construction uses the origin (or
any user-supplied base point) and pulls each point back via `log`:

    ⟨x, y⟩_M  :=  ⟨log_o(x), log_o(y)⟩_o,

where `log_o(·)` is the inverse exponential at the origin and the
right-hand inner product is the Riemannian metric on `T_o M`. The
quantity is symmetric, bilinear in its tangent representatives, and
reduces to the standard Euclidean inner when M = R^n.

For `LorentzManifold`, `log_0` returns the spatial part of the
tangent at the origin (`(B, n)` Euclidean coords), so the Riemannian
inner product collapses to the Euclidean dot product on those
coordinates — efficient and avoids the ambient (n+1)-dim arithmetic.

References:
  Pennec, X. (2006). Intrinsic statistics on Riemannian manifolds.
    J. Math. Imaging Vision 25(1):127–154 (§3 tangent statistics).
  Said, S., Bombrun, L., Berthoumieu, Y. (2015). Riemannian inner
    products in covariance estimation on the SPD manifold. SIIMS.
"""

from __future__ import annotations

import torch

from holonomy_lib.provenance import with_provenance


@with_provenance(
    "holonomy_lib.hyperbolic.manifold_aware_inner", op_version="0.1",
)
def manifold_aware_inner(
    x: torch.Tensor,
    y: torch.Tensor,
    manifold,
) -> torch.Tensor:
    """Riemannian inner product of x and y via the tangent at the origin.

    Computes `⟨log_o(x), log_o(y)⟩_o` where `o` is the manifold's origin.
    The pair `(x, y)` must both lie on `manifold`.

    For `LorentzManifold` the origin is the north pole; `log_0(x)`
    returns the spatial part of the tangent there, so the inner product
    is just the Euclidean dot of those `(B, n)` coordinates.

    Args:
      x, y: points on the manifold, shape `(B, ambient_dim)`.
        For `LorentzManifold(n)` `ambient_dim = n + 1`.
      manifold: a manifold object exposing a `log_0(point)` method that
        returns the tangent-at-origin representation of `point` as a
        `(B, n)` tensor (the spatial / intrinsic coordinates).

    Returns:
      `(B,)` tensor of Riemannian inner products.

    Example:
      >>> from holonomy_lib.manifolds import LorentzManifold
      >>> mfd = LorentzManifold(n=3)
      >>> x = mfd.random_point(batch_size=4)
      >>> sim = manifold_aware_inner(x, x, mfd)  # ‖log_0(x)‖² per batch
      >>> sim.shape
      torch.Size([4])

    References:
      Pennec (2006), §3 — tangent-space statistics on Riemannian manifolds.
    """
    log_x = manifold.log_0(x)        # (B, n) Euclidean tangent coords
    log_y = manifold.log_0(y)        # (B, n)
    return (log_x * log_y).sum(dim=-1)
