"""synoros_lib.manifolds — Riemannian manifold primitives, GPU-batched.

Currently implemented:
  fixed_rank.FixedRankManifold — M_r(m, n), the rank-r matrices in R^{m×n}

Planned (port/wrap from geoopt, geomstats, pymanopt):
  sphere, stiefel, grassmann, spd, hyperbolic, product
"""

from synoros_lib.manifolds.fixed_rank import FixedRankManifold

__all__ = ["FixedRankManifold"]
