"""synoros_lib.manifolds — Riemannian manifold primitives, GPU-batched.

Currently implemented:
  fixed_rank.FixedRankManifold — M_r(m, n), the rank-r matrices in R^{m×n}
  spd.SPDManifold              — P(n), symmetric positive definite,
                                 affine-invariant metric

Planned (port/wrap from geoopt, geomstats, pymanopt):
  sphere, stiefel, grassmann, hyperbolic, product
"""

from synoros_lib.manifolds.fixed_rank import FixedRankManifold
from synoros_lib.manifolds.spd import SPDManifold

__all__ = ["FixedRankManifold", "SPDManifold"]
