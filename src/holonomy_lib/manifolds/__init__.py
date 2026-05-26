"""holonomy_lib.manifolds — Riemannian manifold primitives, GPU-batched.

Currently implemented:
  fixed_rank.FixedRankManifold — M_r(m, n), the rank-r matrices in R^{m×n}
  spd.SPDManifold              — P(n), symmetric positive definite,
                                 affine-invariant metric

Planned (port/wrap from geoopt, geomstats, pymanopt):
  sphere, stiefel, grassmann, hyperbolic, product
"""

from holonomy_lib.manifolds.fixed_rank import FixedRankManifold
from holonomy_lib.manifolds.spd import SPDManifold

__all__ = ["FixedRankManifold", "SPDManifold"]
