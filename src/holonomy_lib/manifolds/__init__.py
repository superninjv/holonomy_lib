"""holonomy_lib.manifolds — Riemannian manifold primitives, GPU-batched.

Currently implemented:
  fixed_rank.FixedRankManifold — M_r(m, n), the rank-r matrices in R^{m×n}
  spd.SPDManifold              — P(n), symmetric positive definite,
                                 affine-invariant metric
  lorentz.LorentzManifold      — H^n_k, hyperboloid model of hyperbolic
                                 space of curvature k < 0 (Nickel-Kiela 2018)

Planned (port/wrap from geoopt, geomstats, pymanopt):
  sphere, stiefel, grassmann, kappa-stereographic, product
"""

from holonomy_lib.manifolds.fixed_rank import FixedRankManifold
from holonomy_lib.manifolds.lorentz import LorentzManifold
from holonomy_lib.manifolds.spd import SPDManifold

__all__ = ["FixedRankManifold", "LorentzManifold", "SPDManifold"]
