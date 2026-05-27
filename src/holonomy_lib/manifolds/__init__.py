"""holonomy_lib.manifolds — Riemannian manifold primitives, GPU-batched.

Currently implemented:
  fixed_rank.FixedRankManifold — M_r(m, n), the rank-r matrices in R^{m×n}
  spd.SPDManifold              — P(n), symmetric positive definite,
                                 affine-invariant metric
  lorentz.LorentzManifold      — H^n_k, hyperboloid model of hyperbolic
                                 space of curvature k < 0 (Nickel-Kiela 2018)
  stereographic.KappaStereographicManifold — parametric κ ∈ R, interpolates
                                 spherical (κ>0), Euclidean (κ=0),
                                 hyperbolic (κ<0) via gyro-algebra
                                 (Bachmann-Bécigneul-Ganea 2020)
  lorentzian.LorentzianManifold — flat Minkowski spacetime R^{1, n-1}
                                 with pseudo-Riemannian (1, n-1)
                                 signature; causal structure (timelike/
                                 null/spacelike) + proper time/distance
                                 (MTW 1973; O'Neill 1983)

Planned (port/wrap from geoopt, geomstats, pymanopt):
  sphere, stiefel, grassmann, product
"""

from holonomy_lib.manifolds.fixed_rank import FixedRankManifold
from holonomy_lib.manifolds.lorentz import LorentzManifold
from holonomy_lib.manifolds.lorentzian import LorentzianManifold
from holonomy_lib.manifolds.spd import SPDManifold
from holonomy_lib.manifolds.stereographic import KappaStereographicManifold

__all__ = [
    "FixedRankManifold",
    "KappaStereographicManifold",
    "LorentzManifold",
    "LorentzianManifold",
    "SPDManifold",
]
