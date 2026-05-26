"""holonomy_lib.info_geometry: information-geometric primitives.

Information geometry studies probability distributions as points on a
Riemannian manifold whose metric is the Fisher information. Two
families of primitives appear here:

  Bregman divergences   — `bregman_divergence(p, q, potential)`.
    The asymmetric "distance" induced by a convex potential function.
    Generalizes squared-Euclidean (potential = ½‖·‖²) and Itakura-
    Saito (potential = −log(·)) and Kullback-Leibler (potential =
    Σ p log p, the negative entropy). Banerjee et al. (2005) is the
    canonical reference.

  KL divergences        — closed forms for the common exponential
    families:
      `kl_divergence_categorical(p, q)`  — discrete distributions.
      `kl_divergence_gaussian(mu_p, Sigma_p, mu_q, Sigma_q)` — multi-
        variate Gaussians.

All primitives are batched-first, GPU-native, with citations.

References:
  Amari, S.-I. (2016). Information Geometry and Its Applications.
    Applied Mathematical Sciences 194, Springer. The textbook.
  Banerjee, A., Merugu, S., Dhillon, I. S., Ghosh, J. (2005).
    Clustering with Bregman divergences. JMLR 6:1705–1749.
  Nielsen, F. (2020). An elementary introduction to information
    geometry. Entropy 22(10):1100.
"""

from holonomy_lib.info_geometry.divergences import (
    bregman_divergence,
    kl_divergence_categorical,
    kl_divergence_gaussian,
)

__all__ = [
    "bregman_divergence",
    "kl_divergence_categorical",
    "kl_divergence_gaussian",
]
