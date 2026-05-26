"""synoros_lib.spectral — graph Laplacians and spectral utilities.

Currently implemented:
  laplacian.combinatorial(A)          — L = D − A
  laplacian.symmetric_normalized(A)   — L_sym = I − D^{−1/2} A D^{−1/2}
  laplacian.random_walk(A)            — L_rw = I − D^{−1} A
  laplacian.signed(A)                 — L^σ = D^{|σ|} − A   (Kunegis 2010)
  laplacian.degree(A)                 — degree vector (signed-aware)

Planned (per HANDOFF.md §4):
  Sign-magnetic / magnetic Laplacian for signed-directed graphs
    (Fiorini et al. 2023, He et al. 2023).
  Hodge Laplacians on simplicial complexes
    (Ribando-Gros et al. 2024, Schaub et al. 2020).
  Lanczos solver for sparse top-K eigenvectors on GPU.
  Spectral embedding (Laplacian eigenmaps) + diffusion maps.
  Heat-kernel computation via Chebyshev polynomials.
  Effective resistance / commute-time distances.
"""

from synoros_lib.spectral import laplacian
from synoros_lib.spectral.embedding import laplacian_eigenmaps

__all__ = ["laplacian", "laplacian_eigenmaps"]
