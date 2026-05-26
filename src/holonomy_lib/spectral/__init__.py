"""holonomy_lib.spectral: graph Laplacians and spectral utilities.

Currently implemented:
  laplacian.combinatorial(A)          : L = D - A
  laplacian.symmetric_normalized(A)   : L_sym = I - D^{-1/2} A D^{-1/2}
  laplacian.random_walk(A)            : L_rw = I - D^{-1} A
  laplacian.signed(A)                 : L^sigma = D^{|sigma|} - A
                                          (Kunegis 2010)
  laplacian.degree(A)                 : degree vector (signed-aware)
  magnetic.combinatorial(A, q)        : Hermitian Laplacian on directed
                                          graphs (Furutani 2020, Fanuel 2017)
  magnetic.symmetric_normalized(A, q) : normalized magnetic Laplacian,
                                          spectrum in [0, 2]
  laplacian_eigenmaps(A, k, ...)      : bottom-k spectral embedding

Planned:
  Hodge Laplacians on simplicial complexes (Ribando-Gros et al. 2024,
    Schaub et al. 2020).
  Lanczos solver for sparse top-K eigenvectors on GPU.
  Heat-kernel computation via Chebyshev polynomials.
  Effective resistance / commute-time distances.
"""

from holonomy_lib.spectral import laplacian, magnetic
from holonomy_lib.spectral.embedding import laplacian_eigenmaps
from holonomy_lib.spectral.heat_kernel import heat_kernel_chebyshev

__all__ = [
    "heat_kernel_chebyshev",
    "laplacian",
    "laplacian_eigenmaps",
    "magnetic",
]
