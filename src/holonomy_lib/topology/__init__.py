"""holonomy_lib.topology: Hodge Laplacians + persistent homology.

Built on `holonomy_lib.simplicial`. Two flavors of output:

  Hodge / Betti
    `hodge_laplacian(complex, k)` — the discrete Laplace-Beltrami
    operator on k-chains: `L_k = ∂_{k+1} ∂_{k+1}^T + ∂_k^T ∂_k`. Its
    kernel has dimension equal to the k-th Betti number.
    `betti_numbers(complex, max_dim)` — counts near-zero eigenvalues
    of each `L_k` to recover `(β_0, β_1, …, β_max_dim)`.

  Persistent homology (added in Phase 4)
    `persistence_diagrams(points, max_dim)` returns the per-dim
    birth/death barcodes for a Vietoris-Rips filtration.
"""

from holonomy_lib.topology.hodge import betti_numbers, hodge_laplacian

__all__ = [
    "betti_numbers",
    "hodge_laplacian",
]
