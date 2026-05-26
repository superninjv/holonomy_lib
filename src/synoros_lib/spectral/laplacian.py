"""Graph Laplacians, batched-first, GPU-native.

Given a weighted adjacency matrix `A` of shape `(B, n, n)`, this module
constructs the standard real-valued Laplacian variants. All variants
assume `A` is symmetric — non-symmetric (directed) graphs need the
magnetic Laplacian (planned).

Variants:
  combinatorial(A)          — L = D − A.            Always PSD.
  symmetric_normalized(A)   — L_sym = I − D^{−1/2} A D^{−1/2}.
                              PSD, eigenvalues in [0, 2].
  random_walk(A)            — L_rw = I − D^{−1} A.
                              Eigenvalues real and in [0, 2] (similar to L_sym).
  signed(A)                 — L^σ = D^{|σ|} − A,
                              D^{|σ|}_{ii} = Σ_j |A_{ij}|.
                              PSD even with negative edge weights (Kunegis 2010).

Isolated nodes (zero degree) are handled by the Moore-Penrose convention:
when D_{ii} = 0, the corresponding row/column of D^{−1/2} and D^{−1} is
set to zero. This makes L_sym and L_rw well-defined for arbitrary graphs;
the isolated node then sits at eigenvalue 1 (for L_rw and the
combinatorial form) or 0 (for L_sym). Convention follows Cheng-Wu (2024)
SIAM J. Math. Data Sci.

References:
  Chung, F. R. K. (1997). Spectral Graph Theory. CBMS Regional Conference
    Series in Mathematics, 92. American Mathematical Society.
  von Luxburg, U. (2007). A tutorial on spectral clustering. Statistics
    and Computing 17(4):395–416.
  Kunegis, J., Schmidt, S., Lommatzsch, A., Lerner, J., De Luca, E. W.,
    Albayrak, S. (2010). Spectral analysis of signed graphs for
    clustering, prediction and visualization. Proceedings of the 2010
    SIAM International Conference on Data Mining, 559–570.
  Ribando-Gros, E., Wang, R., Chen, J., Tong, Y., Wei, G.-W. (2024).
    Combinatorial and Hodge Laplacians: Similarities and differences.
    SIAM Review 66(3):575–601.
  Cheng, X., Wu, N. (2024). Bi-stochastically normalized graph
    Laplacian: convergence to manifold Laplacian and robustness to
    outlier noise. Information and Inference: A Journal of the IMA.
"""

from __future__ import annotations

import torch

from synoros_lib.provenance import with_provenance


def degree(A: torch.Tensor, signed: bool = False) -> torch.Tensor:
    """Degree vector of a (batched) weighted adjacency matrix.

    Args:
      A: (B, n, n) weighted adjacency. Symmetry is assumed.
      signed: if True, use absolute weights for the degree (∑_j |A_ij|),
        the Kunegis (2010) convention used in the signed Laplacian.
        Otherwise use the raw weighted degree (∑_j A_ij), which is the
        standard definition for non-negative weights.

    Returns:
      (B, n) degree vector.

    References:
      Chung (1997), §1.2 — weighted degree.
      Kunegis et al. (2010), eq. 1 — signed degree.
    """
    _check_square_with_batch(A)
    return A.abs().sum(dim=-1) if signed else A.sum(dim=-1)


@with_provenance("synoros_lib.spectral.laplacian.combinatorial", op_version="0.1")
def combinatorial(A: torch.Tensor) -> torch.Tensor:
    """Combinatorial Laplacian L = D − A.

    For symmetric `A` with non-negative weights, L is PSD with eigenvalue
    0 of multiplicity equal to the number of connected components.

    Args:
      A: (B, n, n) symmetric weighted adjacency.
    Returns:
      (B, n, n) combinatorial Laplacian.

    References:
      Chung (1997), §1.2.
      von Luxburg (2007), §3 — unnormalized Laplacian.
    """
    _check_square_with_batch(A)
    d = degree(A, signed=False)              # (B, n)
    return torch.diag_embed(d) - A


@with_provenance("synoros_lib.spectral.laplacian.symmetric_normalized", op_version="0.1")
def symmetric_normalized(A: torch.Tensor) -> torch.Tensor:
    """Symmetric normalized Laplacian L_sym = I − D^{−1/2} A D^{−1/2}.

    PSD with spectrum in [0, 2]. The factor D^{−1/2} uses the
    Moore-Penrose convention for isolated nodes (set to 0 where D = 0).

    Args:
      A: (B, n, n) symmetric weighted adjacency, non-negative weights.
    Returns:
      (B, n, n) normalized Laplacian.

    References:
      Chung (1997), §1.2.
      von Luxburg (2007), §3 — L_sym.
      Cheng-Wu (2024) — pseudoinverse handling of isolated nodes.
    """
    _check_square_with_batch(A)
    d = degree(A, signed=False)               # (B, n)
    d_inv_sqrt = _safe_inv_sqrt(d)            # (B, n) — zeros where d=0
    n = A.shape[-1]
    eye = _batched_eye(n, batch=A.shape[:-2], device=A.device, dtype=A.dtype)
    # D^{-1/2} A D^{-1/2}  via broadcast multiplication
    A_norm = A * d_inv_sqrt.unsqueeze(dim=-1) * d_inv_sqrt.unsqueeze(dim=-2)
    return eye - A_norm


def random_walk(A: torch.Tensor) -> torch.Tensor:
    """Random-walk Laplacian L_rw = I − D^{−1} A.

    Has the same eigenvalues as L_sym (it is similar to L_sym via D^{1/2}
    conjugation). Eigenvectors of L_rw are the random-walk transition
    eigenvectors, useful for clustering and diffusion. Spectrum in [0, 2].

    Args:
      A: (B, n, n) symmetric weighted adjacency, non-negative weights.
    Returns:
      (B, n, n) random-walk Laplacian.

    References:
      Chung (1997), §1.5.
      von Luxburg (2007), §3 — L_rw.
    """
    _check_square_with_batch(A)
    d = degree(A, signed=False)               # (B, n)
    d_inv = _safe_inv(d)                       # (B, n) — zeros where d=0
    n = A.shape[-1]
    eye = _batched_eye(n, batch=A.shape[:-2], device=A.device, dtype=A.dtype)
    A_rw = A * d_inv.unsqueeze(dim=-1)         # broadcast: (B, n, n) * (B, n, 1)
    return eye - A_rw


def signed(A: torch.Tensor) -> torch.Tensor:
    """Signed Laplacian L^σ = D^{|σ|} − A,  D^{|σ|}_{ii} = Σ_j |A_{ij}|.

    For symmetric `A` with arbitrary real weights (positive AND negative),
    L^σ is PSD. The eigenvalue 0 corresponds to *balanced* connected
    components of the signed graph (Kunegis 2010, Theorem 3.4): a
    signed graph is balanced iff zero is in the spectrum of L^σ.

    Args:
      A: (B, n, n) symmetric weighted adjacency, weights may be negative.
    Returns:
      (B, n, n) signed Laplacian, PSD.

    References:
      Kunegis et al. (2010), eq. 5 — definition of the signed Laplacian.
      Kunegis et al. (2010), Theorem 3.4 — PSD property and balance.
      Mercado-Tudisco-Hein (2019), NeurIPS — Signed Power Mean Laplacians
        generalize this to a one-parameter family.
    """
    _check_square_with_batch(A)
    d_abs = degree(A, signed=True)             # (B, n) — ∑_j |A_ij|
    return torch.diag_embed(d_abs) - A


# ----------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------


def _check_square_with_batch(A: torch.Tensor) -> None:
    """A must be ≥3-D with the last two dims square: shape (..., n, n)."""
    if A.ndim < 2:
        raise ValueError(
            f"A must have at least one batch dim and be square (..., n, n); "
            f"got A.shape={tuple(A.shape)}"
        )
    if A.shape[-1] != A.shape[-2]:
        raise ValueError(
            f"A must be square in the last two dims; got A.shape={tuple(A.shape)}"
        )


def _safe_inv_sqrt(d: torch.Tensor) -> torch.Tensor:
    """Pseudoinverse of D^{1/2}: 1/√d where d > 0, zero where d = 0.

    Moore-Penrose convention for isolated nodes (Cheng-Wu 2024).
    """
    out = torch.zeros_like(d)
    mask = d > 0
    out[mask] = torch.rsqrt(d[mask])
    return out


def _safe_inv(d: torch.Tensor) -> torch.Tensor:
    """Pseudoinverse of D: 1/d where d > 0, zero where d = 0.

    Moore-Penrose convention for isolated nodes (Cheng-Wu 2024).
    """
    out = torch.zeros_like(d)
    mask = d > 0
    out[mask] = torch.reciprocal(d[mask])
    return out


def _batched_eye(
    n: int,
    batch: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Identity matrix broadcast to the given batch shape: (*batch, n, n)."""
    eye = torch.eye(n, device=device, dtype=dtype)
    return eye.expand(*batch, n, n)
