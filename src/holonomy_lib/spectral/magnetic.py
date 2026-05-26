"""Magnetic Laplacian for directed and signed-directed graphs.

The standard real-valued graph Laplacians (`spectral.laplacian.*`)
assume a symmetric adjacency. For a directed graph the asymmetry
encodes information that any symmetrization throws away. The
**magnetic Laplacian** (Lieb-Loss 1993; Fanuel et al. 2017; Furutani
et al. 2020) recovers that information by attaching a complex phase
factor `e^{i · 2π · q · (A_{ij} - A_{ji})}` to each edge, while keeping
the magnitude structure symmetric. The result is a Hermitian operator
whose eigenvalues are real and whose eigenvectors are complex; the
phase of those eigenvectors carries the directional information.

Definition (combinatorial form):

  A_s   = (A + A^T) / 2                                (symmetrized)
  D_s   = diag(Σ_j A_s[i, j])                          (symmetric degree)
  H     = exp(i · 2π · q · (A - A^T))                  (Hermitian phase factor)
  L^(q) = D_s - H ⊙ A_s

For symmetric A this reduces to L = D - A regardless of q. For purely
directed A (entries in {0, 1}, A ≠ A^T) the phase encodes "flow
direction": q = 1/4 separates directed eigenmodes most clearly
(Furutani 2020, §5).

Two forms exposed:
  combinatorial(A, q)           — L^(q) = D_s − H ⊙ A_s
  symmetric_normalized(A, q)    — L_sym^(q) = I − D_s^{−1/2} (H ⊙ A_s) D_s^{−1/2}

Both return a complex Hermitian tensor of shape (B, n, n). Use
`torch.linalg.eigh` on the result — its real-eigenvalue output is the
spectrum and its complex eigenvectors are the directed-graph modes.

References:
  Lieb, E. H., Loss, M. (1993). Fluxes, Laplacians, and Kasteleyn's
    theorem. Duke Mathematical Journal 71(2):337–363. The original
    magnetic / flux-discrete Laplacian on graphs.
  Fanuel, M., Alaiz, C. M., Suykens, J. A. K. (2017). Magnetic
    eigenmaps for the visualization of directed networks. Applied and
    Computational Harmonic Analysis 44(1):189–199. The modern spectral-
    embedding application.
  Furutani, S., Shibahara, T., Akiyama, M., Hato, K., Aida, M. (2020).
    Graph Signal Processing for Directed Graphs based on the Hermitian
    Laplacian. Journal of Machine Learning Research 21(122):1–37.
    Gives the q = 1/4 default and the normalization conventions used
    here. Section 5 covers the spectral interpretation.
"""

from __future__ import annotations

import math

import torch

from holonomy_lib.provenance import with_provenance


# Default magnetic charge q ∈ [0, 1) per Furutani et al. (2020), §5.
# `q = 1/4` is the value that maximally separates the directed
# eigenmodes for a cycle and is therefore the standard literature
# default. **Scale of validity**: dimensionless ratio of magnetic flux
# units; does not depend on n. Cataloged as `magnetic_charge_default`.
MAGNETIC_CHARGE_DEFAULT: float = 0.25


@with_provenance(
    "holonomy_lib.spectral.magnetic.combinatorial", op_version="0.1",
)
def combinatorial(
    A: torch.Tensor, q: float = MAGNETIC_CHARGE_DEFAULT,
) -> torch.Tensor:
    """Combinatorial magnetic Laplacian L^(q) = D_s − H ⊙ A_s.

    Args:
      A: (B, n, n) weighted adjacency. May be non-symmetric. Entries
        may be negative (signed-directed graphs); the symmetrized
        adjacency `A_s = (A + A^T)/2` keeps the "edge presence",
        and the antisymmetric part `A − A^T` enters via the phase.
      q: magnetic charge in [0, 1). Default 1/4 per Furutani 2020 §5.

    Returns:
      L: (B, n, n) complex Hermitian Laplacian. Use `torch.linalg.eigh`
        to get real eigenvalues + complex eigenvectors.

    References:
      Furutani et al. (2020), eq. 8.
      Lieb-Loss (1993).
    """
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(
            f"A must be (..., n, n); got A.shape={tuple(A.shape)}"
        )
    if not (0.0 <= q < 1.0):
        raise ValueError(f"magnetic charge q must be in [0, 1), got {q}")

    A_s = 0.5 * (A + A.mT)                       # symmetrized magnitude
    D_s = A_s.sum(dim=-1)                         # (B, n) symmetric degree

    H_times_As = _magnetic_phase_times_adj(A, A_s, q)

    # D_s on the diagonal, embedded in the complex dtype.
    D_diag = torch.diag_embed(D_s).to(H_times_As.dtype)
    return D_diag - H_times_As


@with_provenance(
    "holonomy_lib.spectral.magnetic.symmetric_normalized", op_version="0.1",
)
def symmetric_normalized(
    A: torch.Tensor, q: float = MAGNETIC_CHARGE_DEFAULT,
) -> torch.Tensor:
    """Symmetric-normalized magnetic Laplacian.

    L_sym^(q) = I − D_s^{−1/2} (H ⊙ A_s) D_s^{−1/2}

    Spectrum ⊂ [0, 2] regardless of q (Furutani 2020, Prop. 1). Use
    this form for spectral embeddings of directed graphs; the
    eigenvectors at the bottom-k eigenvalues are the directed analog
    of Laplacian eigenmaps.

    Args:
      A: (B, n, n) weighted adjacency. May be non-symmetric.
      q: magnetic charge in [0, 1).

    Returns:
      L_sym^(q): (B, n, n) complex Hermitian, spectrum in [0, 2].

    References:
      Furutani et al. (2020), Definition 4 and Prop. 1.
    """
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(
            f"A must be (..., n, n); got A.shape={tuple(A.shape)}"
        )
    if not (0.0 <= q < 1.0):
        raise ValueError(f"magnetic charge q must be in [0, 1), got {q}")

    A_s = 0.5 * (A + A.mT)
    D_s = A_s.sum(dim=-1)                         # (B, n)

    # Moore-Penrose pseudoinverse handling for isolated nodes — same
    # convention as `holonomy_lib.spectral.laplacian.symmetric_normalized`.
    # The library-wide `1e-9` (numerical_floor_convention) doubles as
    # the dtype-tiny floor below: for any reasonable dtype, `tiny` is
    # well below 1e-9, so `where(D_s > 0, ...)` selects the same set.
    floor = torch.finfo(A.dtype).tiny
    safe_D = D_s.clamp(min=floor)
    d_inv_sqrt = torch.where(
        D_s > 0, torch.rsqrt(safe_D), torch.zeros_like(D_s),
    )                                              # (B, n)

    H_times_As = _magnetic_phase_times_adj(A, A_s, q)

    # D^{-1/2} H ⊙ A_s D^{-1/2} via broadcasting.
    scale = (
        d_inv_sqrt.unsqueeze(dim=-1) * d_inv_sqrt.unsqueeze(dim=-2)
    ).to(H_times_As.dtype)
    normalized = H_times_As * scale

    n = A.shape[-1]
    eye = torch.eye(n, device=A.device, dtype=H_times_As.dtype).expand_as(
        normalized,
    )
    return eye - normalized


# ============================================================
# Internal helpers
# ============================================================


def _magnetic_phase_times_adj(
    A: torch.Tensor, A_s: torch.Tensor, q: float,
) -> torch.Tensor:
    """Return H ⊙ A_s where H_{ij} = exp(i · 2π · q · (A_{ij} - A_{ji})).

    Computed as a single complex tensor of shape (*A_s.shape).
    For symmetric A the phase is identically 1 and the result reduces
    to A_s + 0j — but we still pay the complex allocation. For q = 0
    the phase is also identically 1; we short-circuit that case.
    """
    if q == 0.0:
        # Phase is identically 1; result is A_s as a complex tensor.
        # Cast to the appropriate complex dtype matching A.dtype.
        complex_dtype = _matching_complex_dtype(A.dtype)
        return A_s.to(complex_dtype)

    # Antisymmetric part of A; for purely symmetric inputs this is 0
    # and we land in the q-doesn't-matter regime anyway.
    asym = A - A.mT                                # (B, n, n)
    angle = (2.0 * math.pi * q) * asym             # (B, n, n) real
    phase = torch.complex(torch.cos(angle), torch.sin(angle))
    complex_dtype = phase.dtype
    return phase * A_s.to(complex_dtype)


def _matching_complex_dtype(real_dtype: torch.dtype) -> torch.dtype:
    """Pick a complex dtype matching the precision of a real dtype."""
    if real_dtype == torch.float64:
        return torch.complex128
    if real_dtype == torch.float32:
        return torch.complex64
    # Fallback for half-precision and others — go up to complex64.
    return torch.complex64
