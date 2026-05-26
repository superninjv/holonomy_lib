"""Symmetric Lanczos iteration for top-k eigenpairs of a batched matrix.

For a symmetric matrix `A ∈ ℝ^{n×n}`, the Lanczos process builds an
orthonormal Krylov basis `V ∈ ℝ^{n × m}` (with `m ≪ n` typically) such
that the projected matrix `T = Vᵀ A V` is tridiagonal. Diagonalizing
`T` (a tiny `m × m` problem) gives Ritz values and Ritz vectors that
approximate the extreme eigenpairs of `A`. After `m` iterations,
typically `m / 2` of those approximate the `m / 2` most-extreme
eigenvalues (largest by magnitude) to machine precision; the others
converge more slowly.

Algorithm (Lanczos 1950, Paige 1972):

  V[:, 0] = v_0 / ‖v_0‖                                 # random unit vector
  β_{-1} = 0
  for j in 0, …, m − 1:
      w   = A · V[:, j]                                 # 1 matmul
      α_j = ⟨w, V[:, j]⟩
      w   = w − α_j · V[:, j] − β_{j-1} · V[:, j-1]    # 3-term recurrence
      w   = orthogonalize(w, V[:, :j+1])                # full reorth
      β_j = ‖w‖
      V[:, j+1] = w / β_j
  T = tridiag(α, β)
  ritz_vals, ritz_vecs = eigh(T)                        # small
  eigenvectors of A ≈ V[:, :m] @ ritz_vecs

For numerical robustness on finite-precision GPUs we use **full
reorthogonalization** at every step: the cheaper two-step
Gram-Schmidt and the partial reorthogonalization variants are
known to lose orthogonality of `V` once Ritz values converge
(Paige's "ghost eigenvalues"). Full reorth costs `O(m² · n)` extra
flops on top of the `O(m · n²)` baseline — fine when `m` is small.

The default convention is `which = "largest_algebraic"` (top-`k` by
value). For bottom-`k` of a known-bounded-spectrum operator (e.g.
the symmetric-normalized Laplacian with spectrum in `[0, 2]`), call
`lanczos_eigsh(2.0 * I − L_sym, k=k)` and recover smallest eigenvalues
of `L_sym` as `2.0 − ritz_vals`. Shift-and-invert for general
"smallest" mode is planned.

References:
  Lanczos, C. (1950). An iteration method for the solution of the
    eigenvalue problem of linear differential and integral
    operators. Journal of Research of the National Bureau of
    Standards 45:255–282. Original algorithm.
  Paige, C. C. (1972). Computational variants of the Lanczos method
    for the eigenproblem. Journal of the Institute of Mathematics
    and Its Applications 10:373–381. Identified loss of orthogonality
    and prescribed reorthogonalization.
  Saad, Y. (2011). Numerical Methods for Large Eigenvalue Problems,
    2nd ed. SIAM. §6.5 covers full reorthogonalization.
  Golub, G. H., Van Loan, C. F. (2013). Matrix Computations, 4th ed.
    Johns Hopkins University Press. §10.1.
"""

from __future__ import annotations

from typing import Optional

import torch

from holonomy_lib.provenance import with_provenance


# Default oversampling: extra Lanczos iterations beyond `k` to improve
# accuracy of the top-k Ritz values. Saad (2011), §6.5 recommends
# ~5–10 extra for reliable top-k convergence. Halko-Martinsson-Tropp
# (2011), §1.2 use the same value (5) for randomized SVD, which is
# a closely related Krylov-projection scheme. Cataloged as
# `lanczos_oversample_default`.
LANCZOS_OVERSAMPLE_DEFAULT: int = 10


_SPARSE_LAYOUTS = (
    torch.sparse_coo, torch.sparse_csr, torch.sparse_csc,
)


@with_provenance(
    "holonomy_lib.algebra.lanczos_eigsh", op_version="0.2",
)
def lanczos_eigsh(
    A: torch.Tensor,
    k: int,
    n_iter: Optional[int] = None,
    oversample: int = LANCZOS_OVERSAMPLE_DEFAULT,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k largest-algebraic eigenpairs of a symmetric matrix.

    Dispatches on layout: dense batched `(B, n, n)` runs the standard
    path; sparse `(n, n)` (CSR/CSC/COO) runs a single-instance path
    with sparse matmul on each iteration. Sparse input must be 2-D
    (no batch); the output shape is `(k,)` / `(n, k)` without a batch
    dim. For batched sparse you can either iterate yourself or convert
    to a sparse batched COO and call repeatedly.

    Args:
      A: dense `(B, n, n)` or sparse-CSC/CSR/COO `(n, n)` symmetric.
        We assume but do not check `A == A.T`; violation produces
        meaningless results.
      k: number of eigenpairs to return. `1 ≤ k ≤ n`.
      n_iter: total Lanczos iterations. Default `k + oversample`,
        clamped to `n`. More iterations → better convergence to interior
        eigenvalues but more compute.
      oversample: extra iterations beyond `k`. Default 10.
      generator: torch.Generator for the random starting vector.

    Returns:
      Dense input  → `(eigvals: (B, k), eigvecs: (B, n, k))`.
      Sparse input → `(eigvals: (k,), eigvecs: (n, k))`.

      Eigenvalues descending in value; eigenvectors orthonormal with
      columns aligned to the eigenvalues.

    Notes:
      Cost is `O(B · n_iter · n²)` for the matmuls plus
      `O(B · n_iter² · n)` for full reorthogonalization. For `n_iter ≪ n`
      this is much cheaper than the `O(B · n³)` dense `torch.linalg.eigh`.

      Convergence: the top-`oversample` Ritz values are the most
      accurate; the worst-converged of the returned `k` may have
      error `O(λ_{k+1} / λ_k)^{2·oversample}` in the typical case
      (Saad 2011, §6.7).

    References:
      Lanczos (1950); Paige (1972); Saad (2011), §6.5.
    """
    # Dispatch on layout. Sparse inputs go through a 2-D no-batch path.
    if A.layout in _SPARSE_LAYOUTS:
        return _lanczos_sparse(A, k, n_iter, oversample, generator)
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(
            f"A must be (..., n, n); got A.shape={tuple(A.shape)}"
        )
    n = A.shape[-1]
    if k <= 0 or k > n:
        raise ValueError(f"k must satisfy 1 <= k <= n={n}, got k={k}")

    # Total Lanczos iterations. Cap at n (the Krylov subspace can't
    # exceed n dimensions in exact arithmetic).
    if n_iter is None:
        n_iter = k + oversample
    n_iter = min(n_iter, n)
    if n_iter < k:
        raise ValueError(
            f"n_iter={n_iter} must be >= k={k} to return k eigenpairs"
        )

    *batch, _ = A.shape[:-1]
    device, dtype = A.device, A.dtype

    # Initial random unit vector. Same per-batch shape conventions.
    v0 = torch.randn(*batch, n, generator=generator, device=device, dtype=dtype)
    v0 = v0 / torch.linalg.norm(v0, dim=-1, keepdim=True)

    # Pre-allocate the Lanczos basis tensor and accumulators. Writing
    # in-place to `V[..., j]` avoids the O(m² · n) of re-stacking
    # `V_cols` every iteration. The (n_iter)-th column is the boundary
    # vector v_{n_iter}; it's computed inside the loop for the
    # 3-term recurrence but not used in the projected problem.
    V = torch.empty(*batch, n, n_iter + 1, device=device, dtype=dtype)
    V[..., 0] = v0
    alphas: list[torch.Tensor] = []                     # each (B,)
    betas: list[torch.Tensor] = []                      # each (B,)

    # Beta_{-1} = 0, v_{-1} = 0 (3-term recurrence boundary condition).
    v_prev = torch.zeros_like(v0)
    beta_prev = torch.zeros(*batch, device=device, dtype=dtype)

    for j in range(n_iter):
        v_curr = V[..., j]
        # Matmul: A · v_curr.
        w = torch.matmul(A, v_curr.unsqueeze(dim=-1)).squeeze(dim=-1)
        alpha_j = (w * v_curr).sum(dim=-1)
        alphas.append(alpha_j)

        # 3-term recurrence: w = w − α_j v_curr − β_{j−1} v_prev.
        w = (
            w
            - alpha_j.unsqueeze(dim=-1) * v_curr
            - beta_prev.unsqueeze(dim=-1) * v_prev
        )

        # Full reorthogonalization against all V[:, :j+1]. Slicing
        # the pre-allocated tensor is O(1) — no copy.
        V_used = V[..., :j + 1]                          # (B, n, j+1)
        coeffs = torch.matmul(
            V_used.mT, w.unsqueeze(dim=-1),
        ).squeeze(dim=-1)                                 # (B, j+1)
        w = w - torch.matmul(V_used, coeffs.unsqueeze(dim=-1)).squeeze(dim=-1)

        beta_j = torch.linalg.norm(w, dim=-1)             # (B,)
        # Avoid division by zero on Lanczos breakdown. We clamp to the
        # dtype's tiny positive; the resulting basis vector is
        # numerical noise but doesn't propagate NaN. In batched code we
        # can't selectively stop per-batch, so this is the safest
        # behaviour. (Production-grade: replace with a fresh random
        # vector and continue — left as future work.)
        floor = torch.finfo(dtype).tiny
        safe_beta = beta_j.clamp(min=floor)
        V[..., j + 1] = w / safe_beta.unsqueeze(dim=-1)

        betas.append(beta_j)
        v_prev = v_curr
        beta_prev = beta_j

    # The Krylov basis we project onto is the first `n_iter` columns
    # of V; the (n_iter)-th column is the boundary vector for the
    # 3-term recurrence and is not part of the projected problem.
    V_basis = V[..., :n_iter]                             # (B, n, n_iter)

    # Build tridiagonal T = diag(alpha) + offdiag(beta).
    # alphas: list of (B,) of length n_iter
    # betas:  list of (B,) of length n_iter (we only need the first
    # n_iter − 1 for the tridiagonal off-diagonal).
    T = _build_tridiagonal(alphas, betas[:-1] if len(betas) >= 1 else [])

    # Solve the small (B, n_iter, n_iter) symmetric eigenproblem.
    ritz_vals, ritz_vecs = torch.linalg.eigh(T)            # ascending

    # Approximate eigenvectors of A.
    approx_eigvecs = torch.matmul(V_basis, ritz_vecs)      # (B, n, n_iter)

    # Take top-k by VALUE (descending). eigh returns ascending; flip.
    top_vals = ritz_vals.flip(dims=(-1,))[..., :k]
    top_vecs = approx_eigvecs.flip(dims=(-1,))[..., :k]
    return top_vals, top_vecs


# ============================================================
# Internal helpers
# ============================================================


def _lanczos_sparse(
    A: torch.Tensor,
    k: int,
    n_iter: Optional[int],
    oversample: int,
    generator: Optional[torch.Generator],
) -> tuple[torch.Tensor, torch.Tensor]:
    """2-D no-batch Lanczos for sparse `A`.

    Mirrors the dense path's algorithm — the only difference is that
    `A @ v` goes through sparse matmul, and there's no leading batch
    dimension. Used by `betti_numbers` on sparse Hodge Laplacians and
    by callers who pass sparse `A` directly.

    Limitations:
      - 2-D only (no batch). Caller wraps for batched sparse cases.
      - Provenance recording of sparse `A` reads its dense form via
        `to_dense()` before hashing, which negates the memory win of
        sparse inputs inside `record()`. Outside `record()` the
        decorator is transparent and this doesn't matter.
    """
    if A.ndim != 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(
            f"sparse A must be (n, n); got A.shape={tuple(A.shape)}"
        )
    n = A.shape[-1]
    if k <= 0 or k > n:
        raise ValueError(f"k must satisfy 1 <= k <= n={n}, got k={k}")

    if n_iter is None:
        n_iter = k + oversample
    n_iter = min(n_iter, n)
    if n_iter < k:
        raise ValueError(
            f"n_iter={n_iter} must be >= k={k} to return k eigenpairs"
        )

    device, dtype = A.device, A.dtype

    v0 = torch.randn(n, generator=generator, device=device, dtype=dtype)
    v0 = v0 / torch.linalg.norm(v0)

    V = torch.empty(n, n_iter + 1, device=device, dtype=dtype)
    V[:, 0] = v0
    alphas: list[torch.Tensor] = []
    betas: list[torch.Tensor] = []

    v_prev = torch.zeros_like(v0)
    beta_prev = torch.zeros((), device=device, dtype=dtype)

    for j in range(n_iter):
        v_curr = V[:, j]
        # Sparse matmul: A @ v. For sparse-CSC + dense vector this
        # uses the appropriate spmv kernel. The (n, 1) shape is needed
        # because torch.matmul on sparse expects a 2-D RHS.
        w = torch.matmul(A, v_curr.unsqueeze(-1)).squeeze(-1)
        alpha_j = (w * v_curr).sum()
        alphas.append(alpha_j)

        w = w - alpha_j * v_curr - beta_prev * v_prev

        # Full reorthogonalization against V[:, :j+1] (all dense).
        V_used = V[:, :j + 1]                            # (n, j+1)
        coeffs = V_used.mT @ w                            # (j+1,)
        w = w - V_used @ coeffs

        beta_j = torch.linalg.norm(w)
        floor = torch.finfo(dtype).tiny
        safe_beta = beta_j.clamp(min=floor)
        V[:, j + 1] = w / safe_beta

        betas.append(beta_j)
        v_prev = v_curr
        beta_prev = beta_j

    V_basis = V[:, :n_iter]                              # (n, n_iter)

    # Build tridiagonal T = diag(alpha) + offdiag(beta).
    # alphas/betas here are 0-D tensors; stack and use the dense path.
    diag = torch.stack(alphas)                            # (n_iter,)
    T = torch.diag(diag)
    if len(betas) > 1:
        off = torch.stack(betas[:-1])                     # (n_iter - 1,)
        T = T + torch.diag(off, diagonal=1) + torch.diag(off, diagonal=-1)

    ritz_vals, ritz_vecs = torch.linalg.eigh(T)
    approx_eigvecs = V_basis @ ritz_vecs

    top_vals = ritz_vals.flip(dims=(0,))[:k]
    top_vecs = approx_eigvecs.flip(dims=(-1,))[:, :k]
    return top_vals, top_vecs


def _build_tridiagonal(
    alphas: list[torch.Tensor],
    betas: list[torch.Tensor],
) -> torch.Tensor:
    """Build a batched tridiagonal matrix from diagonal + sub/super-diagonal.

    Args:
      alphas: list of length n, each (B,). The diagonal entries.
      betas: list of length n − 1, each (B,). The sub/super-diagonal entries
        (the matrix is symmetric).

    Returns:
      T: (B, n, n) tridiagonal.
    """
    n = len(alphas)
    # Explicit ValueError (rather than assert) so this guard survives
    # `python -O`: an internal indexing mismatch would silently
    # produce a wrong-sized T and corrupt the Ritz pairs.
    if len(betas) != n - 1:
        raise ValueError(
            f"need {n - 1} off-diagonal entries for {n}x{n} tridiagonal, "
            f"got {len(betas)}"
        )
    diag = torch.stack(alphas, dim=-1)                    # (B, n)
    T = torch.diag_embed(diag)
    if n > 1:
        off = torch.stack(betas, dim=-1)                  # (B, n-1)
        # Place off on both sub- and super-diagonal.
        T = T + torch.diag_embed(off, offset=1) + torch.diag_embed(off, offset=-1)
    return T
