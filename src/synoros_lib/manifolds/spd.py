"""SPD (symmetric positive definite) matrix manifold, GPU-native, batched-first.

The SPD manifold

    P(n) = { S ∈ R^{n × n} : S = Sᵀ, x ⊺ S x > 0 ∀ x ≠ 0 }

is an open submanifold of the symmetric matrices Sym(n) ⊂ R^{n × n} of
dimension n(n+1)/2. The tangent space at any S ∈ P(n) is canonically
identified with Sym(n) (it is the open cone, so the tangent at any
interior point is the ambient symmetric matrices).

This implementation uses the **affine-invariant metric** (Pennec et al.
2006):

    ⟨U, V⟩_S = tr(S⁻¹ U S⁻¹ V),   U, V ∈ Sym(n).

Under this metric the geodesics, exponential and logarithm maps admit
the closed forms

    exp_S(V) = S^{1/2} expm( S^{−1/2} V S^{−1/2} ) S^{1/2},
    log_S(T) = S^{1/2} logm( S^{−1/2} T S^{−1/2} ) S^{1/2},
    d(S, T)² = ‖ logm( S^{−1/2} T S^{−1/2} ) ‖_F² = Σ_i log²(λ_i),

where λ_i are the generalized eigenvalues solving T v = λ S v. The
affine-invariant metric is invariant under congruence (S ↦ Aᵀ S A for
invertible A), which is the property that makes it the canonical choice
for covariance-matrix and information-geometric work.

Tangent vectors are stored as `(B, n, n)` symmetric tensors.

References:
  Pennec, X., Fillard, P., Ayache, N. (2006). A Riemannian framework for
    tensor computing. International Journal of Computer Vision, 66(1):41–66.
  Bhatia, R. (2007). Positive Definite Matrices. Princeton University Press,
    chapters 4–6.
  Sra, S., Hosseini, R. (2015). Conic geometric optimization on the manifold
    of positive definite matrices. SIAM J. Optim. 25(1):713–739.
"""

from __future__ import annotations

from typing import Optional

import torch


class SPDManifold:
    """Affine-invariant SPD(n) manifold, GPU-native + batched-first.

    Args:
      n: matrix size (operations work on (B, n, n) symmetric tensors).
      device, dtype: tensor placement and precision.

    Example:
      >>> mfd = SPDManifold(n=4)
      >>> S = mfd.random_point(batch_size=3)
      >>> S.shape, mfd.is_spd(S).all().item()
      (torch.Size([3, 4, 4]), True)
    """

    def __init__(self, n: int,
                 device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.float64):
        if n <= 0:
            raise ValueError(f"n must be > 0, got n={n}")
        self.n = n
        self.device = torch.device(device)
        self.dtype = dtype

    @property
    def dim(self) -> int:
        """Manifold dimension n(n+1)/2.

        References:
          Pennec et al. (2006), §3.1.
        """
        # n*(n+1) is a derived even integer; integer division by 2 is exact.
        return self.n * (self.n + 1) // 2

    # ----------------------------------------------------------------
    # Construction
    # ----------------------------------------------------------------

    def random_point(
        self,
        batch_size: int = 1,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Sample B SPD matrices via the Wishart-like construction A Aᵀ.

        Construction: A ~ N(0, I)^{B × n × n}; return A Aᵀ. When A is
        full-rank (with probability one for Gaussian A), A Aᵀ is SPD.
        Distribution-wise this is a (rescaled) Wishart W_n(I, n); we do not
        normalize since callers typically only require *some* SPD draw for
        initialization / testing.

        References:
          Anderson, T. W. (2003). An Introduction to Multivariate Statistical
            Analysis, 3rd ed., chapter 7 — Wishart distribution.

        Returns:
          Tensor of shape (B, n, n).
        """
        if batch_size < 0:
            raise ValueError(f"batch_size must be >= 0, got {batch_size}")
        A = torch.randn(batch_size, self.n, self.n, generator=generator,
                        device=self.device, dtype=self.dtype)
        return torch.bmm(A, A.mT)

    def is_spd(self, S: torch.Tensor) -> torch.Tensor:
        """Per-batch test for SPD: symmetric and all eigenvalues > 0.

        Returns:
          (B,) boolean tensor.
        """
        sym_err = torch.linalg.matrix_norm(S - S.mT, dim=(-2, -1))
        sym_ref = torch.linalg.matrix_norm(S, dim=(-2, -1))
        # Treat as symmetric if asymmetric part is at numerical-noise level.
        is_sym = sym_err <= 1e-9 * torch.clamp(sym_ref, min=1.0)
        eigvals = torch.linalg.eigvalsh(0.5 * (S + S.mT))  # (B, n)
        is_pos = (eigvals > 0).all(dim=-1)
        return is_sym & is_pos

    # ----------------------------------------------------------------
    # Tangent operations
    # ----------------------------------------------------------------

    def projection(self, S: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """Project ambient direction Z onto the tangent space at S.

        T_S P(n) = Sym(n), so the projection is just symmetrization:

            P_T(Z) = ½ (Z + Zᵀ).

        Independent of the base point S (T_S P(n) does not depend on S in
        the embedded ambient form).

        References:
          Bhatia (2007), §6.1.

        Args:
          S: base point (B, n, n), unused (included for API consistency).
          Z: ambient direction (B, n, n).
        Returns:
          Symmetric tangent (B, n, n).
        """
        del S  # tangent space is point-independent in ambient form
        return 0.5 * (Z + Z.mT)

    def inner(
        self, S: torch.Tensor, U: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        """Affine-invariant inner product ⟨U, V⟩_S = tr(S⁻¹ U S⁻¹ V).

        Computed via solve rather than explicit inverse for numerical
        stability: X = S⁻¹ U is `linalg.solve(S, U)`, then
        Y = S⁻¹ V is `linalg.solve(S, V)`, then tr(X Y) by elementwise sum.

        Note: trace(AB) = Σ_ij A_ij B_ji = Σ_ij A_ij (Bᵀ)_ij, and since
        U, V are symmetric (in the tangent space) the result simplifies
        further to Σ A * B elementwise, but we use the general formula
        to keep the implementation correct for any ambient inputs.

        References:
          Pennec et al. (2006), eq. 4.

        Args:
          S: base point (B, n, n).
          U, V: tangent vectors (B, n, n).
        Returns:
          (B,) inner products.
        """
        S_inv_U = torch.linalg.solve(S, U)  # (B, n, n)
        S_inv_V = torch.linalg.solve(S, V)  # (B, n, n)
        # tr(X Y) = sum_{i,j} X_ij Y_ji = sum_{i,j} X_ij (Yᵀ)_ij
        return (S_inv_U * S_inv_V.mT).sum(dim=(-2, -1))

    def norm(self, S: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Riemannian norm sqrt(⟨V, V⟩_S). Shape (B,)."""
        return torch.sqrt(self.inner(S, V, V))

    # ----------------------------------------------------------------
    # Exponential and logarithmic maps, geodesic distance
    # ----------------------------------------------------------------

    @staticmethod
    def _eigh_symmetric_func(
        S: torch.Tensor, func
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply a scalar function to the eigenvalues of a symmetric SPD S.

        Returns (eigvals, eigvecs, S_func) where
          S_func = U diag(func(λ)) Uᵀ.

        Used to build S^{1/2}, S^{−1/2}, log S, etc.
        """
        eigvals, eigvecs = torch.linalg.eigh(S)  # (B, n), (B, n, n)
        func_vals = func(eigvals)
        S_func = torch.matmul(
            eigvecs * func_vals.unsqueeze(dim=-2), eigvecs.mT
        )
        return eigvals, eigvecs, S_func

    def _sqrt_and_inv_sqrt(
        self, S: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (S^{1/2}, S^{−1/2}) via eigh.

        Both are symmetric. Assumes S is SPD; eigenvalues are positive.
        """
        eigvals, eigvecs = torch.linalg.eigh(S)
        sqrt_eig = torch.sqrt(eigvals)
        inv_sqrt_eig = torch.reciprocal(sqrt_eig)
        S_sqrt = torch.matmul(eigvecs * sqrt_eig.unsqueeze(dim=-2), eigvecs.mT)
        S_inv_sqrt = torch.matmul(
            eigvecs * inv_sqrt_eig.unsqueeze(dim=-2), eigvecs.mT
        )
        return S_sqrt, S_inv_sqrt

    def exp(self, S: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Exponential map exp_S(V) = S^{1/2} expm(S^{−1/2} V S^{−1/2}) S^{1/2}.

        References:
          Pennec et al. (2006), eq. 5.

        Args:
          S: base point (B, n, n), SPD.
          V: tangent (B, n, n), symmetric.
        Returns:
          (B, n, n) SPD result.
        """
        S_sqrt, S_inv_sqrt = self._sqrt_and_inv_sqrt(S)
        inner = S_inv_sqrt @ V @ S_inv_sqrt
        # Symmetrize against numerical drift before matrix_exp; result of
        # symmetric @ symmetric @ symmetric is symmetric in exact arithmetic.
        inner = 0.5 * (inner + inner.mT)
        return S_sqrt @ torch.matrix_exp(inner) @ S_sqrt

    def log(self, S: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Logarithm map log_S(T) = S^{1/2} logm(S^{−1/2} T S^{−1/2}) S^{1/2}.

        Matrix log via eigh: logm(M) = U diag(log λ) Uᵀ for M = U diag(λ) Uᵀ
        symmetric. The inner argument is symmetric SPD (congruence preserves
        positivity), so this is well-defined.

        References:
          Pennec et al. (2006), eq. 6.

        Args:
          S, T: base point and target, both (B, n, n) SPD.
        Returns:
          (B, n, n) symmetric tangent.
        """
        S_sqrt, S_inv_sqrt = self._sqrt_and_inv_sqrt(S)
        inner = S_inv_sqrt @ T @ S_inv_sqrt
        inner = 0.5 * (inner + inner.mT)  # symmetrize
        _, _, log_inner = self._eigh_symmetric_func(inner, torch.log)
        return S_sqrt @ log_inner @ S_sqrt

    def distance(self, S: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Geodesic distance d(S, T) = ‖log(S^{−1/2} T S^{−1/2})‖_F.

        Equivalently d(S, T)² = Σ_i log²(λ_i) where λ_i are generalized
        eigenvalues (T v = λ S v). We compute via the whitened form which
        is well-conditioned for SPD inputs.

        References:
          Pennec et al. (2006), eq. 7.

        Returns:
          (B,) geodesic distances.
        """
        _, S_inv_sqrt = self._sqrt_and_inv_sqrt(S)
        whitened = S_inv_sqrt @ T @ S_inv_sqrt
        whitened = 0.5 * (whitened + whitened.mT)
        eigvals = torch.linalg.eigvalsh(whitened)  # (B, n), all positive
        log_eig = torch.log(eigvals)
        return torch.sqrt((log_eig * log_eig).sum(dim=-1))

    # ----------------------------------------------------------------
    # Retraction (exponential map is the canonical choice on SPD)
    # ----------------------------------------------------------------

    def retraction(self, S: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Retraction = exponential map.

        On SPD with the affine-invariant metric, exp is the canonical
        retraction (it is the geodesic, hence trivially a second-order
        retraction). For a cheaper first-order alternative add a separate
        method (e.g. project S + V onto SPD via eigh + positive-clamp)
        when benchmarks demand it.

        References:
          Absil-Mahony-Sepulchre (2008), §4.1 — exponential as retraction.
        """
        return self.exp(S, V)
