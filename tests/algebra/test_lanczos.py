"""Tests for holonomy_lib.algebra.lanczos_eigsh.

Layers:
  1. Validation (k bounds, square input).
  2. Convergence against dense torch.linalg.eigh on small matrices.
  3. Eigenvector orthonormality.
  4. Diagonal-matrix edge case (exact convergence in m = n steps).
  5. Batching shape contract.
"""

from __future__ import annotations

import pytest
import torch

from holonomy_lib.algebra import lanczos_eigsh


def _seeded(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _random_symmetric(
    n: int, batch: int = 1, dtype=torch.float64, seed: int = 0,
) -> torch.Tensor:
    g = _seeded(seed)
    A = torch.randn(batch, n, n, dtype=dtype, generator=g)
    return 0.5 * (A + A.mT)


# --------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------


class TestValidation:
    def test_rejects_non_square(self):
        with pytest.raises(ValueError, match="must be"):
            lanczos_eigsh(torch.zeros(1, 4, 5), k=2)

    def test_rejects_k_zero(self):
        A = _random_symmetric(5)
        with pytest.raises(ValueError, match="k must"):
            lanczos_eigsh(A, k=0)

    def test_rejects_k_too_large(self):
        A = _random_symmetric(5)
        with pytest.raises(ValueError, match="k must"):
            lanczos_eigsh(A, k=6)

    def test_rejects_n_iter_less_than_k(self):
        A = _random_symmetric(8)
        with pytest.raises(ValueError, match="n_iter"):
            lanczos_eigsh(A, k=4, n_iter=3, oversample=0)


# --------------------------------------------------------------------
# Convergence vs dense eigh
# --------------------------------------------------------------------


class TestConvergence:
    @pytest.mark.parametrize("n", [10, 25, 50])
    def test_full_iter_matches_exact(self, n):
        """With `n_iter = n` Lanczos builds the full Krylov subspace and
        is exact (up to rounding) for the top-1 eigenvalue. This pins
        down that the algorithm is correct; `test_partial_iter` below
        characterizes the typical-use convergence rate."""
        A = _random_symmetric(n, seed=42)
        ritz, _ = lanczos_eigsh(
            A, k=1, n_iter=n, oversample=0, generator=_seeded(0),
        )
        ref = torch.linalg.eigvalsh(A).flip(dims=(-1,))[..., :1]
        torch.testing.assert_close(
            ritz, ref, atol=1e-9, rtol=1e-9,
        )

    def test_partial_iter_converges_with_oversample(self):
        """With modest oversampling on a moderately-sized matrix the
        top-1 eigenvalue converges to a useful tolerance. Per Saad
        (2011) §6.7 the convergence is geometric in the spectral gap;
        we use a generous oversample so the test is robust across
        random starts."""
        n = 50
        A = _random_symmetric(n, seed=42)
        ritz, _ = lanczos_eigsh(
            A, k=1, n_iter=n // 2, oversample=0, generator=_seeded(0),
        )
        ref = torch.linalg.eigvalsh(A).flip(dims=(-1,))[..., :1]
        max_diff = (ritz - ref).abs().max().item()
        assert max_diff < 1e-3, (
            f"n=50, n_iter=25 top-1 diff {max_diff:.3e} should be < 1e-3"
        )

    def test_top_eigenvalue_residual_full_iter(self):
        """At `n_iter = n` the residual `‖A v − λ v‖` is machine-precision
        small (Lanczos has spanned the full eigenspace by then)."""
        n = 20
        A = _random_symmetric(n, seed=99)
        vals, vecs = lanczos_eigsh(
            A, k=1, n_iter=n, oversample=0, generator=_seeded(7),
        )
        Av = torch.matmul(A, vecs)
        residual = Av - vecs * vals.unsqueeze(dim=-2)
        max_res = residual.norm(dim=-2).max()
        assert max_res.item() < 1e-9, (
            f"full-iter top-1 residual {max_res.item():.3e} too large"
        )


# --------------------------------------------------------------------
# Orthonormality
# --------------------------------------------------------------------


class TestOrthonormality:
    def test_eigenvectors_orthonormal(self):
        """Returned eigenvectors should be orthonormal (within tolerance)."""
        A = _random_symmetric(15, seed=11)
        _, vecs = lanczos_eigsh(
            A, k=4, oversample=15, generator=_seeded(2),
        )
        # vecs^T vecs should ≈ I_k
        gram = torch.matmul(vecs.mT, vecs)
        k = vecs.shape[-1]
        I_k = torch.eye(k, dtype=vecs.dtype).unsqueeze(0)
        torch.testing.assert_close(gram, I_k, atol=1e-6, rtol=0)


# --------------------------------------------------------------------
# Diagonal matrix: exact case
# --------------------------------------------------------------------


class TestDiagonalCase:
    def test_recovers_top_diagonal_entries(self):
        """For a diagonal matrix A = diag(λ_1, ..., λ_n), the largest
        eigenvalues are the largest diagonal entries. With sufficient
        Lanczos iterations Lanczos should find them exactly (up to
        rounding) — provided the starting vector overlaps each
        eigenvector, which it does for a random Gaussian start with
        probability 1.
        """
        n = 12
        # Distinct, well-separated diagonal entries (already descending).
        diag = torch.tensor(
            [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.1],
            dtype=torch.float64,
        )
        A = torch.diag(diag).unsqueeze(0)
        ritz, _ = lanczos_eigsh(
            A, k=4, n_iter=n, oversample=0, generator=_seeded(5),
        )
        # `diag` is already in descending order; the top-4 are the
        # first four entries.
        expected = diag[:4]
        torch.testing.assert_close(
            ritz[0], expected, atol=1e-8, rtol=0,
        )


# --------------------------------------------------------------------
# Shapes / batching
# --------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 2, 4])
class TestShapes:
    def test_eigvals_shape(self, batch):
        A = _random_symmetric(15, batch=batch, seed=20)
        vals, _ = lanczos_eigsh(A, k=3, generator=_seeded(0))
        assert vals.shape == (batch, 3)

    def test_eigvecs_shape(self, batch):
        A = _random_symmetric(15, batch=batch, seed=21)
        _, vecs = lanczos_eigsh(A, k=3, generator=_seeded(0))
        assert vecs.shape == (batch, 15, 3)
