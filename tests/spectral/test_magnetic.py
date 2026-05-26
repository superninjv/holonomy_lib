"""Tests for holonomy_lib.spectral.magnetic.

Four layers:
  1. Validation (square, q in [0, 1)).
  2. q = 0 collapses to the symmetric real Laplacian of A_s.
  3. Hermitian output + real-eigenvalue spectrum (eigh works).
  4. Symmetric input independence of q + bounded spectrum for the
     normalized form.
"""

from __future__ import annotations

import math

import pytest
import torch

from holonomy_lib.spectral import laplacian, magnetic


def _directed_cycle(n: int, batch: int = 1, dtype=torch.float64) -> torch.Tensor:
    A = torch.zeros(batch, n, n, dtype=dtype)
    for i in range(n):
        A[:, i, (i + 1) % n] = 1.0
    return A


def _undirected_cycle(n: int, batch: int = 1, dtype=torch.float64) -> torch.Tensor:
    A = _directed_cycle(n, batch=batch, dtype=dtype)
    return (A + A.mT) * 0.5


# --------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------


class TestValidation:
    def test_rejects_non_square(self):
        with pytest.raises(ValueError, match="must be"):
            magnetic.combinatorial(torch.zeros(1, 4, 5))

    def test_rejects_q_out_of_range(self):
        A = torch.zeros(1, 3, 3, dtype=torch.float64)
        with pytest.raises(ValueError, match="q"):
            magnetic.combinatorial(A, q=-0.1)
        with pytest.raises(ValueError, match="q"):
            magnetic.combinatorial(A, q=1.0)
        with pytest.raises(ValueError, match="q"):
            magnetic.symmetric_normalized(A, q=1.5)

    def test_rejects_complex_input(self):
        """Regression: previously, a complex `A` would crash deep inside
        `torch.complex(cos, sin)` because cos/sin of complex inputs are
        complex. Now we reject up front with a clear ValueError."""
        A = torch.zeros(1, 3, 3, dtype=torch.complex128)
        with pytest.raises(ValueError, match="real"):
            magnetic.combinatorial(A, q=0.25)
        with pytest.raises(ValueError, match="real"):
            magnetic.symmetric_normalized(A, q=0.25)


# --------------------------------------------------------------------
# q = 0 reduces to symmetric Laplacian of A_s
# --------------------------------------------------------------------


class TestZeroCharge:
    """At q = 0 the phase factor is identically 1; the magnetic Laplacian
    reduces to the real-valued Laplacian of the symmetrized adjacency
    A_s = (A + A.T)/2."""

    def test_combinatorial_q_zero_matches_symmetric(self):
        # Directed cycle — has non-trivial asymmetric part.
        A = _directed_cycle(6)
        A_s = (A + A.mT) * 0.5
        L_mag = magnetic.combinatorial(A, q=0.0)
        L_sym = laplacian.combinatorial(A_s).to(L_mag.dtype)
        torch.testing.assert_close(L_mag, L_sym, atol=1e-12, rtol=0)

    def test_symmetric_normalized_q_zero_matches_real(self):
        A = _directed_cycle(7)
        A_s = (A + A.mT) * 0.5
        L_mag = magnetic.symmetric_normalized(A, q=0.0)
        L_real = laplacian.symmetric_normalized(A_s).to(L_mag.dtype)
        torch.testing.assert_close(L_mag, L_real, atol=1e-12, rtol=0)


# --------------------------------------------------------------------
# Hermitian output, real eigenvalues
# --------------------------------------------------------------------


class TestHermitianProperty:
    """The magnetic Laplacian is Hermitian by construction: L = L.conj().T.
    Its eigenvalues are therefore real, and torch.linalg.eigh succeeds."""

    def test_combinatorial_is_hermitian(self):
        A = _directed_cycle(8)
        L = magnetic.combinatorial(A, q=0.25)
        torch.testing.assert_close(
            L, L.conj().mT, atol=1e-12, rtol=0,
        )

    def test_symmetric_normalized_is_hermitian(self):
        A = _directed_cycle(8)
        L = magnetic.symmetric_normalized(A, q=0.25)
        torch.testing.assert_close(
            L, L.conj().mT, atol=1e-12, rtol=0,
        )

    def test_eigenvalues_are_real(self):
        A = _directed_cycle(8)
        L = magnetic.symmetric_normalized(A, q=0.25)
        eigvals = torch.linalg.eigvalsh(L)
        # eigvalsh returns real-valued tensor; assert dtype.
        assert eigvals.dtype in (torch.float32, torch.float64)
        # Spectrum bounded in [0, 2] for normalized form
        # (Furutani 2020, Prop. 1).
        assert (eigvals >= -1e-10).all()
        assert (eigvals <= 2 + 1e-10).all()


# --------------------------------------------------------------------
# Symmetric input: q has no effect
# --------------------------------------------------------------------


class TestSymmetricInput:
    """For symmetric A, A - A.T = 0 → phase is identically 1
    regardless of q. The magnetic Laplacian must agree across q
    values and match the real Laplacian of A."""

    def test_combinatorial_independent_of_q(self):
        A = _undirected_cycle(8)
        L_q0 = magnetic.combinatorial(A, q=0.0)
        L_q25 = magnetic.combinatorial(A, q=0.25)
        L_q5 = magnetic.combinatorial(A, q=0.5)
        torch.testing.assert_close(L_q0, L_q25, atol=1e-12, rtol=0)
        torch.testing.assert_close(L_q0, L_q5, atol=1e-12, rtol=0)

    def test_symmetric_normalized_independent_of_q(self):
        A = _undirected_cycle(7)
        L_q0 = magnetic.symmetric_normalized(A, q=0.0)
        L_q25 = magnetic.symmetric_normalized(A, q=0.25)
        torch.testing.assert_close(L_q0, L_q25, atol=1e-12, rtol=0)


# --------------------------------------------------------------------
# Directed graph: phase actually moves the spectrum
# --------------------------------------------------------------------


class TestDirectedSensitivity:
    """For a non-trivially directed graph, varying q changes the
    Laplacian non-trivially. This pins down that the phase factor
    is actually being applied."""

    def test_q_changes_spectrum_on_directed_cycle(self):
        A = _directed_cycle(6)
        L0 = magnetic.symmetric_normalized(A, q=0.0)
        L25 = magnetic.symmetric_normalized(A, q=0.25)
        # Eigenvalues should differ once q is non-zero on a directed graph.
        eig0 = torch.linalg.eigvalsh(L0).sort().values
        eig25 = torch.linalg.eigvalsh(L25).sort().values
        diff = (eig0 - eig25).abs().max().item()
        assert diff > 1e-3, (
            f"directed cycle's spectrum should shift between q=0 and q=0.25; "
            f"max|Δλ|={diff:.4e}"
        )


# --------------------------------------------------------------------
# Isolated nodes use Moore-Penrose convention
# --------------------------------------------------------------------


class TestIsolatedNodes:
    def test_isolated_node_normalized_does_not_explode(self):
        """An isolated node should not produce NaN/inf in the normalized
        magnetic Laplacian (consistent with the real-Laplacian
        Moore-Penrose convention)."""
        n = 4
        A = torch.zeros(1, n, n, dtype=torch.float64)
        # Triangle 0-1-2 (undirected), node 3 isolated.
        A[0, 0, 1] = A[0, 1, 0] = 1.0
        A[0, 1, 2] = A[0, 2, 1] = 1.0
        A[0, 0, 2] = A[0, 2, 0] = 1.0
        L = magnetic.symmetric_normalized(A, q=0.25)
        assert torch.isfinite(L.real).all()
        assert torch.isfinite(L.imag).all()


# --------------------------------------------------------------------
# Batching shapes
# --------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 2, 4])
class TestShapes:
    def test_combinatorial_shape(self, batch):
        A = _directed_cycle(5, batch=batch)
        L = magnetic.combinatorial(A, q=0.25)
        assert L.shape == (batch, 5, 5)
        assert L.dtype.is_complex

    def test_normalized_shape(self, batch):
        A = _directed_cycle(5, batch=batch)
        L = magnetic.symmetric_normalized(A, q=0.25)
        assert L.shape == (batch, 5, 5)
        assert L.dtype.is_complex
