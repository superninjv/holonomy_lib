"""Tests for holonomy_lib.lie.so3.

Cover:
  1. Validation (shape mismatches).
  2. Rodrigues formula correctness:
     - R^T R = I, det R = +1.
     - Angle 0 → identity.
     - Angle π around the canonical axes → known matrices.
  3. axis_angle ↔ matrix round-trip (mod sign ambiguity at θ=π).
  4. so3_exp / so3_log round-trip.
  5. random_so3:
     - Output is in SO(3).
     - Haar-uniformity sanity: large-batch mean rotation matrix ≈ 0
       (the trivial-rep projection on SO(3) under Haar is zero
       for the standard 3D representation).
"""

from __future__ import annotations

import math

import pytest
import torch

from holonomy_lib.lie import so3


def _seeded(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# --------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------


class TestValidation:
    def test_axis_angle_rejects_bad_axis_dim(self):
        with pytest.raises(ValueError, match="dim 3"):
            so3.axis_angle_to_matrix(
                torch.zeros(5, 4, dtype=torch.float64),
                torch.zeros(5, dtype=torch.float64),
            )

    def test_axis_angle_rejects_batch_mismatch(self):
        with pytest.raises(ValueError, match="batch"):
            so3.axis_angle_to_matrix(
                torch.zeros(5, 3, dtype=torch.float64),
                torch.zeros(4, dtype=torch.float64),
            )

    def test_matrix_to_axis_angle_rejects_bad_shape(self):
        with pytest.raises(ValueError, match="\\(3, 3\\)"):
            so3.matrix_to_axis_angle(torch.zeros(2, 3, dtype=torch.float64))

    def test_compose_rejects_bad_shape(self):
        with pytest.raises(ValueError, match="\\(3, 3\\)"):
            so3.compose(
                torch.zeros(3, 3, dtype=torch.float64),
                torch.zeros(2, 2, dtype=torch.float64),
            )

    def test_random_so3_rejects_negative_batch(self):
        with pytest.raises(ValueError, match="batch_size"):
            so3.random_so3(batch_size=-1)


# --------------------------------------------------------------------
# Rodrigues formula correctness
# --------------------------------------------------------------------


class TestRodriguesFormula:
    def test_output_is_in_SO3(self):
        """Random axis-angle → R^T R = I, det R = +1."""
        g = _seeded(0)
        axis = torch.randn(8, 3, dtype=torch.float64, generator=g)
        angle = torch.rand(8, dtype=torch.float64, generator=g) * math.pi
        R = so3.axis_angle_to_matrix(axis, angle)
        I = torch.eye(3, dtype=torch.float64).expand(8, 3, 3)
        torch.testing.assert_close(R @ R.mT, I, atol=1e-12, rtol=0)
        det = torch.linalg.det(R)
        torch.testing.assert_close(
            det, torch.ones(8, dtype=torch.float64), atol=1e-12, rtol=0,
        )

    def test_zero_angle_gives_identity(self):
        axis = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        angle = torch.zeros(1, dtype=torch.float64)
        R = so3.axis_angle_to_matrix(axis, angle)
        I = torch.eye(3, dtype=torch.float64).unsqueeze(0)
        torch.testing.assert_close(R, I, atol=1e-12, rtol=0)

    def test_pi_around_z_axis(self):
        """π rotation around z = diag(-1, -1, 1)."""
        axis = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        angle = torch.tensor([math.pi], dtype=torch.float64)
        R = so3.axis_angle_to_matrix(axis, angle)
        expected = torch.diag(torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float64))
        torch.testing.assert_close(R[0], expected, atol=1e-10, rtol=0)

    def test_pi_over_2_around_x_axis(self):
        """π/2 around x sends e_y → e_z, e_z → -e_y."""
        axis = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        angle = torch.tensor([math.pi / 2.0], dtype=torch.float64)
        R = so3.axis_angle_to_matrix(axis, angle).squeeze(0)
        # R @ e_y = R[:, 1]
        torch.testing.assert_close(
            R[:, 1], torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            atol=1e-10, rtol=0,
        )
        torch.testing.assert_close(
            R[:, 2], torch.tensor([0.0, -1.0, 0.0], dtype=torch.float64),
            atol=1e-10, rtol=0,
        )

    def test_unnormalized_axis_works(self):
        """Caller can pass non-unit axis; rotation is the same."""
        unit = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        scaled = torch.tensor([[0.0, 0.0, 5.0]], dtype=torch.float64)
        angle = torch.tensor([math.pi / 3.0], dtype=torch.float64)
        R_unit = so3.axis_angle_to_matrix(unit, angle)
        R_scaled = so3.axis_angle_to_matrix(scaled, angle)
        torch.testing.assert_close(R_unit, R_scaled, atol=1e-12, rtol=0)


# --------------------------------------------------------------------
# axis_angle ↔ matrix round-trip
# --------------------------------------------------------------------


class TestRoundTrip:
    def test_axis_angle_roundtrip_general(self):
        """Random axis/angle → matrix → axis/angle recovers the same
        rotation (axis may be sign-flipped together with angle)."""
        g = _seeded(2)
        n = 32
        axis = torch.randn(n, 3, dtype=torch.float64, generator=g)
        axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)
        # Angles strictly inside (0, π) to avoid edge cases.
        angle = (
            torch.rand(n, dtype=torch.float64, generator=g) * (math.pi - 0.1)
            + 0.05
        )
        R = so3.axis_angle_to_matrix(axis, angle)
        axis_back, angle_back = so3.matrix_to_axis_angle(R)
        # Reconstruct R from the recovered axis/angle; it must match.
        R_back = so3.axis_angle_to_matrix(axis_back, angle_back)
        torch.testing.assert_close(R, R_back, atol=1e-10, rtol=0)

    def test_so3_exp_log_roundtrip(self):
        """log(exp(ω)) = ω for ‖ω‖ < π."""
        g = _seeded(3)
        # Random ω with magnitude < π (the injectivity radius of so(3)).
        omega = torch.randn(16, 3, dtype=torch.float64, generator=g)
        omega_norm = torch.linalg.norm(omega, dim=-1, keepdim=True)
        scale = torch.clamp(omega_norm, max=math.pi - 0.1) / omega_norm
        omega = omega * scale
        R = so3.so3_exp(omega)
        omega_back = so3.so3_log(R)
        torch.testing.assert_close(omega, omega_back, atol=1e-9, rtol=0)

    def test_so3_exp_zero_is_identity(self):
        omega = torch.zeros(3, 3, dtype=torch.float64)
        R = so3.so3_exp(omega)
        I = torch.eye(3, dtype=torch.float64).expand(3, 3, 3)
        torch.testing.assert_close(R, I, atol=1e-12, rtol=0)


# --------------------------------------------------------------------
# random_so3
# --------------------------------------------------------------------


class TestRandomSO3:
    def test_output_is_in_SO3(self):
        n = 128
        R = so3.random_so3(batch_size=n, generator=_seeded(0))
        I = torch.eye(3, dtype=torch.float64).expand(n, 3, 3)
        torch.testing.assert_close(R @ R.mT, I, atol=1e-12, rtol=0)
        det = torch.linalg.det(R)
        torch.testing.assert_close(
            det, torch.ones(n, dtype=torch.float64), atol=1e-12, rtol=0,
        )

    def test_haar_uniform_mean_approaches_zero(self):
        """Under Haar measure on SO(3), E[R] = 0 (the standard 3D
        representation has zero projection onto the trivial rep).
        A large batch should give a mean rotation matrix with
        Frobenius norm decaying like 1/√N."""
        n = 10000
        R = so3.random_so3(batch_size=n, generator=_seeded(7))
        mean_R = R.mean(dim=0)
        # 1/√10000 = 0.01; total Frobenius norm of mean across 9 entries
        # should be small. Use a generous bound.
        fro = torch.linalg.matrix_norm(mean_R, ord="fro").item()
        assert fro < 0.1, f"mean rotation Frobenius norm {fro} too large"

    def test_deterministic_with_seed(self):
        R1 = so3.random_so3(batch_size=4, generator=_seeded(42))
        R2 = so3.random_so3(batch_size=4, generator=_seeded(42))
        torch.testing.assert_close(R1, R2, atol=1e-12, rtol=0)

    def test_zero_batch_size(self):
        R = so3.random_so3(batch_size=0, generator=_seeded(0))
        assert R.shape == (0, 3, 3)


# --------------------------------------------------------------------
# Group law
# --------------------------------------------------------------------


class TestCompose:
    def test_identity_left_and_right(self):
        n = 4
        R = so3.random_so3(batch_size=n, generator=_seeded(0))
        I = torch.eye(3, dtype=torch.float64).expand(n, 3, 3)
        torch.testing.assert_close(so3.compose(I, R), R, atol=1e-12, rtol=0)
        torch.testing.assert_close(so3.compose(R, I), R, atol=1e-12, rtol=0)

    def test_inverse_via_transpose(self):
        """For R ∈ SO(3), R · R^T = I."""
        n = 4
        R = so3.random_so3(batch_size=n, generator=_seeded(1))
        I = torch.eye(3, dtype=torch.float64).expand(n, 3, 3)
        torch.testing.assert_close(
            so3.compose(R, R.mT), I, atol=1e-12, rtol=0,
        )
